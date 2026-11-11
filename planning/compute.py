"""Hardware-independent compute counters for planning (phase 2 harness).

Counts work done through the world model by the *planner* only. Calls made while the
evaluator runs (simulator rollouts, decoding, plotting, the final evaluation) are
excluded: ComputeMeter.paused() is wrapped around every evaluator call in plan.py.

Units (main one first; it matches roadmap.md: CEM@10 = 3,000 per episode, GD@400 = 1,200)
  plan_fwd          plan-rollouts: one wm.rollout of a batch of b plans counts b (one plan
                    pushed through all horizon model steps).
  plan_fwd_grad     the subset of plan_fwd run with autograd on the actions; these are
                    backpropagated, and a backward pass costs about 2 forward passes.
  plan_equiv        plan_fwd + 2 * plan_fwd_grad: the x-axis of the success-vs-compute plot.
  enc_images        images encoded by DINOv2 (wm.encode_obs). Every wm.rollout call
                    re-encodes obs_0, so CEM encodes num_samples images per iteration
                    per episode; this is upstream behaviour and counts as planner cost.
  pred_fwd          predictor sample-steps: one predictor call on a batch of b samples
                    counts b (each sample sees num_hist frames of patch tokens).
  pred_fwd_grad     the subset of pred_fwd run with autograd tracking (inputs require
                    grad). These are backpropagated by gradient planners; a backward
                    pass costs roughly 2x its forward pass.
  pred_equiv        pred_fwd + 2 * pred_fwd_grad, at predictor-step granularity (a cross-check
                    on plan_equiv; equals horizon x plan_equiv for fixed-horizon planners).
"""
import time
from contextlib import contextmanager

import torch


class ComputeMeter:
    def __init__(self):
        self.plan_fwd = 0
        self.plan_fwd_grad = 0
        self.enc_images = 0
        self.pred_fwd = 0
        self.pred_fwd_grad = 0
        self.eval_s = 0.0
        self.eval_calls = 0
        self._paused = 0

    # ---- counting -------------------------------------------------------
    def attach(self, wm):
        """Wrap wm.rollout, wm.encode_obs and wm.predict (instance attributes shadow the
        methods, and rollout calls self.encode_obs / self.predict, so all are seen)."""
        rollout, encode_obs, predict = wm.rollout, wm.encode_obs, wm.predict

        def counted_rollout(obs_0, act, *args, **kwargs):
            if not self._paused:
                b = act.shape[0]
                self.plan_fwd += b
                if torch.is_grad_enabled() and act.requires_grad:
                    self.plan_fwd_grad += b
            return rollout(obs_0, act, *args, **kwargs)

        def counted_encode_obs(obs):
            if not self._paused:
                v = obs["visual"]
                self.enc_images += v.shape[0] * v.shape[1]
            return encode_obs(obs)

        def counted_predict(z):
            if not self._paused:
                b = z.shape[0]
                self.pred_fwd += b
                if torch.is_grad_enabled() and z.requires_grad:
                    self.pred_fwd_grad += b
            return predict(z)

        wm.rollout = counted_rollout
        wm.encode_obs = counted_encode_obs
        wm.predict = counted_predict
        return self

    @contextmanager
    def paused(self):
        self._paused += 1
        try:
            yield
        finally:
            self._paused -= 1

    # ---- evaluator timing ----------------------------------------------
    def wrap_evaluator(self, evaluator):
        """Exclude evaluator calls from compute counts and time them separately."""
        eval_actions = evaluator.eval_actions

        def timed_eval_actions(*args, **kwargs):
            with self.paused():
                _sync()
                t0 = time.time()
                out = eval_actions(*args, **kwargs)
                _sync()
                self.eval_s += time.time() - t0
                self.eval_calls += 1
            return out

        evaluator.eval_actions = timed_eval_actions
        return self

    def stats(self, wall_clock_s):
        return {
            "plan_fwd": self.plan_fwd,
            "plan_fwd_grad": self.plan_fwd_grad,
            "plan_equiv": self.plan_fwd + 2 * self.plan_fwd_grad,
            "enc_images": self.enc_images,
            "pred_fwd": self.pred_fwd,
            "pred_fwd_grad": self.pred_fwd_grad,
            "pred_equiv": self.pred_fwd + 2 * self.pred_fwd_grad,
            "eval_s": self.eval_s,
            "eval_calls": self.eval_calls,
            "plan_s": wall_clock_s - self.eval_s,
        }


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
