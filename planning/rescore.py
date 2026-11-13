"""Phase 6b, experiment E3: score saved final plans with the smoothed cost (no planning).

Question: does the smoothed cost J_sigma(a) = mean_m J(a + sigma eps_m) of a FINAL plan
predict simulator success better than the raw cost J(a)? Plans that fool the model should
sit in sharp, narrow minima: low J, but J_sigma rises quickly with sigma. If so, choosing
among candidate plans (e.g. GD restarts, or the smoothed and the restarts planner's plans)
by J_sigma instead of J should solve more episodes.

Candidates come from earlier runs on the SAME seed / n_evals / goal settings:
  a run folder with restart_actions.pt (GD with +planner.save_restarts=true) gives one
  candidate per restart, labelled <run>#r<j>; any other run gives its final_actions.pt,
  labelled <run>.
For every candidate c, sigma s and episode e it computes J[c, s, e] with M perturbations
drawn once (seeded) and shared by every candidate, sigma and episode (common random
numbers: differences between candidates are not Monte-Carlo noise), and the simulator
success of every candidate via the evaluator (one eval_actions call per candidate,
filename "cand_<label>" in episodes.jsonl). Saved to rescore.npz in the run directory.

plan() returns candidate 0, so the run's final evaluation must reproduce candidate 0's
success: a built-in check that seeds, goals and actions line up.
Analysis: scripts/rescore_report.py.
"""
import os
import pickle

import numpy as np
import torch

from .base_planner import BasePlanner
from .probe import _resolve
from utils import move_to_device


class RescorePlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        runs,
        sigmas=(0.0, 0.1, 0.33, 1.0),
        M=64,
        chunk=100,
        seed=0,
        evaluate=True,
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run, log_filename
        )
        self.horizon = horizon
        self.runs = [_resolve(r) for r in runs]
        self.sigmas = [float(s) for s in sigmas]
        self.M = int(M)
        self.chunk = int(chunk)
        self.seed = int(seed)
        self.evaluate = bool(evaluate)
        assert self.runs, "planner.runs is empty"
        assert all(s >= 0 for s in self.sigmas), self.sigmas
        assert self.M >= 1 and self.chunk >= 1

    def _candidates(self, n_evals):
        """[(label, run name, (n, H, A) tensor)] in the order of self.runs."""
        expect = (n_evals, self.horizon, self.action_dim)
        out = []
        for run in self.runs:
            name = os.path.basename(os.path.normpath(run))
            ra = os.path.join(run, "restart_actions.pt")
            if os.path.exists(ra):
                acts = torch.load(ra, map_location="cpu")  # (n, R, H, A)
                assert acts.shape[0] == n_evals and tuple(acts.shape[2:]) == expect[1:], (
                    f"{ra}: shape {tuple(acts.shape)}, expected ({n_evals}, R, {expect[1]}, {expect[2]})"
                )
                for j in range(acts.shape[1]):
                    out.append((f"{name}#r{j}", name, acts[:, j].contiguous()))
            else:
                fa = os.path.join(run, "final_actions.pt")
                acts = torch.load(fa, map_location="cpu")
                assert tuple(acts.shape) == expect, f"{fa}: shape {tuple(acts.shape)} != {expect}"
                out.append((name, name, acts))
        return out

    def _check_targets(self, obs_0, obs_g):
        """If a run kept plan_targets.pkl, its start/goal images must equal ours."""
        for run in self.runs:
            path = os.path.join(run, "plan_targets.pkl")
            if not os.path.exists(path):
                print(f"rescore: {path} not found; cannot check the episodes match")
                continue
            with open(path, "rb") as f:
                t = pickle.load(f)
            for key, mine, theirs in (("obs_0", obs_0, t["obs_0"]), ("obs_g", obs_g, t["obs_g"])):
                assert np.array_equal(np.asarray(mine["visual"]), np.asarray(theirs["visual"])), (
                    f"{key} differs from {path}: use the same seed, n_evals, goal_source, goal_H"
                )

    def _costs(self, acts, ep_idx, trans_obs_0, z_g):
        """No-grad cost of each row of acts (rows, H, A); row r belongs to episode ep_idx[r]."""
        out = []
        with torch.no_grad():
            for s in range(0, acts.shape[0], self.chunk):
                idx = ep_idx[s : s + self.chunk]
                z, _ = self.wm.rollout(
                    obs_0={k: v[idx] for k, v in trans_obs_0.items()}, act=acts[s : s + self.chunk]
                )
                out.append(self.objective_fn(z, {k: v[idx] for k, v in z_g.items()}))
        return torch.cat(out)

    def _smoothed(self, a, sigma, eps, trans_obs_0, z_g):
        """J_sigma for every episode of plan a (n, H, A); plain J for sigma = 0."""
        n = a.shape[0]
        if sigma == 0:
            return self._costs(a, torch.arange(n, device=self.device), trans_obs_0, z_g)
        M = eps.shape[0]
        rows = (a[:, None] + sigma * eps[None]).reshape(n * M, *a.shape[1:])  # episode-major
        ep_idx = torch.arange(n, device=self.device).repeat_interleave(M)
        return self._costs(rows, ep_idx, trans_obs_0, z_g).view(n, M).mean(1)

    def plan(self, obs_0, obs_g, actions=None):
        n_evals = obs_0["visual"].shape[0]
        cands = self._candidates(n_evals)
        self._check_targets(obs_0, obs_g)
        print(f"rescore: {len(cands)} candidates: {[c[0] for c in cands]}")

        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        with torch.no_grad():
            z_g = {k: v.detach() for k, v in self.wm.encode_obs(trans_obs_g).items()}

        gen = torch.Generator().manual_seed(self.seed)
        eps = torch.randn(self.M, self.horizon, self.action_dim, generator=gen).to(self.device)

        C, S = len(cands), len(self.sigmas)
        J = np.zeros((C, S, n_evals), np.float32)
        success = np.zeros((C, n_evals), bool)
        for c, (label, _, a) in enumerate(cands):
            a = a.to(self.device)
            for s, sigma in enumerate(self.sigmas):
                J[c, s] = self._smoothed(a, sigma, eps, trans_obs_0, z_g).float().cpu().numpy()
            if self.evaluate and self.evaluator is not None:
                _, succ, _, _ = self.evaluator.eval_actions(a.detach(), filename=f"cand_{label}")
                success[c] = np.asarray(succ, dtype=bool)
            print(f"rescore: {label}: J(sigma=0) mean {J[c, 0].mean():.3g}"
                  + (f", success {success[c].mean():.2f}" if self.evaluate else ""))

        np.savez(
            "rescore.npz",
            labels=np.array([c[0] for c in cands]),
            groups=np.array([c[1] for c in cands]),
            sigmas=np.array(self.sigmas, np.float32),
            J=J,
            success=success,
            evaluated=np.array(self.evaluate),
            M=np.array(self.M),
        )
        return cands[0][2].to(self.device), np.full(n_evals, np.inf)
