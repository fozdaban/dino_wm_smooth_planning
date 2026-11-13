"""Phase 6b: GD save_restarts and RescorePlanner on the fake world model (CPU, seconds).

Run from the repo root:  python tests/test_rescore_fake_wm.py
"""
import os
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gd_fake_wm import A, H, FakeWM, episodes, make, objective_fn, plan  # noqa: E402

from planning.compute import ComputeMeter  # noqa: E402
from planning.gd import GDPlanner  # noqa: E402
from planning.rescore import RescorePlanner  # noqa: E402

THRESH = 1.0  # fake "simulator": success iff the plan's plain cost is below this


class FakeEvaluator:
    def __init__(self, wm, obs_0, obs_g):
        self.wm, self.obs_0, self.obs_g, self.calls = wm, obs_0, obs_g, []

    def eval_actions(self, actions, action_len=None, filename="output", save_video=False):
        with torch.no_grad():
            c = objective_fn(self.wm.rollout(self.obs_0, actions)[0], self.obs_g)
        self.calls.append(filename)
        return {}, (c < THRESH).numpy(), None, None


def run_in(tmp, name, fn):
    """Run fn() with cwd = tmp/name (planners write their files to the cwd)."""
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    old = os.getcwd()
    os.chdir(d)
    try:
        return d, fn()
    finally:
        os.chdir(old)


def test_save_restarts_does_not_change_the_plan():
    """save_restarts=True writes (n, R, H, A) and returns the same plan as without it;
    the returned plan is the lowest-cost restart."""
    n, R = 6, 4
    obs_0, obs_g = episodes(n)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(restarts=R, opt_steps=15, optimizer="adam", lr=0.05, episode_chunk=5)
        _, (a0, _) = run_in(tmp, "plain", lambda: plan(make(GDPlanner, **kw), obs_0, obs_g, seed=2))
        d, (a1, _) = run_in(tmp, "saved", lambda: plan(make(GDPlanner, save_restarts=True, **kw),
                                                       obs_0, obs_g, seed=2))
        assert torch.equal(a0, a1)
        assert not os.path.exists(os.path.join(tmp, "plain", "restart_actions.pt"))
        ra = torch.load(os.path.join(d, "restart_actions.pt"))
        assert tuple(ra.shape) == (n, R, H, A), ra.shape
        with torch.no_grad():
            costs = torch.stack([objective_fn(FakeWM().rollout(obs_0, ra[:, j])[0], obs_g)
                                 for j in range(R)], 1)
        best = ra[torch.arange(n), costs.argmin(1)]
        assert torch.allclose(best, a1), (best - a1).abs().max()


def test_rescore():
    """Candidates, J at sigma 0 = plain cost, J_sigma = mean over the fixed eps,
    successes from the evaluator, plan() returns candidate 0, and no gradients."""
    n, R, M = 7, 3, 16
    obs_0, obs_g = episodes(n)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(restarts=R, opt_steps=10, optimizer="adam", lr=0.05, save_restarts=True)
        r_dir, _ = run_in(tmp, "r3", lambda: plan(make(GDPlanner, **kw), obs_0, obs_g, seed=1))
        single = torch.randn(n, H, A, generator=torch.Generator().manual_seed(5))
        s_dir = os.path.join(tmp, "single")
        os.makedirs(s_dir)
        torch.save(single, os.path.join(s_dir, "final_actions.pt"))

        wm = FakeWM()
        ev = FakeEvaluator(wm, obs_0, obs_g)
        meter = ComputeMeter().attach(wm).wrap_evaluator(ev)
        sigmas = [0.0, 0.5]
        p = RescorePlanner(horizon=H, wm=wm, action_dim=A, objective_fn=objective_fn,
                           preprocessor=types.SimpleNamespace(transform_obs=lambda o: o),
                           evaluator=ev, wandb_run=None, runs=[s_dir, r_dir],
                           sigmas=sigmas, M=M, chunk=10, seed=0)
        out_dir, (ret, _) = run_in(tmp, "rescore", lambda: p.plan(obs_0, obs_g))
        st = meter.stats(0.0)  # before the checks below, which also call the model
        d = np.load(os.path.join(out_dir, "rescore.npz"))
        labels = list(d["labels"])
        assert labels == ["single", "r3#r0", "r3#r1", "r3#r2"], labels
        assert list(d["groups"]) == ["single", "r3", "r3", "r3"]
        assert torch.equal(ret, single)
        ra = torch.load(os.path.join(r_dir, "restart_actions.pt"))
        cands = [single] + [ra[:, j] for j in range(R)]
        eps = torch.randn(M, H, A, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            for c, a in enumerate(cands):
                j0 = objective_fn(wm.rollout(obs_0, a)[0], obs_g)
                assert np.allclose(d["J"][c, 0], j0.numpy(), rtol=1e-5), c
                rows = (a[:, None] + 0.5 * eps[None]).reshape(n * M, H, A)
                o0 = {k: v.repeat_interleave(M, 0) for k, v in obs_0.items()}
                og = {k: v.repeat_interleave(M, 0) for k, v in obs_g.items()}
                js = objective_fn(wm.rollout(o0, rows)[0], og).view(n, M).mean(1)
                assert np.allclose(d["J"][c, 1], js.numpy(), rtol=1e-4), c
                assert np.array_equal(d["success"][c], (j0 < THRESH).numpy()), c
        assert ev.calls == [f"cand_{l}" for l in labels], ev.calls
        assert st["plan_fwd_grad"] == 0, st
        assert st["plan_fwd"] == len(cands) * n * (1 + M), st


def test_rescore_report_runs():
    """The report script runs on a rescore.npz and finds the pre-declared rule."""
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        n, C = 20, 5
        np.savez(os.path.join(tmp, "rescore.npz"),
                 labels=np.array(["x#r0", "x#r1", "x#r2", "x#r3", "y"]),
                 groups=np.array(["x", "x", "x", "x", "y"]),
                 sigmas=np.array([0, 1.0], np.float32),
                 J=rng.random((C, 2, n)).astype(np.float32),
                 success=rng.random((C, n)) < 0.4, evaluated=np.array(True), M=np.array(8))
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, os.path.join(root, "scripts", "rescore_report.py"),
                              tmp, "--set", "pair=x#r0,y"], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        for s in ("## 1.", "## 2.", "## 3.", "| pair |", "x (its 4 restarts)", "all candidates"):
            assert s in out.stdout, s


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
