"""GD / SmoothGD planners on a small fake world model (CPU, seconds).

Run from the repo root:  python tests/test_gd_fake_wm.py
Prints PASS/FAIL per test; exit code 1 if any test fails.
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.compute import ComputeMeter  # noqa: E402
from planning.gd import GDPlanner  # noqa: E402
from planning.smooth_gd import SmoothGDPlanner  # noqa: E402

H, A, D = 5, 3, 4  # horizon, action dim, latent dim


class FakeWM(torch.nn.Module):
    """z_t = z_{t-1} + a_t B + amp * sin(freq * a_t C); latents are the "visual" key.

    amp = 0 makes the final latent linear in the actions, so the cost below is quadratic;
    amp > 0 gives a bumpy (non-convex) cost.
    """

    def __init__(self, amp=0.5, freq=4.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(A, D, generator=g))
        self.register_buffer("C", torch.randn(A, D, generator=g))
        self.amp, self.freq = amp, freq
        self._p = torch.nn.Parameter(torch.zeros(1))  # BasePlanner reads the device from here

    def encode_obs(self, obs):
        return {"visual": obs["visual"]}

    def predict(self, z):  # only here so ComputeMeter.attach finds it
        return z

    def rollout(self, obs_0, act):
        z = obs_0["visual"][:, -1]
        zs = [z]
        for t in range(act.shape[1]):
            a = act[:, t]
            z = z + a @ self.B + self.amp * torch.sin(self.freq * (a @ self.C))
            zs.append(z)
        return {"visual": torch.stack(zs, 1)}, None


def objective_fn(z, z_g):
    """Squared distance between the final latent and the goal, per row: (b,)."""
    return ((z["visual"][:, -1] - z_g["visual"][:, -1]) ** 2).sum(-1)


def make(cls=GDPlanner, wm=None, **kw):
    """Build a planner on the fake model; kw override the defaults below."""
    cfg = dict(horizon=H, action_noise=0.0, sample_type="randn", lr=0.1, opt_steps=10,
               eval_every=1, optimizer="sgd", restarts=1, init_scale=1.0, episode_chunk=10)
    if cls is SmoothGDPlanner:
        cfg.update(K=8, sigma0=0.3, sigma_min=0.03, sigma_schedule="geom", antithetic=True)
    cfg.update(kw)
    return cls(wm=wm if wm is not None else FakeWM(), action_dim=A, objective_fn=objective_fn,
               preprocessor=types.SimpleNamespace(transform_obs=lambda o: o),
               evaluator=None, wandb_run=types.SimpleNamespace(log=lambda *a, **k: None), **cfg)


def episodes(n, seed=0):
    g = torch.Generator().manual_seed(1000 + seed)
    return ({"visual": torch.randn(n, 1, D, generator=g)},
            {"visual": 3 * torch.randn(n, 1, D, generator=g)})


def plan(planner, obs_0, obs_g, actions=None, seed=0, meter=None):
    """Seeded planner.plan; returns (final plan, its model cost per episode).
    Pass the attached ComputeMeter so the cost rollout here is not counted."""
    torch.manual_seed(seed)
    out, _ = planner.plan(obs_0, obs_g, None if actions is None else actions.clone())
    out = out.detach()
    with torch.no_grad():
        if meter is not None:
            with meter.paused():
                z = planner.wm.rollout(obs_0, out)[0]
        else:
            z = planner.wm.rollout(obs_0, out)[0]
        cost = objective_fn(z, obs_g)
    return out, cost


# ---------------------------------------------------------------- SmoothGD tests

def test_smooth_k1_sigma0_equals_gd():
    """K=1, sigma=0: bit-for-bit the GD plan, for sgd and adam."""
    obs_0, obs_g = episodes(6)
    smooth0 = dict(K=1, sigma0=0.0, sigma_min=0.0, sigma_schedule="const", antithetic=False)
    for opt, lr in (("sgd", 0.1), ("adam", 0.05)):
        common = dict(optimizer=opt, lr=lr, opt_steps=20, action_noise=0.003, episode_chunk=4)
        a_gd, _ = plan(make(GDPlanner, **common), obs_0, obs_g, seed=3)
        a_sm, _ = plan(make(SmoothGDPlanner, **common, **smooth0), obs_0, obs_g, seed=3)
        assert torch.equal(a_gd, a_sm), f"{opt}: max diff {(a_gd - a_sm).abs().max().item()}"


def test_smooth_compute_counts():
    """plan_fwd == plan_fwd_grad == n*K*S, plan_equiv == 3*n*K*S."""
    n, K, S = 5, 4, 3
    obs_0, obs_g = episodes(n)
    wm = FakeWM()
    meter = ComputeMeter().attach(wm)
    plan(make(SmoothGDPlanner, wm=wm, K=K, opt_steps=S, episode_chunk=7), obs_0, obs_g,
         meter=meter)
    st = meter.stats(0.0)
    assert st["plan_fwd"] == st["plan_fwd_grad"] == n * K * S, st
    assert st["plan_equiv"] == 3 * n * K * S, st


def test_smooth_quadratic_gradient():
    """Quadratic cost: the K=4096 averaged gradient at a fixed plan matches the exact one.

    One sgd step with lr=1 and no action noise gives grad = a0 - a1. With independent
    (non-antithetic) perturbations the estimate is Monte-Carlo: error ~ sigma*|Hess eps|/sqrt(K).
    """
    n, K, sigma = 3, 4096, 0.3
    wm = FakeWM(amp=0.0)  # final latent linear in actions -> quadratic cost
    obs_0, obs_g = episodes(n)
    a0 = torch.randn(n, H, A, generator=torch.Generator().manual_seed(7))
    p = make(SmoothGDPlanner, wm=wm, K=K, sigma0=sigma, sigma_schedule="const",
             antithetic=False, opt_steps=1, lr=1.0, episode_chunk=4096)
    a1, _ = plan(p, obs_0, obs_g, actions=a0)
    est = a0 - a1
    a = a0.clone().requires_grad_(True)
    objective_fn(wm.rollout(obs_0, a)[0], obs_g).sum().backward()
    rel = ((est - a.grad).norm() / a.grad.norm()).item()
    print(f"    relative error of the K={K} gradient: {rel:.4f}")
    assert rel < 0.05, rel


def test_smooth_vs_gd_vs_restarts_bumpy():
    """Sanity check, not evidence: final cost at equal K*S on the bumpy cost."""
    n, K, S, lr = 50, 8, 25, 0.01  # lr >= 0.03 diverges on this cost for all three
    wm = FakeWM(amp=0.5, freq=4.0)
    obs_0, obs_g = episodes(n, seed=1)
    runs = {
        f"smooth K={K} S={S}": make(SmoothGDPlanner, wm=wm, K=K, opt_steps=S, lr=lr),
        f"gd S={K * S}": make(GDPlanner, wm=wm, opt_steps=K * S, lr=lr),
        f"gd R={K} S={S}": make(GDPlanner, wm=wm, restarts=K, opt_steps=S, lr=lr),
    }
    for name, p in runs.items():
        _, cost = plan(p, obs_0, obs_g, seed=5)
        print(f"    {name:18s} median final cost {cost.median().item():.3f}")
        assert not torch.isnan(cost).any(), name


def test_smooth_antithetic_odd_k_raises():
    try:
        make(SmoothGDPlanner, K=3, antithetic=True)
    except AssertionError:
        return
    raise AssertionError("antithetic with K=3 did not raise")


# ---------------------------------------------------------------- runner

def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
