"""Cost-landscape probe between a GD run's plan and a CEM run's plan (no planning).

For each selected episode, with a_GD, a_CEM from the two runs' final_actions.pt and
d = a_CEM - a_GD:
  J_line[s, j]  cost along a(t) = (1 - t) a_GD + t a_CEM = a_GD + t d
  J_rand[s, j]  cost along a_GD + t |d| u, u a random unit direction (per episode)
  where for sigma_s > 0 the cost is the smoothed J_sigma(a) = mean_m J(a + sigma eps_m)
  over M perturbations eps_m drawn ONCE and reused for every point, sigma and episode;
  sigma = 0 is the plain cost J(a).
  grad_norm[s], grad_cos[s]  norm of g_sigma = mean_m grad J(a_GD + sigma eps_m) (plain
  gradient for sigma = 0) and its cosine with d
  J_gd, J_cem   plain J(a_GD), J(a_CEM)
Arrays are saved to probe.npz in the run directory (hydra's cwd), leading axis = episode.

plan() returns a_CEM for every episode, so the final evaluation must reproduce the CEM
run's output_final successes: a built-in check that the episodes (seed, goals) and the
actions line up. The run must therefore use the same model, seed, n_evals, goal_source
and goal_H as the two probed runs.
"""
import json
import os
import pickle

import numpy as np
import torch

from .base_planner import BasePlanner
from utils import move_to_device


def _resolve(path):
    """Relative paths are relative to where plan.py was launched (hydra changes the cwd)."""
    if os.path.isabs(path):
        return path
    try:
        import hydra
        return os.path.join(hydra.utils.get_original_cwd(), path)
    except (ImportError, ValueError):  # not running under hydra
        return os.path.abspath(path)


def _final_successes(run_dir):
    """Per-episode success of the run's final evaluation (episodes.jsonl, output_final)."""
    path = os.path.join(run_dir, "episodes.jsonl")
    final = None
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["filename"] == "output_final":
                    final = rec
    if final is None:
        raise ValueError(f"no output_final line in {path}")
    return np.asarray(final["success"], dtype=bool)


class ProbePlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        gd_run,
        cem_run,
        episodes=None,
        max_eps=8,
        sigmas=(0.0, 0.03, 0.1, 0.3, 1.0),
        M=64,
        n_t=61,
        t_range=(-0.25, 1.25),
        chunk=25,
        seed=0,
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm, action_dim, objective_fn, preprocessor, evaluator, wandb_run, log_filename
        )
        self.horizon = horizon
        self.gd_run = _resolve(gd_run)
        self.cem_run = _resolve(cem_run)
        self.episodes = None if episodes is None else [int(e) for e in episodes]
        self.max_eps = int(max_eps)
        self.sigmas = [float(s) for s in sigmas]
        self.M = int(M)
        self.n_t = int(n_t)
        self.t_range = (float(t_range[0]), float(t_range[1]))
        self.chunk = int(chunk)  # rows per wm.rollout call (with and without grad)
        self.seed = int(seed)
        assert all(s >= 0 for s in self.sigmas), self.sigmas
        assert self.M >= 1 and self.n_t >= 2 and self.chunk >= 1

    # ---------------------------------------------------------------- helpers
    def _select_episodes(self, n_evals):
        if self.episodes is not None:
            eps = self.episodes
            assert all(0 <= e < n_evals for e in eps), (eps, n_evals)
            return eps
        gd_ok = _final_successes(self.gd_run)
        cem_ok = _final_successes(self.cem_run)
        assert len(gd_ok) == len(cem_ok) == n_evals, (len(gd_ok), len(cem_ok), n_evals)
        eps = np.flatnonzero(~gd_ok & cem_ok)[: self.max_eps].tolist()
        if not eps:
            raise ValueError(
                f"no episode where GD failed and CEM succeeded "
                f"({self.gd_run}, {self.cem_run}); pass planner.episodes=[...]"
            )
        return eps

    def _check_targets(self, obs_0, obs_g):
        """If a probed run kept plan_targets.pkl, its start/goal images must equal ours."""
        for run in (self.gd_run, self.cem_run):
            path = os.path.join(run, "plan_targets.pkl")
            if not os.path.exists(path):
                print(f"probe: {path} not found; cannot check the episodes match")
                continue
            with open(path, "rb") as f:
                t = pickle.load(f)
            for name, mine, theirs in (("obs_0", obs_0, t["obs_0"]), ("obs_g", obs_g, t["obs_g"])):
                assert np.array_equal(np.asarray(mine["visual"]), np.asarray(theirs["visual"])), (
                    f"{name} differs from {path}: use the same seed, n_evals, goal_source, goal_H"
                )

    def _costs(self, acts, obs_0_e, z_g_e):
        """No-grad cost of every row of acts (rows, H, A) for one episode, in chunks."""
        out = []
        with torch.no_grad():
            for s in range(0, acts.shape[0], self.chunk):
                a = acts[s : s + self.chunk]
                b = a.shape[0]
                z, _ = self.wm.rollout(
                    obs_0={k: v.expand(b, *v.shape[1:]) for k, v in obs_0_e.items()}, act=a
                )
                out.append(self.objective_fn(z, {k: v.expand(b, *v.shape[1:]) for k, v in z_g_e.items()}))
        return torch.cat(out)

    def _smoothed(self, points, sigma, eps, obs_0_e, z_g_e):
        """J_sigma at each of points (P, H, A): mean over the M fixed eps; plain J for 0."""
        if sigma == 0:
            return self._costs(points, obs_0_e, z_g_e)
        P, M = points.shape[0], eps.shape[0]
        rows = (points[:, None] + sigma * eps[None]).reshape(P * M, *points.shape[1:])
        return self._costs(rows, obs_0_e, z_g_e).view(P, M).mean(1)

    def _grad(self, a, sigma, eps, obs_0_e, z_g_e):
        """g_sigma = mean_m grad J(a + sigma eps_m) (plain gradient for sigma = 0)."""
        a = a.detach().clone().requires_grad_(True)
        n_rows = 1 if sigma == 0 else eps.shape[0]
        for s in range(0, n_rows, self.chunk):
            e = min(s + self.chunk, n_rows)
            rows = a.expand(e - s, *a.shape) if sigma == 0 else a[None] + sigma * eps[s:e]
            z, _ = self.wm.rollout(
                obs_0={k: v.expand(e - s, *v.shape[1:]) for k, v in obs_0_e.items()}, act=rows
            )
            loss = self.objective_fn(z, {k: v.expand(e - s, *v.shape[1:]) for k, v in z_g_e.items()})
            (loss.sum() / n_rows).backward()
        return a.grad.detach()

    # ---------------------------------------------------------------- probe
    def plan(self, obs_0, obs_g, actions=None):
        n_evals = obs_0["visual"].shape[0]
        a_gd_all = torch.load(os.path.join(self.gd_run, "final_actions.pt"), map_location="cpu")
        a_cem_all = torch.load(os.path.join(self.cem_run, "final_actions.pt"), map_location="cpu")
        expect = (n_evals, self.horizon, self.action_dim)
        assert tuple(a_gd_all.shape) == expect, f"GD final_actions {tuple(a_gd_all.shape)} != {expect}"
        assert tuple(a_cem_all.shape) == expect, f"CEM final_actions {tuple(a_cem_all.shape)} != {expect}"
        self._check_targets(obs_0, obs_g)
        episodes = self._select_episodes(n_evals)
        print(f"probe: episodes {episodes}")

        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        with torch.no_grad():
            z_g = {k: v.detach() for k, v in self.wm.encode_obs(trans_obs_g).items()}

        H, A = self.horizon, self.action_dim
        gen = torch.Generator().manual_seed(self.seed)
        eps = torch.randn(self.M, H, A, generator=gen).to(self.device)  # shared by everything
        t = torch.linspace(self.t_range[0], self.t_range[1], self.n_t)
        tt = t.to(self.device)[:, None, None]
        E, S = len(episodes), len(self.sigmas)
        res = {
            "J_line": np.zeros((E, S, self.n_t), np.float32),
            "J_rand": np.zeros((E, S, self.n_t), np.float32),
            "grad_norm": np.zeros((E, S), np.float32),
            "grad_cos": np.zeros((E, S), np.float32),
            "J_gd": np.zeros(E, np.float32),
            "J_cem": np.zeros(E, np.float32),
            "d_norm": np.zeros(E, np.float32),
            "rand_dir": np.zeros((E, H, A), np.float32),
        }
        for i, ep in enumerate(episodes):
            o0 = {k: v[ep : ep + 1] for k, v in trans_obs_0.items()}
            zg = {k: v[ep : ep + 1] for k, v in z_g.items()}
            a_gd = a_gd_all[ep].to(self.device)
            a_cem = a_cem_all[ep].to(self.device)
            d = a_cem - a_gd
            u = torch.randn(H, A, generator=gen).to(self.device)
            u = u / u.norm()
            line = a_gd[None] + tt * d[None]
            rand = a_gd[None] + tt * d.norm() * u[None]
            for s, sigma in enumerate(self.sigmas):
                res["J_line"][i, s] = self._smoothed(line, sigma, eps, o0, zg).cpu().numpy()
                res["J_rand"][i, s] = self._smoothed(rand, sigma, eps, o0, zg).cpu().numpy()
                g = self._grad(a_gd, sigma, eps, o0, zg)
                res["grad_norm"][i, s] = g.norm().item()
                res["grad_cos"][i, s] = (
                    torch.nn.functional.cosine_similarity(g.flatten(), d.flatten(), dim=0).item()
                )
            j = self._costs(torch.stack([a_gd, a_cem]), o0, zg).cpu().numpy()
            res["J_gd"][i], res["J_cem"][i] = j
            res["d_norm"][i] = d.norm().item()
            res["rand_dir"][i] = u.cpu().numpy()
            print(f"probe: episode {ep}: J_gd {j[0]:.3f} J_cem {j[1]:.3f} |d| {d.norm().item():.3f} "
                  f"cos(g_sigma, d) [{', '.join(f'{c:.3f}' for c in res['grad_cos'][i])}]")

        np.savez(
            "probe.npz",
            episodes=np.asarray(episodes),
            sigmas=np.asarray(self.sigmas, np.float32),
            t=t.numpy(),
            M=self.M,
            seed=self.seed,
            gd_run=self.gd_run,
            cem_run=self.cem_run,
            **res,
        )
        # a_CEM for every episode: the final evaluation must reproduce CEM's successes
        return a_cem_all.to(self.device), np.full(n_evals, np.inf)
