#!/usr/bin/env python
# [NEW] Written for this project; not part of upstream DINO-WM.
"""Entry point: Gaussian-smoothed gradient planning in DINO-WM.

The modification (planning/smooth_gd.py) replaces the gradient of the planning cost J(a)
with the gradient of its Gaussian smoothing, J_sigma(a) = E_eps[J(a + sigma eps)],
estimated with K (antithetic) perturbations per step, sigma annealed from sigma0 to
sigma_min. Everything else (world model, objective, optimizer, evaluation) is upstream.

Modes (each writes to --out, default outputs/):

  python main.py test       CPU, ~1 min. Unit tests of the planners on a small fake
                            world model (tests/*.py).
  python main.py toy        CPU, ~1 min. Smoothed GD vs plain GD on a bumpy fake world
                            model, every parameter adjustable; plots the cost landscape,
                            the sigma schedule and the cost per optimization step.
  python main.py analyze    CPU, ~1 min. Rebuilds every table and figure of the report
                            from the saved runs in results/ (no GPU, no dataset needed).
  python main.py demo       GPU + checkpoint + dataset. Runs CEM, tuned GD and smoothed
                            GD with the real DINO-WM on a few Wall episodes and plots
                            success / distance per optimization step.
  python main.py all        test + toy + analyze (everything that runs without a GPU).

`python main.py <mode> -h` lists the parameters of each mode.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# planners compared in the held-out evaluation (folder name prefix in results/wall)
PLANNERS = {
    "CEM@10": "cem10",
    "Tuned GD": "gd_adam_lr0.1_randn_R1_S1000",
    "Smoothed GD (K16)": "sgd_adam_lr0.1_randn_K16_S62_sig1_geom",
}
# compute per episode per logged step, in plan-rollout equivalents (planning/compute.py:
# one rollout = 1, one backward = 2): CEM 300 samples, GD 1 fwd + 1 bwd, K16 16 x that
PER_STEP = {"CEM@10": 300, "Tuned GD": 3, "Smoothed GD (K16)": 48}
HELDOUT_SEEDS = [101, 103, 107, 109, 113]
COLORS = {"CEM@10": "#7f7f7f", "Tuned GD": "#d62728", "Smoothed GD (K16)": "#1f77b4"}


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    p.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.grid": True,
                       "grid.alpha": 0.3, "savefig.dpi": 150})
    return p


def log(out, name, text):
    (out / name).write_text(text)
    print(text)


def run_cmd(cmd, out_file=None, env=None):
    """Run a command from the repo root; optionally save its stdout to out_file."""
    print("$", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if out_file is not None:
        Path(out_file).write_text(r.stdout)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return r.stdout


# ===================================================================== test

def cmd_test(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    text = ""
    for t in sorted((ROOT / "tests").glob("test_*.py")):
        text += f"== {t.name}\n" + run_cmd([sys.executable, str(t)])
    log(out, "test_log.txt", text)


# ===================================================================== toy

def cmd_toy(a):
    """GD vs smoothed GD on the fake world model of tests/test_gd_fake_wm.py."""
    import torch
    from tests.test_gd_fake_wm import FakeWM, objective_fn, episodes, H, A
    from planning.gd import GDPlanner
    from planning.smooth_gd import SmoothGDPlanner

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = plt()
    wm = FakeWM(amp=a.amp, freq=a.freq)
    obs_0, obs_g = episodes(a.n, seed=a.seed)

    def cost(actions):
        with torch.no_grad():
            return objective_fn(wm.rollout(obs_0, actions)[0], obs_g)

    class Recorder:
        """Stands in for PlanEvaluator: records the model cost of the plan being optimized
        (the unperturbed center for smoothed GD) every eval_every steps."""
        def __init__(self):
            self.trace = []

        def eval_actions(self, actions, *args, **kw):
            self.trace.append(cost(actions).numpy())
            return {}, np.zeros(a.n, bool), None, None

    def make(cls, rec, steps, **kw):
        cfg = dict(horizon=H, action_noise=0.0, sample_type="randn", lr=a.lr, opt_steps=steps,
                   eval_every=1, optimizer=a.optimizer, restarts=1, init_scale=1.0,
                   episode_chunk=4096, oracle_stop=False, **kw)
        return cls(wm=wm, action_dim=A, objective_fn=objective_fn,
                   preprocessor=types.SimpleNamespace(transform_obs=lambda o: o),
                   evaluator=rec, wandb_run=types.SimpleNamespace(log=lambda *x, **k: None),
                   log_filename=None, **cfg)

    # same compute per episode: smoothed runs S steps x K perturbations, GD runs K*S steps
    runs = {}
    for name, cls, steps, kw in [
        ("GD", GDPlanner, a.K * a.steps, {}),
        (f"Smoothed GD (K={a.K})", SmoothGDPlanner, a.steps,
         dict(K=a.K, sigma0=a.sigma0, sigma_min=a.sigma_min, sigma_schedule=a.schedule,
              antithetic=a.K % 2 == 0)),
    ]:
        rec = Recorder()
        planner = make(cls, rec, steps, **kw)
        torch.manual_seed(a.seed)
        final, _ = planner.plan(obs_0, obs_g)
        trace = np.stack(rec.trace)  # (evaluations, n)
        per_step = a.K if cls is GDPlanner else 1
        x = (np.arange(1, len(trace) + 1)) * (3 * (1 if cls is GDPlanner else a.K))
        runs[name] = dict(trace=trace, x=x, final=cost(final.detach()).numpy(),
                          planner=planner, every=per_step)

    # --- figure 1: cost per step (median and quartiles over episodes)
    fig, ax = p.subplots(figsize=(6.4, 4), constrained_layout=True)
    for (name, r), c in zip(runs.items(), ["#d62728", "#1f77b4"]):
        med = np.median(r["trace"], 1)
        q1, q3 = np.percentile(r["trace"], [25, 75], axis=1)
        ax.plot(r["x"], med, color=c, label=name)
        ax.fill_between(r["x"], q1, q3, color=c, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("compute per episode (rollout equivalents: fwd = 1, bwd = 2)")
    ax.set_ylabel("model cost J(a) of the current plan")
    ax.set_title(f"Fake bumpy world model (amp={a.amp}, freq={a.freq}), {a.n} episodes")
    ax.legend()
    fig.savefig(out / "toy_cost_per_step.png"); p.close(fig)

    # --- figure 2: sigma schedule
    sm = runs[f"Smoothed GD (K={a.K})"]["planner"]
    fig, ax = p.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.plot(range(1, a.steps + 1), [sm.sigma_at(i) for i in range(a.steps)], color="#1f77b4")
    ax.set_xlabel("optimization step"); ax.set_ylabel("sigma")
    ax.set_title(f"{a.schedule} schedule: sigma0={a.sigma0}, sigma_min={a.sigma_min}")
    fig.savefig(out / "toy_sigma_schedule.png"); p.close(fig)

    # --- figure 3: cost along a random line through a plan, raw and smoothed
    g = torch.Generator().manual_seed(a.seed)
    a0 = torch.randn(1, H, A, generator=g)
    u = torch.randn(1, H, A, generator=g); u /= u.norm()
    t = torch.linspace(-3, 3, 301)
    eps = torch.randn(512, H, A, generator=g)
    o0 = {k: v[:1] for k, v in obs_0.items()}; og = {k: v[:1] for k, v in obs_g.items()}
    fig, ax = p.subplots(figsize=(6.4, 4), constrained_layout=True)
    with torch.no_grad():
        for s, c in zip([0.0, 0.1, 0.3, 1.0], ["k", "#9ecae1", "#4292c6", "#08306b"]):
            ys = []
            for ti in t:
                pts = a0 + ti * u + s * eps if s > 0 else a0 + ti * u
                n_ = pts.shape[0]
                z = wm.rollout({k: v.expand(n_, *v.shape[1:]) for k, v in o0.items()}, pts)[0]
                ys.append(objective_fn(z, {k: v.expand(n_, *v.shape[1:]) for k, v in og.items()}).mean().item())
            ax.plot(t, ys, color=c, lw=1.2 if s == 0 else 2, label=f"sigma = {s}" + (" (raw J)" if s == 0 else ""))
    ax.set_xlabel("position t along a random direction, a = a0 + t u")
    ax.set_ylabel("J_sigma(a)")
    ax.set_title("Gaussian smoothing removes the small local minima")
    ax.legend()
    fig.savefig(out / "toy_landscape.png"); p.close(fig)

    lines = [f"# Toy experiment (fake world model, CPU)\n",
             f"settings: n={a.n} episodes, amp={a.amp}, freq={a.freq}, lr={a.lr}, optimizer={a.optimizer}, "
             f"K={a.K}, steps={a.steps}, sigma {a.sigma0} -> {a.sigma_min} ({a.schedule}), seed={a.seed}\n",
             "| Planner | median final cost | mean final cost | episodes with cost < 1 |", "|---|---|---|---|"]
    for name, r in runs.items():
        f = r["final"]
        lines.append(f"| {name} | {np.median(f):.3f} | {f.mean():.3f} | {(f < 1).mean():.2f} |")
    lines.append("\nFigures: toy_cost_per_step.png, toy_sigma_schedule.png, toy_landscape.png")
    log(out, "toy_summary.md", "\n".join(lines) + "\n")


# ===================================================================== analyze

def read_run(root, name, seed):
    """Per-episode final success / state distance / model cost, plus the per-step log."""
    d = Path(root) / f"{name}_s{seed}"
    final = None
    with open(d / "episodes.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["filename"] == "output_final":
                final = e
    steps = []
    with open(d / "logs.json") as f:
        for line in f:
            e = json.loads(line)
            if "plan_0/success_rate" in e:
                steps.append((e["step"], e["plan_0/success_rate"], e["plan_0/mean_state_dist"]))
    stats = json.loads((d / "run_stats.json").read_text())
    return dict(success=np.array(final["success"], bool), dist=np.array(final["state_dist"], float),
                cost=np.array(final["model_cost"], float), steps=np.array(steps), stats=stats)


def ci95(p, n):
    h = 1.96 * np.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)


def cmd_analyze(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.results)
    py = sys.executable
    seeds = [str(s) for s in a.seeds]

    # 1. the analysis scripts of record (markdown reports)
    run_cmd([py, "scripts/replication_report.py", str(root)], out / "replication_report.md")
    run_cmd([py, "scripts/phase6_report.py", str(root)], out / "phase6_report.md")
    run_cmd([py, "scripts/collect_results.py", str(root), "--heldout-seeds", *seeds,
             "--csv", str(out / "summary.csv")], out / "summary.md")
    if (root.parent / "wall_6b").is_dir():
        run_cmd([py, "scripts/budget_curve.py"], out / "budget_curve.md")
        if (root.parent / "wall_6b/e3_rescore_s99").is_dir():
            run_cmd([py, "scripts/rescore_report.py", str(root.parent / "wall_6b/e3_rescore_s99")],
                    out / "rescore_report.md")
    if (root / "probe_s99").is_dir():
        run_cmd([py, "scripts/probe_report.py", str(root / "probe_s99")], out / "probe_report.md")

    # 2. per-episode data of the three compared planners on the held-out seeds
    data = {lab: {s: read_run(root, name, s) for s in a.seeds} for lab, name in PLANNERS.items()}
    cat = lambda lab, k: np.concatenate([data[lab][s][k] for s in a.seeds])
    cem_cost = cat("CEM@10", "cost")
    n = len(cem_cost)

    rows = ["# Key metrics, held-out seeds " + ", ".join(seeds) + f" (n = {n} episodes)\n",
            "Fooled = failed with final model cost below CEM@10's on the same episode "
            "(the world model rates the plan better than CEM's, yet it fails); stuck = other failures.\n",
            "| Planner | Success (95% CI) | Mean / median final dist | Mean model cost | Fooled | Stuck | Compute / episode | Planning time / run |",
            "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for lab in PLANNERS:
        s, d, c = cat(lab, "success"), cat(lab, "dist"), cat(lab, "cost")
        fooled = (~s) & (c < cem_cost) if lab != "CEM@10" else np.zeros(n, bool)
        stuck = (~s) & ~fooled
        st = [data[lab][x]["stats"] for x in a.seeds]
        pe = np.mean([x["plan_equiv"] for x in st]) / 50
        ps = np.mean([x.get("plan_s", x["wall_clock_s"]) for x in st]) / 60
        lo, hi = ci95(s.mean(), n)
        summary[lab] = dict(s=s, fooled=fooled.sum(), stuck=stuck.sum())
        f_txt = "ref." if lab == "CEM@10" else str(fooled.sum())
        s_txt = str((~s).sum()) if lab == "CEM@10" else str(stuck.sum())
        rows.append(f"| {lab} | {s.mean():.2f} [{lo:.2f}, {hi:.2f}] | {d.mean():.1f} / {np.median(d):.1f} | "
                    f"{c.mean():.3f} | {f_txt} | {s_txt} | {pe:,.0f} | {ps:.0f} min |")
    from scipy.stats import binomtest
    sm, gd = summary["Smoothed GD (K16)"]["s"], summary["Tuned GD"]["s"]
    b, c_ = int((sm & ~gd).sum()), int((gd & ~sm).sum())
    p = binomtest(b, b + c_, 0.5).pvalue if b + c_ else 1.0
    rows.append(f"\nPaired comparison, smoothed vs tuned GD: only smoothed solves {b}, only GD solves {c_}, "
                f"exact McNemar p = {p:.3g} (all {n} held-out episodes pooled; the pre-registered test "
                "is in replication_report.md, seeds 107/109/113 only).")
    log(out, "key_metrics.md", "\n".join(rows) + "\n")

    p_ = plt()
    labs = list(PLANNERS)

    # fig 1: final success with 95% CI, per-seed points
    fig, ax = p_.subplots(figsize=(6.4, 4), constrained_layout=True)
    for i, lab in enumerate(labs):
        s = summary[lab]["s"]; lo, hi = ci95(s.mean(), n)
        ax.bar(i, s.mean(), color=COLORS[lab], alpha=0.85, yerr=[[s.mean() - lo], [hi - s.mean()]], capsize=6)
        per_seed = [data[lab][x]["success"].mean() for x in a.seeds]
        ax.scatter(i + np.linspace(-0.2, 0.2, len(per_seed)), per_seed, color="k", s=14, zorder=3)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    ax.set_ylabel("final success rate"); ax.set_ylim(0, 1)
    ax.set_title(f"Wall, open-loop, equal compute; {n} held-out episodes\n(bars: pooled, 95% CI; dots: per seed)")
    fig.savefig(out / "fig_success.png"); p_.close(fig)

    # fig 2: failure types
    fig, ax = p_.subplots(figsize=(5.4, 4), constrained_layout=True)
    two = ["Tuned GD", "Smoothed GD (K16)"]
    f_ = [summary[l]["fooled"] for l in two]; s_ = [summary[l]["stuck"] for l in two]
    ax.bar(two, f_, color="#ff7f0e", label="fooled (cost below CEM's)")
    ax.bar(two, s_, bottom=f_, color="#bdbdbd", label="stuck (cost above CEM's)")
    for i, (x, y) in enumerate(zip(f_, s_)):
        ax.text(i, x / 2, str(x), ha="center", va="center"); ax.text(i, x + y / 2, str(y), ha="center", va="center")
    ax.set_ylabel(f"failed episodes (of {n})"); ax.legend(loc="upper right")
    ax.set_title("Why gradient planners fail")
    fig.savefig(out / "fig_failure_types.png"); p_.close(fig)

    # fig 3: success and distance during optimization vs compute
    fig, axs = p_.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for lab in labs:
        st = [data[lab][x]["steps"] for x in a.seeds]
        L = min(len(x) for x in st)
        arr = np.stack([x[:L] for x in st])  # (seeds, evals, 3)
        x = arr[0, :, 0] * PER_STEP[lab]
        axs[0].plot(x, arr[:, :, 1].mean(0), color=COLORS[lab], label=lab, marker="o" if L < 20 else None, ms=3)
        axs[1].plot(x, arr[:, :, 2].mean(0), color=COLORS[lab], label=lab, marker="o" if L < 20 else None, ms=3)
    for ax_, yl in zip(axs, ["success rate (mean over seeds)", "mean final state distance"]):
        ax_.set_xlabel("compute per episode so far (rollout equivalents)"); ax_.set_ylabel(yl)
    axs[0].legend(); axs[0].set_title("Intermediate evaluations within each run")
    axs[1].set_title("Distance to goal in the simulator")
    fig.savefig(out / "fig_progress.png"); p_.close(fig)

    # fig 4: final distance CDF
    fig, ax = p_.subplots(figsize=(6.4, 4), constrained_layout=True)
    for lab in labs:
        d = np.sort(cat(lab, "dist"))
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=COLORS[lab], label=lab)
    ax.set_xlabel("final state distance to goal"); ax.set_ylabel("fraction of episodes ≤ x")
    ax.set_title("Final distance, all held-out episodes"); ax.legend()
    fig.savefig(out / "fig_distance_cdf.png"); p_.close(fig)

    # fig 5: cost landscape between the GD and CEM plans (probe run, seed 99)
    pf = root / "probe_s99" / "probe.npz"
    if pf.is_file():
        z = np.load(pf)
        sig, t, J = z["sigmas"], z["t"], z["J_line"]
        k = min(4, J.shape[0])
        fig, axs = p_.subplots(1, k, figsize=(3.2 * k, 3.4), constrained_layout=True, squeeze=False)
        shades = p_.cm.Blues(np.linspace(0.35, 1, len(sig)))
        for e in range(k):
            ax_ = axs[0, e]
            for j, s in enumerate(sig):
                ax_.plot(t, J[e, j], color="k" if s == 0 else shades[j], lw=1 if s == 0 else 1.8,
                         label=f"σ={s:g}" + (" (raw)" if s == 0 else ""))
            ax_.axvline(0, ls=":", c="#d62728"); ax_.axvline(1, ls=":", c="#7f7f7f")
            ax_.set_title(f"episode {int(z['episodes'][e])}"); ax_.set_xlabel("t  (0 = GD plan, 1 = CEM plan)")
        axs[0, 0].set_ylabel("J_σ along the line"); axs[0, 0].legend(fontsize=8)
        fig.suptitle("DINO-WM cost between a failed GD plan and a successful CEM plan (seed 99)")
        fig.savefig(out / "fig_landscape.png"); p_.close(fig)

    print(f"\nwrote reports and figures to {out}/")


# ===================================================================== demo

def cmd_demo(a):
    out = Path(a.out) / "demo"; out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(a.ckpt).expanduser()
    if not (ckpt / "outputs" / "wall_single").is_dir():
        raise SystemExit(f"checkpoint not found: {ckpt}/outputs/wall_single (see model.txt; pass --ckpt)")
    if not os.environ.get("DATASET_DIR"):
        raise SystemExit("set DATASET_DIR to the folder that contains wall_single/ (see data.txt)")
    env = {**os.environ, "WANDB_MODE": "disabled"}
    common = ["model_name=wall_single", "goal_source=random_state", "goal_H=5", f"n_evals={a.n_evals}",
              f"seed={a.seed}", f"ckpt_base_path={ckpt}", "+planner.oracle_stop=false", f"n_plot_samples={a.n_evals}"]
    adam = ["planner.optimizer=adam", f"planner.lr={a.lr}", "planner.sample_type=randn"]
    cfgs = {
        "CEM@10": ["planner=cem", f"planner.opt_steps={a.cem_steps}"],
        "Tuned GD": ["planner=gd", *adam, f"planner.opt_steps={a.gd_steps}", "planner.eval_every=10",
                     "+planner.episode_chunk=25"],
        "Smoothed GD (K16)": ["planner=smooth_gd", *adam, f"planner.K={a.K}", f"planner.opt_steps={a.steps}",
                              f"planner.sigma0={a.sigma0}", f"planner.sigma_min={a.sigma_min}",
                              f"planner.sigma_schedule={a.schedule}", "planner.eval_every=1",
                              "+planner.episode_chunk=25"],
    }
    per_step = {"CEM@10": 300, "Tuned GD": 3, "Smoothed GD (K16)": 3 * a.K}
    res = {}
    for lab in a.planners:
        d = out / PLANNERS[lab]
        t0 = time.time()
        run_cmd([sys.executable, "plan.py", *cfgs[lab], *common, f"hydra.run.dir={d}"],
                out / f"{PLANNERS[lab]}.log", env=env)
        steps = [json.loads(l) for l in open(d / "logs.json")]
        fin = [e for e in steps if "final_eval/success_rate" in e][0]
        st = np.array([(e["step"], e["plan_0/success_rate"], e["plan_0/mean_state_dist"])
                       for e in steps if "plan_0/success_rate" in e])
        res[lab] = dict(steps=st, final=fin, minutes=(time.time() - t0) / 60)

    p_ = plt()
    fig, axs = p_.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for lab, r in res.items():
        x = r["steps"][:, 0] * per_step[lab]
        axs[0].plot(x, r["steps"][:, 1], color=COLORS[lab], label=lab, marker="o", ms=3)
        axs[1].plot(x, r["steps"][:, 2], color=COLORS[lab], label=lab, marker="o", ms=3)
    axs[0].set_ylabel("success rate"); axs[1].set_ylabel("mean state distance")
    for ax_ in axs:
        ax_.set_xlabel("compute per episode (rollout equivalents)")
    axs[0].legend(); fig.suptitle(f"Demo: {a.n_evals} Wall episodes, seed {a.seed}")
    fig.savefig(out / "demo_progress.png"); p_.close(fig)

    lines = [f"# Demo ({a.n_evals} episodes, seed {a.seed})\n", "| Planner | Final success | Mean final dist | Run time |",
             "|---|---|---|---|"]
    for lab, r in res.items():
        lines.append(f"| {lab} | {r['final']['final_eval/success_rate']:.2f} | "
                     f"{r['final']['final_eval/mean_state_dist']:.2f} | {r['minutes']:.1f} min |")
    lines.append(f"\nPer-run folders in {out}/: logs.json (every evaluation), episodes.jsonl (per episode), "
                 "run_stats.json (compute, time), final_actions.pt, output_final*.png (rollout images).")
    log(out, "demo_summary.md", "\n".join(lines) + "\n")


# ===================================================================== cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def add(name, help_):
        s = sub.add_parser(name, help=help_, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        s.add_argument("--out", default="outputs", help="output folder")
        return s

    add("test", "unit tests on a fake world model (CPU)")

    t = add("toy", "smoothed vs plain GD on a bumpy fake world model (CPU)")
    t.add_argument("--n", type=int, default=50, help="episodes")
    t.add_argument("--K", type=int, default=8, help="perturbations per step (smoothed)")
    t.add_argument("--steps", type=int, default=25, help="smoothed GD steps; GD gets K*steps")
    t.add_argument("--sigma0", type=float, default=0.3)
    t.add_argument("--sigma-min", type=float, default=0.03)
    t.add_argument("--schedule", choices=["const", "linear", "geom"], default="geom")
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    t.add_argument("--amp", type=float, default=0.5, help="bumpiness of the fake model (0 = quadratic cost)")
    t.add_argument("--freq", type=float, default=4.0)
    t.add_argument("--seed", type=int, default=1)

    z = add("analyze", "rebuild report tables and figures from results/ (CPU)")
    z.add_argument("--results", default="results/wall")
    z.add_argument("--seeds", type=int, nargs="+", default=HELDOUT_SEEDS, help="held-out seeds")

    d = add("demo", "real DINO-WM planning on a few Wall episodes (GPU)")
    d.add_argument("--ckpt", default=os.environ.get("DINO_WM_CKPT", "./checkpoints"),
                   help="folder containing outputs/wall_single (default: $DINO_WM_CKPT)")
    d.add_argument("--planners", nargs="+", default=list(PLANNERS), choices=list(PLANNERS))
    d.add_argument("--n-evals", type=int, default=5, help="episodes")
    d.add_argument("--seed", type=int, default=107)
    d.add_argument("--K", type=int, default=16)
    d.add_argument("--steps", type=int, default=62, help="smoothed GD steps")
    d.add_argument("--sigma0", type=float, default=1.0)
    d.add_argument("--sigma-min", type=float, default=0.1)
    d.add_argument("--schedule", choices=["const", "linear", "geom"], default="geom")
    d.add_argument("--lr", type=float, default=0.1, help="Adam learning rate (GD and smoothed)")
    d.add_argument("--gd-steps", type=int, default=1000)
    d.add_argument("--cem-steps", type=int, default=10)

    add("all", "test + toy + analyze")

    a = ap.parse_args()
    if a.mode == "all":
        ns = lambda **kw: argparse.Namespace(out=a.out, **kw)
        cmd_test(ns())
        cmd_toy(ap.parse_args(["toy", "--out", a.out]))
        cmd_analyze(ap.parse_args(["analyze", "--out", a.out]))
    else:
        {"test": cmd_test, "toy": cmd_toy, "analyze": cmd_analyze, "demo": cmd_demo}[a.mode](a)


if __name__ == "__main__":
    main()
