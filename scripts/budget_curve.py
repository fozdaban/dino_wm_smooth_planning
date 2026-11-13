#!/usr/bin/env python
"""Phase 6b E1: success against compute budget (plan-rollout equivalents per episode).

usage: python scripts/budget_curve.py [run dirs ...] [--budgets 3000 6000 9000 12000]
       default runs: every results/wall_6b/*_s99 plus the phase-5/6 seed-99 references.

Budget of an evaluation, per episode (the unit of roadmap.md, backward = 2 forwards):
  CEM (planner cem)          300 samples x iteration
  GD  (planner gd)           3 x restarts x step         (+ restarts for the final pick)
  smoothed (planner smooth_gd) 3 x K x step
For CEM and GD, stopping a long run at step N gives exactly the plan a run with
opt_steps = N would return (nothing depends on opt_steps), so one long run gives the
whole curve. GD restarts at an intermediate step are picked by the cost before that
step's update (logging) instead of a final pass: a close approximation.
For smoothed GD the sigma schedule is stretched over opt_steps, so ONLY a run's final
evaluation is a valid point: each smoothed budget needs its own run.

A budget cell shows the success of the last evaluation at or below that budget (blank
if the run has none within 10% below it), with the actual budget in brackets.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_rates import FINAL, read_episodes, run_rates  # noqa: E402

REFS = [
    "results/wall/endpoint_cem10_s99",
    "results/wall/gd_adam_lr0.1_randn_R1_S1000_s99",
    "results/wall/gd_adam_lr0.1_randn_R8_S125_s99",
    "results/wall/sgd_adam_lr0.1_randn_K16_S62_sig1_geom_s99",
    "results/wall/sgd_adam_lr0.1_randn_K8_S125_sig1_geom_s99",
]


def per_step_cost(cfg):
    p = cfg["planner"]
    name = p.get("name")
    if name == "cem":
        return "cem", p.get("num_samples", 300), 0
    if name == "gd":
        r = int(p.get("restarts", 1))
        return "gd", 3 * r, (r if r > 1 else 0)
    if name == "smooth_gd":
        return "smooth_gd", 3 * int(p["K"]), 0
    return name, None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--budgets", type=int, nargs="+", default=[3000, 6000, 9000, 12000])
    a = ap.parse_args()
    runs = a.runs or sorted(str(p) for p in Path("results/wall_6b").glob("*_s99")) + REFS
    head = ["run", "kind", "final (budget)"] + [f"@{b}" for b in a.budgets]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in runs:
        d = Path(r)
        if not (d / "episodes.jsonl").is_file():
            print(f"| {d.name} | missing | |" + " |" * len(a.budgets))
            continue
        cfg = yaml.safe_load((d / ".hydra" / "config.yaml").read_text())
        kind, unit, extra = per_step_cost(cfg)
        lines, _ = read_episodes(d)
        rates = run_rates(lines)
        steps = int(cfg["planner"]["opt_steps"])
        final_budget = unit * steps + extra if unit else None
        cells = [d.name, kind,
                 f"{rates['final_rate']:.2f} ({final_budget})" if rates["final_rate"] is not None else ""]
        for b in a.budgets:
            if kind == "smooth_gd" or unit is None:
                ok = final_budget is not None and 0.9 * b <= final_budget <= 1.01 * b
                cells.append(f"{rates['final_rate']:.2f}" if ok else "")
                continue
            pts = [(n * unit, rate) for n, rate in rates["rates"].items() if n * unit <= 1.01 * b]
            if final_budget is not None and final_budget <= 1.01 * b:
                pts.append((final_budget, rates["final_rate"]))
            pts = [pt for pt in pts if pt[0] >= 0.9 * b]
            cells.append(f"{max(pts)[1]:.2f} ({max(pts)[0]})" if pts else "")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
