#!/usr/bin/env python
"""Success rates from the episodes.jsonl files written by PlanEvaluator.eval_actions.

usage: python scripts/phase1_rates.py <run_dir>... --at N

For each run (and pooled over all episodes of all runs) prints:
  at_N   success rate at planning iteration N
  final  success rate at the final evaluation (filename "output_final")
  best   highest success rate of any single evaluated iteration
  any    fraction of episodes that succeeded at one or more evaluated iterations

Iterations are read from the evaluation filenames:
  CEM / GD   "plan_0_output_<step>"  -> iteration <step> (same "step" as logs.json)
  MPC        "plan<k>"               -> iteration k + 1 (one MPC replanning round)
Only iterations that were evaluated exist (GD evaluates every `eval_every` steps).
Planners stop early once every episode succeeds; for N past the last evaluation of
such a run, its last (all-success) evaluation is used and the value is marked "*".
"Final" is not counted as an iteration for "best" or "any".
Pooled "best" is the best iteration of the pooled rate (same N for every run).
"""
import argparse
import json
import re
import sys
from pathlib import Path

CEM_GD = re.compile(r"^plan_0_output_(\d+)$")
MPC = re.compile(r"^plan(\d+)$")


def load_run(run_dir):
    path = Path(run_dir)
    path = path if path.is_file() else path / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no episodes.jsonl in {run_dir}")
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    seen = {}
    for line in lines:  # a re-run into the same directory appends; keep the latest
        seen[line["filename"]] = line
    if len(seen) != len(lines):
        print(f"warning: {path} has repeated filenames; using the last of each", file=sys.stderr)

    mpc = {int(m.group(1)) + 1: l["success"] for f, l in seen.items() if (m := MPC.match(f))}
    cem_gd = {int(m.group(1)): l["success"] for f, l in seen.items() if (m := CEM_GD.match(f))}
    iters = mpc or cem_gd  # MPC runs also contain sub-planner lines (plan_<k>_output_*)
    final = seen.get("output_final", {}).get("success")

    sizes = {len(s) for s in list(iters.values()) + ([final] if final else [])}
    if len(sizes) > 1:
        raise ValueError(f"{path}: evaluations have different episode counts {sorted(sizes)}")
    n_eps = sizes.pop() if sizes else 0
    return {"name": str(run_dir), "iters": dict(sorted(iters.items())), "final": final,
            "n_eps": n_eps, "kind": "mpc" if mpc else "cem/gd"}


def at_iteration(run, n):
    """Per-episode successes at iteration n, and whether it was carried forward."""
    iters = run["iters"]
    if n in iters:
        return iters[n], False
    if iters:
        last = max(iters)
        if n > last and all(iters[last]):
            return iters[last], True
    return None, False


def rate(successes):
    return sum(successes) / len(successes) if successes else None


def fmt(value, mark=False):
    return "    n/a" if value is None else f"{value:7.3f}" + ("*" if mark else " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", help="run directories (or episodes.jsonl files)")
    parser.add_argument("--at", type=int, required=True, metavar="N", help="planning iteration N")
    args = parser.parse_args()

    runs = []
    for d in args.run_dirs:
        try:
            runs.append(load_run(d))
        except (FileNotFoundError, ValueError) as e:
            print(f"skipping: {e}", file=sys.stderr)
    if not runs:
        sys.exit("no usable runs")

    width = max(len(r["name"]) for r in runs + [{"name": "POOLED"}])
    print(f"{'run':{width}}  {'kind':6}  {'eps':>4}  {'at_' + str(args.at):>8} {'final':>8} {'best':>8} {'(iter)':>6} {'any':>8}")

    pooled_at, pooled_final, pooled_any, all_iters = [], [], [], set()
    missing_at = missing_final = False
    for r in runs:
        at, carried = at_iteration(r, args.at)
        rates = {n: rate(s) for n, s in r["iters"].items()}
        best_n = max(rates, key=lambda n: (rates[n], -n)) if rates else None  # earliest best
        any_ok = [any(col) for col in zip(*r["iters"].values())] if r["iters"] else None
        print(f"{r['name']:{width}}  {r['kind']:6}  {r['n_eps']:4d}  {fmt(rate(at), carried)}"
              f"{fmt(rate(r['final']))}{fmt(rates.get(best_n))}{best_n if best_n else '-':>6}"
              f" {fmt(rate(any_ok))}")
        if at is None:
            avail = list(r["iters"]) or "none"
            print(f"{'':{width}}  iteration {args.at} not evaluated; available: {avail}")
            missing_at = True
        else:
            pooled_at += at
        if r["final"] is None:
            missing_final = True
        else:
            pooled_final += r["final"]
        pooled_any += any_ok or []
        all_iters |= set(r["iters"])

    # pooled best: same iteration n for every run (carrying forward early-stopped runs)
    pooled_rates = {}
    for n in sorted(all_iters):
        cols = [at_iteration(r, n)[0] for r in runs]
        if all(c is not None for c in cols):
            pooled_rates[n] = rate([s for c in cols for s in c])
    best_n = max(pooled_rates, key=lambda n: (pooled_rates[n], -n)) if pooled_rates else None
    total = sum(r["n_eps"] for r in runs)
    print(f"{'POOLED':{width}}  {'':6}  {total:4d}  {fmt(None if missing_at else rate(pooled_at))}"
          f"{fmt(None if missing_final else rate(pooled_final))}{fmt(pooled_rates.get(best_n))}"
          f"{best_n if best_n else '-':>6} {fmt(rate(pooled_any))}")
    if missing_at or missing_final:
        print("pooled at_N / final are n/a because at least one run lacks that evaluation")
    if any(at_iteration(r, args.at)[1] for r in runs):
        print("* carried forward: the run stopped early after every episode succeeded")


if __name__ == "__main__":
    main()
