#!/usr/bin/env python
"""Success rates from the episodes.jsonl files written by PlanEvaluator.eval_actions.

usage: python scripts/phase1_rates.py <run_dir>... --at N

For each run (and pooled over all episodes of all runs) prints:
  at_N   success rate at planning iteration N
  final  success rate at the final evaluation (filename "output_final")
  best   highest success rate of any single evaluation, and where it occurred
  any    fraction of episodes that succeeded at one or more evaluations

The definitions live in scripts/episode_rates.py, shared with collect_results.py:
"best" and "any" include the final evaluation as well as every evaluated iteration.
Iterations come from the evaluation filenames (CEM/GD "plan_0_output_<step>", MPC
"plan<k>" = round k+1); only evaluated iterations exist (GD: every eval_every steps).
Planners stop early once every episode succeeds; for N past the last evaluation of
such a run, its last (all-success) evaluation is used and the value is marked "*".
Pooled "best" is the best single evaluation of the pooled rate (same one for every run).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared module lives next to this file
from episode_rates import pool_rates, read_episodes, run_rates  # noqa: E402


def fmt(value, mark=False):
    return "    n/a " if value is None else f"{value:7.3f}" + ("*" if mark else " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", help="run directories (or episodes.jsonl files)")
    parser.add_argument("--at", type=int, required=True, metavar="N", help="planning iteration N")
    args = parser.parse_args()

    runs = []
    for d in args.run_dirs:
        try:
            lines, repeated = read_episodes(d)
            if repeated:
                print(f"warning: {d} has repeated filenames; using the last of each", file=sys.stderr)
            runs.append((str(d), run_rates(lines, at=args.at)))
        except (FileNotFoundError, ValueError) as e:
            print(f"skipping {d}: {e}", file=sys.stderr)
    if not runs:
        sys.exit("no usable runs")

    width = max(len(name) for name, _ in runs + [("POOLED", None)])
    print(f"{'run':{width}}  {'kind':6}  {'eps':>4}  {'at_' + str(args.at):>8}"
          f"{'final':>8}{'best':>9} {'(at)':>6} {'any':>8}")

    def row(name, kind, r):
        best_at = "-" if r["best_at"] is None else r["best_at"]
        print(f"{name:{width}}  {kind:6}  {r['n_eps']:4d}  {fmt(r['at_rate'], r.get('at_carried'))}"
              f"{fmt(r['final_rate'])}{fmt(r['best_rate'])}{best_at:>6} {fmt(r['any_rate'])}")

    for name, r in runs:
        row(name, r["kind"], r)
        if r["at"] is None:
            print(f"{'':{width}}  iteration {args.at} not evaluated; available: {list(r['iters']) or 'none'}")

    rates = [r for _, r in runs]
    row("POOLED", "", pool_rates(rates, at=args.at))
    if any(r["at"] is None for r in rates) or any(r["final"] is None for r in rates):
        print("pooled at_N / final are n/a because at least one run lacks that evaluation")
    if any(r["at_carried"] for r in rates):
        print("* carried forward: the run stopped early after every episode succeeded")


if __name__ == "__main__":
    main()
