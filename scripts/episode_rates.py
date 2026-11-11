"""Shared success-rate definitions for episodes.jsonl (written by PlanEvaluator.eval_actions).

Imported by scripts/phase1_rates.py and scripts/collect_results.py so that both report
the same numbers. Per run (run_rates) and pooled over runs (pool_rates):
  at_N   success at planning iteration N
  final  success of the final evaluation (filename "output_final")
  best   highest success rate of any single evaluation, the final one included
  any    fraction of episodes that succeeded in at least one evaluation, final included

Iterations are read from the evaluation filenames:
  CEM / GD   "plan_0_output_<step>"  -> iteration <step> (same "step" as logs.json)
  MPC        "plan<k>"               -> iteration k + 1 (one MPC replanning round)
Other lines (an MPC sub-planner's own "plan_<k>_output_<j>" evaluations) are ignored.
Only evaluated iterations exist (GD evaluates every `eval_every` steps). Planners stop
early once every episode succeeds: for N past the last evaluation of such a run, that
last (all-success) evaluation is used and at_carried is set.
Pooled best uses the same evaluation (iteration N, or final) for every run.
"""
import json
import re
from pathlib import Path

import numpy as np

CEM_GD = re.compile(r"^plan_0_output_(\d+)$")
MPC = re.compile(r"^plan(\d+)$")
FINAL = "output_final"


def read_episodes(path):
    """{filename: line} from a run dir or an episodes.jsonl path, plus whether filenames
    repeat (a re-run appended into the same directory; the latest line wins)."""
    path = Path(path)
    path = path if path.is_file() else path / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no episodes.jsonl in {path.parent}")
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    latest = {line["filename"]: line for line in lines}
    return latest, len(latest) != len(lines)


def success_at(iters, n):
    """Per-episode successes at iteration n and whether they were carried forward."""
    if n in iters:
        return iters[n], False
    if iters:
        last = max(iters)
        if n > last and iters[last].all():
            return iters[last], True
    return None, False


def _rate(successes):
    return float(successes.mean()) if successes is not None and len(successes) else None


def _best(candidates):
    """candidates: [(label, rate)] in evaluation order; earliest of the highest wins."""
    candidates = [(label, r) for label, r in candidates if r is not None]
    if not candidates:
        return None, None
    best_rate = max(r for _, r in candidates)
    return next(label for label, r in candidates if r == best_rate), best_rate


def run_rates(lines, at=None):
    """Rates for one run from read_episodes() output. Raises ValueError if evaluations
    disagree on the number of episodes."""
    mpc = {int(m.group(1)) + 1: l for f, l in lines.items() if (m := MPC.match(f))}
    cem_gd = {int(m.group(1)): l for f, l in lines.items() if (m := CEM_GD.match(f))}
    chosen = mpc or cem_gd
    iters = {n: np.asarray(chosen[n]["success"], dtype=bool) for n in sorted(chosen)}
    final = np.asarray(lines[FINAL]["success"], dtype=bool) if FINAL in lines else None

    evals = list(iters.values()) + ([final] if final is not None else [])
    sizes = {len(s) for s in evals}
    if len(sizes) > 1:
        raise ValueError(f"evaluations have different episode counts {sorted(sizes)}")

    rates = {n: _rate(s) for n, s in iters.items()}
    best_at, best_rate = _best(list(rates.items()) + [("final", _rate(final))])
    any_ok = np.logical_or.reduce(evals) if evals else None
    at_ok, carried = success_at(iters, at) if at is not None else (None, False)
    return {
        "kind": "mpc" if mpc else "cem/gd",
        "n_eps": sizes.pop() if sizes else 0,
        "iters": iters,
        "final": final,
        "rates": rates,
        "at": at_ok,
        "at_rate": _rate(at_ok),
        "at_carried": carried,
        "final_rate": _rate(final),
        "best_rate": best_rate,
        "best_at": best_at,
        "any": any_ok,
        "any_rate": _rate(any_ok),
    }


def pool_rates(runs, at=None):
    """Pool run_rates() results over all their episodes. at_N / final / best are None
    when some run lacks the needed evaluation (no partial pools)."""
    def pooled(arrays):
        return None if any(a is None for a in arrays) else np.concatenate(arrays)

    at_ok = pooled([success_at(r["iters"], at)[0] for r in runs]) if at is not None else None
    final = pooled([r["final"] for r in runs])
    candidates = []
    for n in sorted(set().union(*(r["iters"] for r in runs))):
        s = pooled([success_at(r["iters"], n)[0] for r in runs])
        candidates.append((n, _rate(s)))
    best_at, best_rate = _best(candidates + [("final", _rate(final))])
    any_ok = pooled([r["any"] for r in runs])
    return {
        "n_eps": sum(r["n_eps"] for r in runs),
        "at": at_ok,
        "at_rate": _rate(at_ok),
        "final": final,
        "final_rate": _rate(final),
        "best_rate": best_rate,
        "best_at": best_at,
        "any": any_ok,
        "any_rate": _rate(any_ok),
    }
