#!/usr/bin/env python
"""Phase 6 analysis, exactly as pre-registered in claude/phase6_preregistration.md.

usage: python scripts/phase6_report.py [results/wall] [--seeds 101 103] > results/wall/phase6_report.md

Written before any held-out smoothing result. Reads results/wall/<name>_s<seed>/ for the
five planners below and prints, in markdown:
  1. validity checks (all runs present, git_dirty false, same code, plan_equiv, same goals)
  2. success per seed and pooled (95% normal interval)
  3. exact two-sided McNemar tests: primary (K16 vs tuned GD, GD + restarts, CEM@10),
     secondary (K8 vs GD + restarts), alpha 0.05, no multiplicity correction
  4. failures split into fooled (final model cost < CEM@10's on the same episode) / stuck,
     and successes on the episodes where tuned GD failed, by type
  5. descriptive secondary measures (no tests)
"""
import argparse
import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_rates import FINAL, read_episodes, run_rates  # noqa: E402

PLANNERS = [  # (short name, run name without _s<seed>, expected plan_equiv per 50-episode run)
    ("CEM@10", "cem10", 150_000),
    ("Tuned GD", "gd_adam_lr0.1_randn_R1_S1000", 150_000),
    ("GD + restarts", "gd_adam_lr0.1_randn_R8_S125", 150_400),
    ("Smoothed K16", "sgd_adam_lr0.1_randn_K16_S62_sig1_geom", 148_800),
    ("Smoothed K8", "sgd_adam_lr0.1_randn_K8_S125_sig1_geom", 150_000),
]
PRIMARY = [("Smoothed K16", "Tuned GD"), ("Smoothed K16", "GD + restarts"), ("Smoothed K16", "CEM@10")]
SECONDARY = [("Smoothed K8", "GD + restarts")]
CEM = "CEM@10"


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: (only a, only b, p)."""
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = only_a + only_b, min(only_a, only_b)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return only_a, only_b, p


def ci(p, n):
    h = 1.96 * math.sqrt(max(p * (1 - p), 0) / n)
    return f"{p:.2f} [{max(0, p - h):.2f}, {min(1, p + h):.2f}]"


def load(root, name, seed):
    d = Path(root) / f"{name}_s{seed}"
    if not d.is_dir():
        return None
    lines, _ = read_episodes(d)
    fin = lines[FINAL]
    rates = run_rates(lines)
    stats = json.loads((d / "run_stats.json").read_text())
    return {
        "dir": d,
        "success": np.asarray(fin["success"], dtype=bool),
        "dist": np.asarray(fin["state_dist"], dtype=float),
        "cost": np.asarray(fin["model_cost"], dtype=float) if "model_cost" in fin else None,
        "gap": np.asarray(fin["div_visual_emb"], dtype=float) if "div_visual_emb" in fin else None,
        "any": rates["any"], "best_rate": rates["best_rate"], "best_at": rates["best_at"],
        "stats": stats,
    }


def same_code(c1, c2):
    """True if two commits differ only in results/, logs and summary files (None if unknown)."""
    if c1 == c2:
        return True
    try:
        r = subprocess.run(["git", "diff", "--quiet", c1, c2, "--", ".",
                            ":(exclude)results", ":(exclude)*.log"], capture_output=True)
        return {0: True, 1: False}.get(r.returncode)
    except Exception:
        return None


def goals_match(dirs):
    targets = []
    for d in dirs:
        p = d / "plan_targets.pkl"
        if not p.is_file():
            return None
        with open(p, "rb") as f:
            t = pickle.load(f)
        targets.append([np.asarray(t[k]) for k in ("state_0", "state_g")])
    return all(all(np.array_equal(a, b) for a, b in zip(targets[0], t)) for t in targets[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/wall")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 103])
    a = ap.parse_args()
    seeds = a.seeds

    runs = {short: {s: load(a.root, name, s) for s in seeds} for short, name, _ in PLANNERS}
    print(f"# Phase 6 report (seeds {', '.join(map(str, seeds))})\n")

    # ---- 1. validity ---------------------------------------------------------------
    print("## 1. Validity checks\n")
    problems = []
    missing = [f"{n}_s{s}" for sh, n, _ in PLANNERS for s in seeds if runs[sh][s] is None]
    if missing:
        print("**Missing runs:** " + ", ".join(missing) + "\n")
        problems.append("missing runs")
    present = [(sh, s, r) for sh in runs for s, r in runs[sh].items() if r is not None]
    dirty = [f"{sh} s{s}" for sh, s, r in present if r["stats"].get("git_dirty") is not False]
    commits = sorted({r["stats"].get("git_commit") or "?" for _, _, r in present})
    code_same = all(same_code(commits[0], c) for c in commits[1:]) if len(commits) > 1 else True
    print(f"- git_dirty false for all runs: {'yes' if not dirty else 'NO: ' + ', '.join(dirty)}")
    print(f"- commits: {', '.join(c[:7] for c in commits)}; same code: "
          f"{ {True: 'yes', False: 'NO', None: 'unknown (run inside the repo)'}[code_same] }")
    if dirty or code_same is False:
        problems.append("dirty or mixed code")
    bad_pe = [f"{sh} s{s}: {r['stats'].get('plan_equiv')} (expected {pe})"
              for sh, n, pe in PLANNERS for s in seeds if (r := runs[sh][s]) is not None
              and r["stats"].get("plan_equiv") != pe]
    print(f"- plan_equiv as expected: {'yes' if not bad_pe else 'NO: ' + '; '.join(bad_pe)}")
    if bad_pe:
        problems.append("plan_equiv")
    for s in seeds:
        dirs = [runs[sh][s]["dir"] for sh in runs if runs[sh][s] is not None]
        g = goals_match(dirs)
        print(f"- same starts/goals across planners, seed {s}: "
              f"{ {True: 'yes', False: 'NO', None: 'not checked (plan_targets.pkl missing)'}[g] }")
        if g is False:
            problems.append(f"goals differ seed {s}")
    print(f"\n**{'All checks pass.' if not problems else 'PROBLEMS: ' + ', '.join(problems)}**\n")

    complete = [sh for sh in runs if all(runs[sh][s] is not None for s in seeds)]
    pooled = {sh: {k: np.concatenate([runs[sh][s][k] for s in seeds])
                   for k in ("success", "dist", "any")} for sh in complete}
    for sh in complete:
        for k in ("cost", "gap"):
            vals = [runs[sh][s][k] for s in seeds]
            pooled[sh][k] = None if any(v is None for v in vals) else np.concatenate(vals)

    # ---- 2. success ----------------------------------------------------------------
    print("## 2. Final success\n")
    print("| Planner | " + " | ".join(f"Seed {s}" for s in seeds) + " | Pooled (95% CI) |")
    print("|---|" + "---|" * len(seeds) + "---|")
    for sh in runs:
        cells = [f"{runs[sh][s]['success'].mean():.2f}" if runs[sh][s] is not None else "—" for s in seeds]
        pool = ci(pooled[sh]["success"].mean(), len(pooled[sh]["success"])) if sh in pooled else "—"
        print(f"| {sh} | " + " | ".join(cells) + f" | {pool} |")
    n = len(next(iter(pooled.values()))["success"]) if pooled else 0
    print(f"\nn = {n} paired episodes.\n")

    # ---- 3. tests --------------------------------------------------------------------
    def tests(title, pairs):
        print(f"## {title}\n")
        print("| Comparison | Only first solves | Only second solves | Exact McNemar p | Significant (α 0.05) |")
        print("|---|---|---|---|---|")
        for x, y in pairs:
            if x in pooled and y in pooled:
                oa, ob, p = mcnemar(pooled[x]["success"], pooled[y]["success"])
                print(f"| {x} vs {y} | {oa} | {ob} | {p:.3g} | {'yes' if p < 0.05 else 'no'} |")
            else:
                print(f"| {x} vs {y} | — | — | — | runs missing |")
        print()
    tests("3a. Primary comparisons", PRIMARY)
    tests("3b. Secondary comparison (identical compute: averaging vs independent starts)", SECONDARY)

    # ---- 4. fooled vs stuck ----------------------------------------------------------
    print("## 4. Failure types (fooled = final model cost below CEM@10's on the same episode)\n")
    if CEM in pooled and pooled[CEM]["cost"] is not None:
        ref = pooled[CEM]["cost"]
        types = {}
        print("| Planner | Failures | Fooled | Stuck |")
        print("|---|---|---|---|")
        for sh in pooled:
            if sh == CEM:
                print(f"| {sh} | {int((~pooled[sh]['success']).sum())} | (not classified) | |")
                continue
            if pooled[sh]["cost"] is None:
                print(f"| {sh} | — | no model_cost | |")
                continue
            fail = ~pooled[sh]["success"]
            fooled = fail & (pooled[sh]["cost"] < ref)
            types[sh] = (fooled, fail & ~fooled)
            print(f"| {sh} | {int(fail.sum())} | {int(fooled.sum())} | {int((fail & ~fooled).sum())} |")
        if "Tuned GD" in types:
            gf, gs = types["Tuned GD"]
            print(f"\nOn the {int(gs.sum())} episodes where tuned GD failed **stuck** and the "
                  f"{int(gf.sum())} where it failed **fooled**, successes by planner:\n")
            print("| Planner | Solves of tuned-GD stuck failures | Solves of tuned-GD fooled failures |")
            print("|---|---|---|")
            for sh in pooled:
                if sh == "Tuned GD":
                    continue
                s_ = pooled[sh]["success"]
                print(f"| {sh} | {int((s_ & gs).sum())} / {int(gs.sum())} | {int((s_ & gf).sum())} / {int(gf.sum())} |")
        print()
    else:
        print("CEM@10 model costs unavailable; not classified.\n")

    # ---- 5. secondary measures ---------------------------------------------------------
    print("## 5. Secondary measures (descriptive; best/any are oracle diagnostics)\n")
    print("| Planner | Mean / median dist | Mean model cost | Mean gap | Best (at), per seed | Any | plan_equiv / run | Solo plan_s per seed | GPU |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sh in runs:
        if sh not in pooled:
            continue
        P = pooled[sh]
        rs = [runs[sh][s] for s in seeds]
        best = ", ".join(f"{r['best_rate']:.2f} ({r['best_at']})" for r in rs)
        plan_s = ", ".join(f"{r['stats'].get('plan_s', float('nan')) / 60:.0f} min" for r in rs)
        pe = ", ".join(str(r["stats"].get("plan_equiv")) for r in rs)
        gpus = sorted({str(r["stats"].get("gpu")) for r in rs})
        cost = f"{P['cost'].mean():.3g}" if P["cost"] is not None else "—"
        gap = f"{P['gap'].mean():.0f}" if P["gap"] is not None else "—"
        print(f"| {sh} | {P['dist'].mean():.1f} / {np.median(P['dist']):.1f} | {cost} | {gap} | "
              f"{best} | {P['any'].mean():.2f} | {pe} | {plan_s} | {', '.join(gpus)} |")
    print("\nFootnotes: episode 0 has the same start/goal states in every seed (simulator seed "
          "`seed × n + 1`); all tuning was on seed 99, with K8/K16 and σ₀ ties broken by mean distance.")


if __name__ == "__main__":
    main()
