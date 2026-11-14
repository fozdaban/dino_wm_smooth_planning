#!/usr/bin/env python
"""Replication analysis, exactly as pre-registered in docs/replication_preregistration.md.

usage: python scripts/replication_report.py [results/wall] > results/wall/replication_report.md

Fresh seeds 107, 109, 113 only (n = 150 paired episodes).
  Primary:   smoothed K16 vs tuned GD, exact two-sided McNemar on final success, alpha 0.05.
  Secondary: same test on the per-episode "fooled failure" indicator (failed, and final
             model cost below CEM@10's on the same episode).
  Descriptive: rates with 95% CIs per seed and pooled, fooled/stuck counts, GD + restarts
             if it was run, and phase 6 + replication pooled (250 episodes, NOT a test).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase6_report import ci, goals_match, load, mcnemar, same_code  # noqa: E402

SEEDS = [107, 109, 113]
OLD_SEEDS = [101, 103]
RUNS = {
    "CEM@10": ("cem10", 150_000),
    "Tuned GD": ("gd_adam_lr0.1_randn_R1_S1000", 150_000),
    "Smoothed K16": ("sgd_adam_lr0.1_randn_K16_S62_sig1_geom", 148_800),
    "GD + restarts": ("gd_adam_lr0.1_randn_R8_S125", 150_400),  # optional, descriptive
}
REQUIRED = ["CEM@10", "Tuned GD", "Smoothed K16"]


def pooled(root, short, seeds):
    rs = [load(root, RUNS[short][0], s) for s in seeds]
    if any(r is None for r in rs):
        return None, rs
    P = {k: np.concatenate([r[k] for r in rs]) for k in ("success", "dist")}
    P["cost"] = None if any(r["cost"] is None for r in rs) else np.concatenate([r["cost"] for r in rs])
    return P, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/wall")
    a = ap.parse_args()
    print(f"# Replication report (fresh seeds {', '.join(map(str, SEEDS))})\n")

    data = {sh: pooled(a.root, sh, SEEDS) for sh in RUNS}

    # ---- validity ------------------------------------------------------------------
    print("## 1. Validity checks\n")
    problems = []
    missing = [f"{RUNS[sh][0]}_s{s}" for sh in REQUIRED for s, r in zip(SEEDS, data[sh][1]) if r is None]
    if missing:
        problems.append("missing runs")
        print("**Missing required runs:** " + ", ".join(missing) + "\n")
    present = [(sh, s, r) for sh in RUNS for s, r in zip(SEEDS, data[sh][1]) if r is not None]
    dirty = [f"{sh} s{s}" for sh, s, r in present if r["stats"].get("git_dirty") is not False]
    commits = sorted({r["stats"].get("git_commit") or "?" for _, _, r in present})
    code_same = all(same_code(commits[0], c) for c in commits[1:]) if len(commits) > 1 else True
    bad_pe = [f"{sh} s{s}: {r['stats'].get('plan_equiv')}" for sh, s, r in present
              if r["stats"].get("plan_equiv") != RUNS[sh][1]]
    print(f"- git_dirty false for all runs: {'yes' if not dirty else 'NO: ' + ', '.join(dirty)}")
    print(f"- commits: {', '.join(c[:7] for c in commits)}; same code: "
          f"{ {True: 'yes', False: 'NO', None: 'unknown (run inside the repo)'}[code_same] }")
    print(f"- plan_equiv as expected: {'yes' if not bad_pe else 'NO: ' + '; '.join(bad_pe)}")
    for s in SEEDS:
        dirs = [r["dir"] for sh in RUNS for ss, r in zip(SEEDS, data[sh][1]) if ss == s and r is not None]
        g = goals_match(dirs) if len(dirs) > 1 else None
        print(f"- same starts/goals across planners, seed {s}: "
              f"{ {True: 'yes', False: 'NO', None: 'not checked'}[g] }")
        if g is False:
            problems.append(f"goals differ seed {s}")
    problems += ["dirty or mixed code"] if dirty or code_same is False else []
    problems += ["plan_equiv"] if bad_pe else []
    print(f"\n**{'All checks pass.' if not problems else 'PROBLEMS: ' + ', '.join(problems)}**\n")
    if missing:
        return

    P = {sh: d[0] for sh, d in data.items() if d[0] is not None}
    n = len(P["Tuned GD"]["success"])

    # ---- success ----------------------------------------------------------------------
    print("## 2. Final success\n")
    print("| Planner | " + " | ".join(f"Seed {s}" for s in SEEDS) + f" | Pooled (n = {n}, 95% CI) |")
    print("|---|" + "---|" * len(SEEDS) + "---|")
    for sh in P:
        cells = [f"{r['success'].mean():.2f}" for r in data[sh][1]]
        print(f"| {sh} | " + " | ".join(cells) + f" | {ci(P[sh]['success'].mean(), n)} |")

    # ---- primary ----------------------------------------------------------------------
    oa, ob, p = mcnemar(P["Smoothed K16"]["success"], P["Tuned GD"]["success"])
    print("\n## 3. Primary test: smoothed K16 vs tuned GD (final success)\n")
    print(f"Only K16 solves: **{oa}**; only tuned GD solves: **{ob}**; exact two-sided McNemar "
          f"p = **{p:.3g}** → **{'significant' if p < 0.05 else 'not significant'}** at α = 0.05"
          + (" (K16 better)" if p < 0.05 and oa > ob else " (K16 worse)" if p < 0.05 else "") + ".\n")

    # ---- secondary --------------------------------------------------------------------
    print("## 4. Secondary test: fooled failures (failed, final model cost < CEM@10's)\n")
    ref = P["CEM@10"]["cost"]
    if ref is None or P["Tuned GD"]["cost"] is None or P["Smoothed K16"]["cost"] is None:
        print("model_cost missing; not computed.\n")
    else:
        fooled = {sh: (~P[sh]["success"]) & (P[sh]["cost"] < ref) for sh in P if sh != "CEM@10" and P[sh]["cost"] is not None}
        print("| Planner | Failures | Fooled | Stuck |\n|---|---|---|---|")
        for sh, f in fooled.items():
            fail = ~P[sh]["success"]
            print(f"| {sh} | {int(fail.sum())} | {int(f.sum())} | {int((fail & ~f).sum())} |")
        print(f"| CEM@10 | {int((~P['CEM@10']['success']).sum())} | (reference) | |")
        fa, fb, fp = mcnemar(fooled["Tuned GD"], fooled["Smoothed K16"])
        print(f"\nEpisodes fooled only under tuned GD: **{fa}**; only under K16: **{fb}**; exact two-sided "
              f"McNemar p = **{fp:.3g}** → **{'significant' if fp < 0.05 else 'not significant'}**"
              + (" (K16 fooled less)" if fp < 0.05 and fa > fb else "") + ".\n")

    # ---- descriptive ------------------------------------------------------------------
    print("## 5. Descriptive (no tests)\n")
    for x, y in [("Smoothed K16", "CEM@10"), ("Smoothed K16", "GD + restarts")]:
        if x in P and y in P:
            oa, ob, p = mcnemar(P[x]["success"], P[y]["success"])
            print(f"- {x} vs {y}: only first {oa}, only second {ob} (p = {p:.3g}, descriptive)")
    for sh in P:
        rs = data[sh][1]
        print(f"- {sh}: mean / median final distance {P[sh]['dist'].mean():.1f} / {np.median(P[sh]['dist']):.1f}; "
              f"solo plan_s per seed " + ", ".join(f"{r['stats'].get('plan_s', float('nan')) / 60:.0f} min" for r in rs))
    old = {sh: pooled(a.root, sh, OLD_SEEDS)[0] for sh in ("Smoothed K16", "Tuned GD")}
    if all(v is not None for v in old.values()):
        k = np.concatenate([old["Smoothed K16"]["success"], P["Smoothed K16"]["success"]])
        g = np.concatenate([old["Tuned GD"]["success"], P["Tuned GD"]["success"]])
        oa, ob, p = mcnemar(k, g)
        print(f"- Phase 6 + replication pooled (seeds 101, 103, {', '.join(map(str, SEEDS))}; n = {len(k)}): "
              f"K16 {k.mean():.2f} vs tuned GD {g.mean():.2f}; only K16 {oa}, only GD {ob}, p = {p:.3g}. "
              "Descriptive only: the phase-6 seeds motivated this replication.")


if __name__ == "__main__":
    main()
