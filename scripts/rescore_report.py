#!/usr/bin/env python
"""Phase 6b E3 report: does the smoothed cost of a final plan predict simulator success?

usage: python scripts/rescore_report.py results/wall_6b/e3_rescore_s99 [--set NAME=a,b ...]

Reads rescore.npz (planning/rescore.py). Sections:
 1. per candidate: success, mean J_sigma, AUC of "lower J_sigma -> success" per sigma
 2. picking per episode the candidate with the lowest J_sigma, for candidate sets:
    every run that saved its restarts (pick among its R restarts; sigma 0 reproduces the
    run's own choice), all candidates, every pair of runs, and --set NAME=a,b where a, b
    are run names (all their candidates) or candidate labels. Shown with the oracle
    ("any candidate succeeds") and exact McNemar of each sigma's pick vs the sigma-0 pick.

Decision rule for E4 (smoothed restarts), fixed Nov 13 before any E3 result: proceed only
if, on seed 99, picking among the restarts by J_sigma for some sigma > 0 solves at least
5 more of the 50 episodes than picking by J (sigma 0), AND that sigma's AUC on the
restart candidates is higher than sigma 0's. Seed 99 is the tuning seed: E3 numbers are
exploratory; a selection rule that passes is confirmed on fresh seeds 107/109/113.
"""
import argparse
import itertools
import math
from pathlib import Path

import numpy as np


def auc(score_low_good, success):
    """P(score of a random success < score of a random failure); ties count 1/2."""
    s, f = score_low_good[success], score_low_good[~success]
    if len(s) == 0 or len(f) == 0:
        return float("nan")
    less = (s[:, None] < f[None]).sum() + 0.5 * (s[:, None] == f[None]).sum()
    return float(less) / (len(s) * len(f))


def mcnemar(a, b):
    oa, ob = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = oa + ob, min(oa, ob)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return oa, ob, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="rescore run folder (or the rescore.npz path)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=a,b,...")
    a = ap.parse_args()
    p = Path(a.run)
    d = np.load(p if p.suffix == ".npz" else p / "rescore.npz")
    labels, groups = list(d["labels"]), list(d["groups"])
    sig, J, succ = d["sigmas"], d["J"], d["success"]
    C, S, n = J.shape
    if not bool(d["evaluated"]):
        raise SystemExit("rescore ran with evaluate=false: no simulator successes to analyse")
    sh = [f"σ={s:g}" for s in sig]

    print(f"# E3 rescore report ({n} episodes, {C} candidates, M = {int(d['M'])})\n")
    print("## 1. Per candidate: success, mean J_σ, AUC(lower J_σ → success)\n")
    print("| Candidate | Success | " + " | ".join(f"mean J {h}" for h in sh) + " | "
          + " | ".join(f"AUC {h}" for h in sh) + " |")
    print("|---|---|" + "---|" * (2 * S))
    for c in range(C):
        print(f"| {labels[c]} | {succ[c].mean():.2f} | "
              + " | ".join(f"{J[c, s].mean():.3g}" for s in range(S)) + " | "
              + " | ".join(f"{auc(J[c, s], succ[c]):.2f}" for s in range(S)) + " |")
    restart_groups = [g for g in dict.fromkeys(groups) if groups.count(g) > 1]
    for g in restart_groups:
        idx = [c for c in range(C) if groups[c] == g]
        print(f"| **all restarts of {g} pooled** | {succ[idx].mean():.2f} | "
              + " | ".join(f"{J[idx, s].mean():.3g}" for s in range(S)) + " | "
              + " | ".join(f"{auc(J[idx, s].ravel(), succ[idx].ravel()):.2f}" for s in range(S)) + " |")

    # ---- candidate sets ------------------------------------------------------------
    sets = {}
    for g in restart_groups:
        sets[f"{g} (its {groups.count(g)} restarts)"] = [c for c in range(C) if groups[c] == g]
    uniq = list(dict.fromkeys(groups))
    if len(uniq) > 1:
        sets["all candidates"] = list(range(C))
        for g1, g2 in itertools.combinations(uniq, 2):
            sets[f"{g1} + {g2}"] = [c for c in range(C) if groups[c] in (g1, g2)]
    for spec in a.set:
        name, _, items = spec.partition("=")
        idx = sorted({c for it in items.split(",") for c in range(C)
                      if labels[c] == it or groups[c] == it})
        if not idx:
            raise SystemExit(f"--set {spec}: no candidate matches")
        sets[name] = idx

    print("\n## 2. Pick per episode the candidate with the lowest J_σ\n")
    print("| Candidate set | Oracle (any) | " + " | ".join(f"pick {h}" for h in sh)
          + " | " + " | ".join(f"{h} vs σ=0 (only / only, p)" for h in sh[1:]) + " |")
    print("|---|---|" + "---|" * (2 * S - 1))
    for name, idx in sets.items():
        idx = np.array(idx)
        oracle = succ[idx].any(0).mean()
        picks = []
        for s in range(S):
            best = idx[J[idx, s].argmin(0)]           # (n,) candidate per episode
            picks.append(succ[best, np.arange(n)])
        cmp = []
        for s in range(1, S):
            oa, ob, pv = mcnemar(picks[s], picks[0])
            cmp.append(f"{oa} / {ob}, {pv:.2g}")
        print(f"| {name} | {oracle:.2f} | " + " | ".join(f"{pk.mean():.2f}" for pk in picks)
              + " | " + " | ".join(cmp) + " |")

    # ---- decision rule ------------------------------------------------------------
    print("\n## 3. E4 decision rule (pre-declared)\n")
    if not restart_groups:
        print("No run with saved restarts: rule not applicable.")
        return
    for g in restart_groups:
        idx = np.array([c for c in range(C) if groups[c] == g])
        base = succ[idx[J[idx, 0].argmin(0)], np.arange(n)].sum()
        a0 = auc(J[idx, 0].ravel(), succ[idx].ravel())
        passing = []
        for s in range(1, S):
            got = succ[idx[J[idx, s].argmin(0)], np.arange(n)].sum()
            a_s = auc(J[idx, s].ravel(), succ[idx].ravel())
            if got - base >= 5 and a_s > a0:
                passing.append(f"σ={sig[s]:g} ({got} vs {base} solved, AUC {a_s:.2f} vs {a0:.2f})")
        print(f"- {g}: " + ("**PASS** at " + "; ".join(passing) if passing
                            else f"no pass (σ=0 pick solves {base}/{n}, AUC {a0:.2f})"))


if __name__ == "__main__":
    main()
