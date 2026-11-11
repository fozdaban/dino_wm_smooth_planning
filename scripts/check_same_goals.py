#!/usr/bin/env python
"""Check that planning runs faced identical problems (same starts and goals).

usage: python scripts/check_same_goals.py <run_dir> <run_dir> [...]
Run dirs are plan_outputs/... or results/... folders containing plan_targets.pkl
(gitignored, so use local copies). Exits 1 if any run differs from the first.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

KEYS = ["state_0", "state_g"]


def load(d):
    p = Path(d) / "plan_targets.pkl"
    if not p.is_file():
        sys.exit(f"missing {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


runs = sys.argv[1:]
if len(runs) < 2:
    sys.exit(__doc__)
ref = load(runs[0])
ok = True
for d in runs[1:]:
    t = load(d)
    same = all(np.array_equal(np.asarray(ref[k]), np.asarray(t[k])) for k in KEYS)
    ok &= same
    print(f"{'same' if same else 'DIFFERENT'}  {d}")
print("all runs share the same goals" if ok else "goal mismatch: these runs are not comparable")
sys.exit(0 if ok else 1)
