#!/usr/bin/env python
"""Collect saved runs into one table (phase 2 harness).

usage: python scripts/collect_results.py [results/wall] [--heldout-seeds 101 103] [--csv out.csv]

Runs are grouped by their resolved configuration (.hydra/config.yaml), not by folder
name. The group key is every config setting except the seed and Hydra/path keys
(hydra, saved_folder, ckpt_base_path, wandb_logging): the full planner block,
model_name, model_epoch, goal_H, goal_source, n_evals, objective, wm_eval_mode, ...
Runs whose keys differ are never pooled; each group gets a label summarising the
settings that distinguish it (e.g. "gd steps=400"), and the differing settings are
printed. Note that settings that do not change results (rollout_chunk, episode_chunk,
n_plot_samples) are part of the key too, so runs that differ only in them are not pooled.

Pooled rows use only the --heldout-seeds; runs with other seeds (e.g. the tuning seed
99) are reported as separate tuning rows. A pooled group warns if its runs come from
different git commits, or if any run has git_dirty: true (both from run_stats.json).

Columns (rates as defined in scripts/episode_rates.py, shared with phase1_rates.py):
  final    success of the returned plan (95% normal interval when pooled)
  best     best single evaluation (iteration N or the final one) and where it occurred
  any      fraction of episodes solved at any evaluation, final included (oracle)
  dist     mean final state distance          gap   mean per-episode model-simulator gap
  cost     mean model (planning) cost of the returned plan; needs episodes.jsonl with
           model_cost (runs from phase 3 on). Low cost + failure = model fooled.
  plan_eq  plan-rollout equivalents (forward + 2 x backward), in thousands
  plan_s   planning time excluding evaluation (falls back to wall-clock for old runs)
Runs made before the harness have no compute counters or episodes.jsonl; those cells
are blank, and such runs are left out of pooled rows (with a warning).
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared module lives next to this file
from episode_rates import FINAL, pool_rates, read_episodes, run_rates  # noqa: E402

SEED_IN_NAME = re.compile(r"_s(\d+)$")
NOT_IN_KEY = {"seed", "hydra", "saved_folder", "ckpt_base_path", "wandb_logging"}
SHORT = {"opt_steps": "steps"}  # label abbreviations


def flatten(d, prefix=""):
    """Nested dict -> {dotted.key: json value}, so configs compare and print leaf by leaf."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = json.dumps(v, sort_keys=True)
    return out


def load_run(d):
    if not (d / "run_stats.json").is_file():
        return None
    stats = json.loads((d / "run_stats.json").read_text())
    cfg_path = d / ".hydra" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else None

    name_seed = SEED_IN_NAME.search(d.name)
    name_seed = int(name_seed.group(1)) if name_seed else None
    seed = cfg.get("seed", name_seed) if cfg else name_seed
    if cfg and name_seed is not None and name_seed != seed:
        print(f"warning: {d.name}: folder name says seed {name_seed}, config says {seed}; "
              f"using {seed}", file=sys.stderr)

    if cfg is None:
        print(f"warning: {d.name}: no .hydra/config.yaml; it gets its own group", file=sys.stderr)
        key = {"__no_config__": json.dumps(d.name)}
    else:
        key = flatten({k: v for k, v in cfg.items() if k not in NOT_IN_KEY})

    row = {
        "run": d.name,
        "seed": seed,
        "key": key,
        "planner": (cfg or {}).get("planner", {}).get("name", "?"),
        "final": stats.get("final_success_rate"),
        "best": None, "best_at": None, "any": None, "dist": None, "gap": None, "cost": None,
        "plan_equiv": stats.get("plan_equiv"),
        "enc_images": stats.get("enc_images"),
        "plan_s": stats.get("plan_s", stats.get("wall_clock_s")),
        "peak_gpu_mb": stats.get("peak_gpu_mb"),
        "gpu": stats.get("gpu"),
        "git_commit": stats.get("git_commit"),
        "git_dirty": stats.get("git_dirty"),
        "oracle_stop": stats.get("oracle_stop"),
        "_rates": None,
    }
    try:
        lines, repeated = read_episodes(d)
    except FileNotFoundError:
        return row
    if repeated:
        print(f"warning: {d.name}: repeated filenames in episodes.jsonl; using the last of each",
              file=sys.stderr)
    try:
        rates = run_rates(lines)
    except ValueError as e:
        print(f"warning: {d.name}: {e}; episode data ignored", file=sys.stderr)
        return row
    final_line = lines.get(FINAL, {})
    gap_key = "div_visual_emb" if "div_visual_emb" in final_line else "gap_visual_emb"
    row.update(
        final=rates["final_rate"] if rates["final"] is not None else row["final"],
        best=rates["best_rate"], best_at=rates["best_at"], any=rates["any_rate"],
        dist=float(np.mean(final_line["state_dist"])) if final_line.get("state_dist") else None,
        gap=float(np.mean(final_line[gap_key])) if final_line.get(gap_key) else None,
        cost=float(np.mean(final_line["model_cost"])) if final_line.get("model_cost") else None,
        _rates=rates,
    )
    return row


def group_labels(groups):
    """Label each group by the planner name plus the settings in which it differs from
    other groups of the same planner (groups of different planners differ in their
    planner block anyway; the name already tells them apart).
    Returns {group: label} and {planner name: [(setting, {group: value})]} of differences."""
    dicts = {g: dict(g) for g in groups}
    configured = [g for g in groups if "__no_config__" not in dicts[g]]

    def short(setting):
        leaf = setting[len("planner."):] if setting.startswith("planner.") else setting
        head, _, last = leaf.rpartition(".")
        return (head + "." if head else "") + SHORT.get(last, last)

    def value(g, setting):
        v = dicts[g].get(setting)
        return "-" if v is None else str(json.loads(v))

    by_planner = defaultdict(list)
    for g in configured:
        by_planner[value(g, "planner.name")].append(g)

    labels, differences = {}, {}
    for g in groups:
        if g not in configured:
            labels[g] = f"?config:{json.loads(dicts[g]['__no_config__'])}"
    for name, pool in by_planner.items():
        settings = sorted(set().union(*(dicts[g] for g in pool)))
        differ = [s for s in settings if len({dicts[g].get(s) for g in pool}) > 1]
        differences[name] = [(s, {g: value(g, s) for g in pool}) for s in differ]
        for g in pool:
            labels[g] = " ".join([name] + [f"{short(s)}={value(g, s)}" for s in differ])
    return labels, differences


def ci(p, n):
    h = 1.96 * math.sqrt(max(p * (1 - p), 0) / n)
    return f"{p:.2f} [{max(0, p - h):.2f},{min(1, p + h):.2f}]"


def fmt(v, kind):
    if v is None:
        return ""
    if kind == "rate":
        return f"{v:.2f}"
    if kind == "c":
        return f"{v:.3g}"
    if kind == "k":
        return f"{v / 1e3:.1f}"
    if kind == "M":
        return f"{v / 1e6:.2f}"
    if kind == "s":
        return f"{v / 60:.0f} min" if v < 3600 else f"{v / 3600:.2f} h"
    if kind == "f1":
        return f"{v:.1f}"
    return str(v)


def git_tag(r):
    commit = (r["git_commit"] or "?")[:7]
    return commit + (" dirty" if r["git_dirty"] else "" if r["git_dirty"] is not None else " ?")


def main():
    ap = argparse.ArgumentParser(description="Collect saved planning runs into one table.")
    ap.add_argument("root", nargs="?", default="results/wall")
    ap.add_argument("--heldout-seeds", type=int, nargs="+", default=[101, 103],
                    help="seeds pooled into the reported rows; other seeds are tuning rows")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    heldout = set(a.heldout_seeds)

    rows = [r for d in sorted(Path(a.root).iterdir()) if d.is_dir() and (r := load_run(d))]
    if not rows:
        sys.exit(f"no runs found in {a.root}")

    groups = defaultdict(list)
    for r in rows:
        r["group"] = tuple(sorted(r["key"].items()))
        groups[r["group"]].append(r)
    labels, differences = group_labels(groups)
    for r in rows:
        r["label"] = labels[r["group"]]
        r["role"] = "held-out" if r["seed"] in heldout else "tuning"

    head = ["run", "settings", "seed", "role", "final", "best (at)", "any", "dist", "gap", "cost",
            "plan_eq (k)", "enc img (M)", "plan time", "peak GB", "git"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in rows:
        best = f"{fmt(r['best'], 'rate')} ({r['best_at']})" if r["best"] is not None else ""
        print("| " + " | ".join([
            r["run"], r["label"], fmt(r["seed"], ""), r["role"], fmt(r["final"], "rate"), best,
            fmt(r["any"], "rate"), fmt(r["dist"], "f1"), fmt(r["gap"], "f1"), fmt(r["cost"], "c"),
            fmt(r["plan_equiv"], "k"), fmt(r["enc_images"], "M"), fmt(r["plan_s"], "s"),
            fmt(r["peak_gpu_mb"] and r["peak_gpu_mb"] / 1024, "f1"), git_tag(r),
        ]) + " |")

    print(f"\nPooled over held-out seeds {sorted(heldout)} (95% normal interval), one row per configuration:")
    pooled_any = False
    for g, rs in groups.items():
        held = [r for r in rs if r["role"] == "held-out"]
        if not held:
            continue
        pooled_any = True
        usable = [r for r in held if r["_rates"] is not None]
        warnings = []
        missing = sorted(heldout - {r["seed"] for r in held})
        if missing:
            warnings.append(f"no run for held-out seed(s) {missing}")
        by_seed = defaultdict(list)
        for r in held:
            by_seed[r["seed"]].append(r["run"])
        for seed, runs in sorted(by_seed.items()):
            if len(runs) > 1:
                warnings.append(f"seed {seed} appears in {len(runs)} runs ({', '.join(runs)}): "
                                "same problems, so its episodes are counted more than once")
        if len(usable) < len(held):
            warnings.append("left out (no usable episodes.jsonl): "
                            + ", ".join(r["run"] for r in held if r["_rates"] is None))
        commits = {r["git_commit"] for r in held}
        if None in commits:
            warnings.append("git commit unknown for: "
                            + ", ".join(r["run"] for r in held if r["git_commit"] is None))
        if len(commits - {None}) > 1:
            warnings.append("runs come from different git commits: "
                            + ", ".join(f"{r['run']}={r['git_commit'][:7]}" for r in held if r["git_commit"]))
        dirty = [r["run"] for r in held if r["git_dirty"]]
        if dirty:
            warnings.append("uncommitted changes (git_dirty) in: " + ", ".join(dirty))

        seeds = sorted(r["seed"] for r in usable)
        if usable:
            p = pool_rates([r["_rates"] for r in usable])
            n = p["n_eps"]
            pe = [r["plan_equiv"] for r in usable if r["plan_equiv"] is not None]
            parts = [f"seeds {seeds}", f"n={n}"]
            parts.append(f"final {ci(p['final_rate'], n)}" if p["final_rate"] is not None else "final n/a")
            parts.append(f"any {ci(p['any_rate'], n)}" if p["any_rate"] is not None else "any n/a")
            if p["best_rate"] is not None:
                parts.append(f"best {p['best_rate']:.2f} ({p['best_at']})")
            if pe:
                parts.append(f"plan_eq/run {np.mean(pe) / 1e3:.1f}k")
            print(f"  {labels[g]:<28} " + "  ".join(parts))
        else:
            print(f"  {labels[g]:<28} nothing to pool")
        for w in warnings:
            print(f"  {'':<28} warning: {w}")
    if not pooled_any:
        print("  (no runs with held-out seeds)")

    tuning = [r for r in rows if r["role"] == "tuning"]
    if tuning:
        print(f"\nTuning runs (seeds outside {sorted(heldout)}), not pooled:")
        for r in tuning:
            best = f"best {r['best']:.2f} ({r['best_at']})" if r["best"] is not None else ""
            any_ = f"any {r['any']:.2f}" if r["any"] is not None else "any n/a"
            final = f"final {r['final']:.2f}" if r["final"] is not None else "final n/a"
            print(f"  {r['label']:<28} seed {r['seed']}  {r['run']}  {final}  {any_}  {best}".rstrip())

    if len(groups) > 1:
        print("\nConfigurations are pooled separately; runs with different settings are never pooled.")
        if len(differences) > 1:
            print(f"  planners: {', '.join(differences)} (their planner settings differ by design)")
        for name, diffs in differences.items():
            if diffs:
                print(f"  {name}: settings that differ between its {len(diffs[0][1])} configurations:")
                for s, vals in diffs:
                    print(f"    {s}:  " + "  |  ".join(f"{labels[g]}: {v}" for g, v in vals.items()))
        for g in groups:
            if labels[g].startswith("?config:"):
                print(f"  {labels[g]}: no .hydra/config.yaml, so it is never pooled with other runs")

    if a.csv:
        cols = ["run", "label", "seed", "role", "planner", "final", "best", "best_at", "any",
                "dist", "gap", "cost", "plan_equiv", "enc_images", "plan_s", "peak_gpu_mb", "gpu",
                "git_commit", "git_dirty", "oracle_stop"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
