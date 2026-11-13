#!/usr/bin/env bash
# Copy the report-relevant files of one planning run into results/.
# plan_targets.pkl holds the start/goal images (~1.2 MB per episode, 60 MB at n_evals=50);
# it is kept for reproducibility but gitignored, so only the small files get committed.
# usage: scripts/save_run.sh <plan_outputs/RUN_DIR> <name> [env]
#   e.g. scripts/save_run.sh plan_outputs/20261109120126_wall_single_gH5 gd_s99
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <run_dir> <name> [env=wall]"; exit 1; }
src=${1%/}; name=$2; env=${3:-wall}
dst=results/$env/$name
[ -f "$src/logs.json" ] || { echo "no logs.json in $src (run unfinished?)"; exit 1; }
[ -e "$dst" ] && { echo "refusing to overwrite existing $dst"; exit 1; }
mkdir -p "$dst"
for f in logs.json episodes.jsonl run_stats.json plan_targets.pkl final_actions.pt probe.npz restart_actions.pt rescore.npz; do
  if [ -f "$src/$f" ]; then cp "$src/$f" "$dst/"
  else case $f in probe.npz|restart_actions.pt|rescore.npz) ;; *) echo "warning: $f missing in $src";; esac; fi
done
cp -r "$src/.hydra" "$dst/"
echo "$src" > "$dst/source_dir.txt"
# a run with no overrides has no "- " lines; grep then exits 1, which must not abort the script
grep -h -- '- ' "$src/.hydra/overrides.yaml" | sed 's/^/  /' || true
echo "saved $src -> $dst"
