#!/usr/bin/env bash
# Replication on FRESH seeds 107, 109, 113 (docs/replication_preregistration.md).
# Same planner settings as phase 6 (configs identical apart from seed), ONE RUN AT A TIME.
#
#   bash scripts/replication_run.sh 2>&1 | tee -a p7rep.log       # CEM@10, tuned GD, smoothed K16
#   WITH_R8=1 bash scripts/replication_run.sh ...                  # also GD + restarts (descriptive)
#   SEEDS=107 bash scripts/replication_run.sh ...                  # one seed (short lease)
#   DRY=1 bash scripts/replication_run.sh                          # print the commands only
# Finished runs are skipped (resumable). Ctrl-C releases the lock and stops.
# Afterwards: python scripts/replication_report.py > results/wall/replication_report.md
set -uo pipefail
SEEDS=${SEEDS:-"107 109 113"}
CHUNK=25
CKPT=${CKPT:-$HOME/ckpt}
export WANDB_MODE=disabled
mkdir -p results/wall

if [ -z "${DRY:-}" ]; then
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ABORT: tracked files are modified; runs would be git_dirty:"; git status --short --untracked-files=no; exit 1
  fi
  if [ ! -f docs/replication_preregistration.md ] || ! git ls-files --error-unmatch docs/replication_preregistration.md >/dev/null 2>&1; then
    echo "ABORT: commit docs/replication_preregistration.md before running"; exit 1
  fi
  if ls -d results/wall/.lock_* >/dev/null 2>&1; then
    echo "ABORT: stale lock folders (check nothing is running, then rmdir them):"; ls -d results/wall/.lock_*; exit 1
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "ABORT: another process is using the GPU; timings must be solo:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 1
  fi
  echo "code commit $(git rev-parse --short HEAD), free disk $(df -h --output=avail . | tail -1)"
fi

COMMON=(model_name=wall_single goal_source=random_state goal_H=5 n_evals=50
        ckpt_base_path=$CKPT +planner.oracle_stop=false)
GD=(planner.optimizer=adam planner.lr=0.1 planner.sample_type=randn)
SM=(planner.optimizer=adam planner.lr=0.1 planner.sample_type=randn planner.sigma0=1
    planner.sigma_schedule=geom planner.sigma_min=0.1)

LOCK=""
trap '[ -n "$LOCK" ] && rmdir "$LOCK" 2>/dev/null; echo "interrupted"; exit 130' INT TERM

run() {  # run <name> <seed> <hydra overrides...>  (same as scripts/phase6_heldout.sh)
  local name=$1 seed=$2; shift 2
  local out=results/wall/${name}_s${seed}
  if [ -e "$out" ]; then echo "skip ${name}_s${seed} (done)"; return 0; fi
  if [ -n "${DRY:-}" ]; then echo "python plan.py $* ${COMMON[*]} seed=$seed"; return 0; fi
  LOCK=results/wall/.lock_${name}_s${seed}
  mkdir "$LOCK" || { echo "ABORT: $LOCK exists"; exit 1; }
  local dir=plan_outputs/$(date +%Y%m%d%H%M%S)_${name}_s${seed}
  echo ">>> $(date '+%m-%d %H:%M') ${name}_s${seed}"
  if python plan.py "$@" "${COMMON[@]}" seed=$seed hydra.run.dir="$dir" \
       && bash scripts/save_run.sh "$dir" "${name}_s${seed}"; then
    echo "<<< $(date '+%m-%d %H:%M') ${name}_s${seed} done"
  else
    echo "FAILED: ${name}_s${seed} (see $dir)"
  fi
  rmdir "$LOCK"; LOCK=""
}

for S in $SEEDS; do
  run cem10                                   $S planner=cem planner.opt_steps=10 +planner.rollout_chunk=100
  run gd_adam_lr0.1_randn_R1_S1000            $S planner=gd +planner.episode_chunk=$CHUNK "${GD[@]}" planner.restarts=1 planner.opt_steps=1000
  run sgd_adam_lr0.1_randn_K16_S62_sig1_geom  $S planner=smooth_gd +planner.episode_chunk=$CHUNK "${SM[@]}" planner.K=16 planner.opt_steps=62
done
if [ -n "${WITH_R8:-}" ]; then
  for S in $SEEDS; do
    run gd_adam_lr0.1_randn_R8_S125           $S planner=gd +planner.episode_chunk=$CHUNK "${GD[@]}" planner.restarts=8 planner.opt_steps=125
  done
fi
[ -n "${DRY:-}" ] || echo "replication loop finished $(date '+%m-%d %H:%M'); next: python scripts/replication_report.py"
