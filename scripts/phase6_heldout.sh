#!/usr/bin/env bash
# Phase 6: the pre-registered held-out runs (claude/phase6_preregistration.md).
# Five planners x seeds 101, 103, ONE RUN AT A TIME so every plan_s is a solo-GPU timing.
# Overrides are copied from the seed-99 runs' .hydra/overrides.yaml, so each planner's
# held-out runs have the same resolved config (and pool in collect_results.py).
#
#   bash scripts/phase6_heldout.sh 2>&1 | tee -a p6.log       # all 10 runs
#   SEEDS=101 bash scripts/phase6_heldout.sh ...              # one seed (short lease)
#   DRY=1 bash scripts/phase6_heldout.sh                      # print the commands only
#
# Finished runs (results/wall/<name>_s<seed>) are skipped, so rerunning resumes.
# Ctrl-C / kill releases the lock and stops the whole script.
set -uo pipefail
SEEDS=${SEEDS:-"101 103"}
CHUNK=25                       # fixed: part of the config key, must match seed 99
CKPT=${CKPT:-$HOME/ckpt}
export WANDB_MODE=disabled
mkdir -p results/wall

# ---- pre-flight (skipped for DRY) ----------------------------------------------
if [ -z "${DRY:-}" ]; then
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ABORT: tracked files are modified; runs would be git_dirty. git status:"; git status --short --untracked-files=no; exit 1
  fi
  if ls -d results/wall/.lock_* >/dev/null 2>&1; then
    echo "ABORT: stale lock folders (check nothing is running, then rmdir them):"; ls -d results/wall/.lock_*; exit 1
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "ABORT: another process is using the GPU; phase 6 timings must be solo:"
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

run() {  # run <name> <seed> <hydra overrides...>
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
  run gd_adam_lr0.1_randn_R8_S125             $S planner=gd +planner.episode_chunk=$CHUNK "${GD[@]}" planner.restarts=8 planner.opt_steps=125
  run sgd_adam_lr0.1_randn_K16_S62_sig1_geom  $S planner=smooth_gd +planner.episode_chunk=$CHUNK "${SM[@]}" planner.K=16 planner.opt_steps=62
  run sgd_adam_lr0.1_randn_K8_S125_sig1_geom  $S planner=smooth_gd +planner.episode_chunk=$CHUNK "${SM[@]}" planner.K=8 planner.opt_steps=125
done
[ -n "${DRY:-}" ] || echo "phase 6 loop finished $(date '+%m-%d %H:%M'); next: python scripts/phase6_report.py"
