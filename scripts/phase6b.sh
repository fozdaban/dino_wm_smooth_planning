#!/usr/bin/env bash
# Phase 6b exploration on the TUNING seed (99) only; results go to results/wall_6b/.
# Phase-6 results (results/wall/) are never touched. See roadmap.md "Phase 6b".
#
#   E2     higher sigma floor: smoothed K16x62 and K8x125, sigma 1 -> 0.333 (geom)   2 runs
#   E3prep GD + restarts R8x125 that also saves all 8 restart plans                   1 run
#   E1     budget scaling: CEM 40 it, GD R1x4000, GD R8x500 (one long run each gives
#          the whole success-vs-budget curve), smoothed K16 x {125, 250}              5 runs
#   E3     rescore saved seed-99 plans with the smoothed cost (needs E3prep)          1 run
#
#   bash scripts/phase6b.sh 2>&1 | tee -a p6b_w1.log            # worker 1
#   bash scripts/phase6b.sh 2>&1 | tee -a p6b_w2.log            # worker 2 (2nd tmux window)
#   STAGES="E3" bash scripts/phase6b.sh ...                      # one stage
#   DRY=1 bash scripts/phase6b.sh                                # print the commands only
# Two workers share the GPU (each run is claimed with a lock folder; finished runs are
# skipped, so rerunning resumes). Timings from these runs are NOT solo timings.
# Afterwards: python scripts/budget_curve.py; python scripts/collect_results.py results/wall_6b;
#             python scripts/rescore_report.py results/wall_6b/e3_rescore_s99
set -uo pipefail
STAGES=${STAGES:-"E2 E3prep E1 E3"}
SEED=99
CHUNK=25                       # same as every lease run
CKPT=${CKPT:-$HOME/ckpt}
ENV=wall_6b
OUT=results/$ENV
export WANDB_MODE=disabled
mkdir -p "$OUT"

if [ -z "${DRY:-}" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ABORT: tracked files are modified; runs would be git_dirty. git status:"
  git status --short --untracked-files=no; exit 1
fi

COMMON=(model_name=wall_single goal_source=random_state goal_H=5 n_evals=50 seed=$SEED
        ckpt_base_path=$CKPT)
GD=(planner=gd +planner.episode_chunk=$CHUNK +planner.oracle_stop=false planner.optimizer=adam
    planner.lr=0.1 planner.sample_type=randn)
SM=(planner=smooth_gd +planner.episode_chunk=$CHUNK +planner.oracle_stop=false planner.optimizer=adam
    planner.lr=0.1 planner.sample_type=randn planner.sigma0=1 planner.sigma_schedule=geom)

LOCK=""
trap '[ -n "$LOCK" ] && rmdir "$LOCK" 2>/dev/null; echo "interrupted"; exit 130' INT TERM

run() {  # run <name> <hydra overrides...>
  local name=$1; shift
  local out=$OUT/${name}_s${SEED}
  if [ -e "$out" ]; then echo "skip ${name} (done)"; return 0; fi
  if [ -n "${DRY:-}" ]; then echo "python plan.py $* ${COMMON[*]}"; return 0; fi
  LOCK=$OUT/.lock_${name}_s${SEED}
  mkdir "$LOCK" 2>/dev/null || { echo "skip ${name} (another worker has it)"; LOCK=""; return 0; }
  local dir=plan_outputs/$(date +%Y%m%d%H%M%S)_6b_${name}_s${SEED}
  echo ">>> $(date '+%m-%d %H:%M') ${name}"
  if python plan.py "$@" "${COMMON[@]}" hydra.run.dir="$dir" \
       && bash scripts/save_run.sh "$dir" "${name}_s${SEED}" "$ENV"; then
    echo "<<< $(date '+%m-%d %H:%M') ${name} done"
  else
    echo "FAILED: ${name} (see $dir)"
  fi
  rmdir "$LOCK"; LOCK=""
}

R8SAVE=$OUT/gd_adam_lr0.1_randn_R8_S125_save_s${SEED}

for st in $STAGES; do
  case $st in
  E2)
    run sgd_adam_lr0.1_randn_K16_S62_sig1_geom0.333 "${SM[@]}" planner.sigma_min=0.333 planner.K=16 planner.opt_steps=62
    run sgd_adam_lr0.1_randn_K8_S125_sig1_geom0.333 "${SM[@]}" planner.sigma_min=0.333 planner.K=8 planner.opt_steps=125 ;;
  E3prep)
    run gd_adam_lr0.1_randn_R8_S125_save "${GD[@]}" planner.restarts=8 planner.opt_steps=125 +planner.save_restarts=true ;;
  E1)  # longest first, so two workers finish close together
    run cem40 planner=cem planner.opt_steps=40 +planner.rollout_chunk=100 +planner.oracle_stop=false
    run gd_adam_lr0.1_randn_R1_S4000 "${GD[@]}" planner.restarts=1 planner.opt_steps=4000
    run gd_adam_lr0.1_randn_R8_S500 "${GD[@]}" planner.restarts=8 planner.opt_steps=500
    run sgd_adam_lr0.1_randn_K16_S250_sig1_geom "${SM[@]}" planner.sigma_min=0.1 planner.K=16 planner.opt_steps=250
    run sgd_adam_lr0.1_randn_K16_S125_sig1_geom "${SM[@]}" planner.sigma_min=0.1 planner.K=16 planner.opt_steps=125 ;;
  E3)
    if [ -z "${DRY:-}" ] && [ ! -f "$R8SAVE/restart_actions.pt" ]; then
      echo "E3: $R8SAVE not finished yet; run STAGES=E3 once it is"; continue
    fi
    # candidate 0 (K16) is what the run returns: its final evaluation must reproduce 0.38
    cands=(results/wall/sgd_adam_lr0.1_randn_K16_S62_sig1_geom_s99
           results/wall/sgd_adam_lr0.1_randn_K8_S125_sig1_geom_s99
           results/wall/endpoint_gd_s99 results/wall/endpoint_cem10_s99 "$R8SAVE")
    list=$(printf '"%s",' "${cands[@]}"); list="[${list%,}]"
    run e3_rescore planner=rescore "planner.runs=$list" ;;
  *) echo "unknown stage $st"; exit 1 ;;
  esac
done
[ -n "${DRY:-}" ] || echo "worker finished $(date '+%m-%d %H:%M')"
