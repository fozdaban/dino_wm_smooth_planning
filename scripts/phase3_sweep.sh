#!/usr/bin/env bash
# Phase 3: tune gradient planning on the TUNING seed (99) only.
#
#   STAGE=A bash scripts/phase3_sweep.sh                      # optimizer x learning rate
#   STAGE=B OPT=adam LR=0.1 bash scripts/phase3_sweep.sh      # initialization, best of A
#   STAGE=C OPT=adam LR=0.1 INIT=randn bash scripts/phase3_sweep.sh   # restarts at CEM@10 budget
#
# Env: CHUNK (episode_chunk, default 10 = laptop-safe), SEED (default 99), DRY=1 to print only.
# Each run gets its own hydra dir, so several stages can run at once (e.g. separate tmux
# windows on the lease GPU). Existing results/wall/<name>_s<seed> folders are skipped.
set -uo pipefail
STAGE=${STAGE:?set STAGE=A, B or C}
SEED=${SEED:-99}
CHUNK=${CHUNK:-10}
COMMON=(model_name=wall_single planner=gd goal_source=random_state goal_H=5 n_evals=50
        seed=$SEED +planner.oracle_stop=false +planner.episode_chunk=$CHUNK)

run() {  # run <name> <hydra overrides...>
  local name=$1; shift
  local out=results/wall/${name}_s${SEED}
  if [ -e "$out" ]; then echo "skip $name (exists)"; return 0; fi
  local dir=plan_outputs/$(date +%Y%m%d%H%M%S)_${name}_s${SEED}
  echo ">>> $name: $*"
  [ -n "${DRY:-}" ] && return 0
  python plan.py "${COMMON[@]}" "$@" hydra.run.dir="$dir" \
    && bash scripts/save_run.sh "$dir" "${name}_s${SEED}" \
    || echo "FAILED: $name (see $dir)"
}

case $STAGE in
A)  # 400 steps (the phase-1 GD budget), randn init, upstream noise 0.003
    for lr in 0.3 1 3; do
      run gd_sgd_lr${lr} planner.optimizer=sgd planner.lr=$lr planner.opt_steps=400
    done
    for lr in 0.01 0.03 0.1 0.3; do
      run gd_adam_lr${lr} planner.optimizer=adam planner.lr=$lr planner.opt_steps=400
    done ;;
B)  : "${OPT:?set OPT}" "${LR:?set LR}"
    run gd_${OPT}_lr${LR}_normzero planner.optimizer=$OPT planner.lr=$LR planner.opt_steps=400 \
        planner.sample_type=normzero
    run gd_${OPT}_lr${LR}_init0.3 planner.optimizer=$OPT planner.lr=$LR planner.opt_steps=400 \
        planner.init_scale=0.3 ;;
C)  : "${OPT:?set OPT}" "${LR:?set LR}" "${INIT:?set INIT (randn, or randn0.3 for init_scale 0.3)}"
    init=(planner.sample_type=randn)
    [ "$INIT" = randn0.3 ] && init=(planner.sample_type=randn planner.init_scale=0.3)
    # every run costs R x S x 3 = 3000 plan-rollouts per episode = CEM at 10 iterations
    # (+ R final no-grad rollouts to pick the restart)
    for rs in "1 1000" "2 500" "4 250" "8 125"; do
      set -- $rs
      run gd_${OPT}_lr${LR}_${INIT}_R$1_S$2 planner.optimizer=$OPT planner.lr=$LR "${init[@]}" \
          planner.restarts=$1 planner.opt_steps=$2
    done ;;
*)  echo "unknown STAGE $STAGE"; exit 1 ;;
esac
