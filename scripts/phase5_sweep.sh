#!/usr/bin/env bash
# Phase 5: smoothed GD (planner=smooth_gd) on the TUNING seed (99) only.
# OPT, LR, INIT are the tuned GD settings from phase 3, passed exactly as to its stage C;
# SIGMAS are three sigma0 values from the probe.
#
#   STAGE=S1 OPT=adam LR=0.1 INIT=randn0.3 SIGMAS="0.1 0.3 1.0" bash scripts/phase5_sweep.sh
#       sigma0 x {const, geom to sigma0/10}, K=8 S=125
#   STAGE=S2 OPT=adam LR=0.1 INIT=randn0.3 SIG=0.3 SCHED=geom bash scripts/phase5_sweep.sh
#       K at fixed K x S, best S1 setting
#
# Every run costs K x S x 3 = 3000 plan-rollouts per episode = CEM at 10 iterations
# (K=16 S=62: 2976).
# Env: CHUNK (episode_chunk over the n_evals x K rows, default 25; use one value for every
#      lease run), SEED (default 99), CKPT (checkpoint base, default $HOME/ckpt), DRY=1 to
#      print only.
# Parallel workers: start the same command in several tmux windows. Each run is claimed
# with a lock directory (results/wall/.lock_<name>_s<seed>), so workers never duplicate
# a run; finished runs (results/wall/<name>_s<seed>) are skipped. A failed run releases
# its lock, so starting the command again retries it.
set -uo pipefail
STAGE=${STAGE:?set STAGE=S1 or S2}
SEED=${SEED:-99}
CHUNK=${CHUNK:-25}
CKPT=${CKPT:-$HOME/ckpt}
: "${OPT:?set OPT}" "${LR:?set LR}" "${INIT:?set INIT (randn, or randn0.3 for init_scale 0.3)}"
export WANDB_MODE=disabled
COMMON=(model_name=wall_single planner=smooth_gd goal_source=random_state goal_H=5 n_evals=50
        seed=$SEED ckpt_base_path=$CKPT +planner.oracle_stop=false +planner.episode_chunk=$CHUNK)
mkdir -p results/wall

run() {  # run <name> <hydra overrides...>
  local name=$1; shift
  local out=results/wall/${name}_s${SEED}
  local lock=results/wall/.lock_${name}_s${SEED}
  if [ -e "$out" ]; then echo "skip $name (done)"; return 0; fi
  if [ -n "${DRY:-}" ]; then echo ">>> $name: $*"; return 0; fi
  mkdir "$lock" 2>/dev/null || { echo "skip $name (another worker has it)"; return 0; }
  local dir=plan_outputs/$(date +%Y%m%d%H%M%S)_${name}_s${SEED}
  echo ">>> $(date +%H:%M) $name: $*"
  if python plan.py "${COMMON[@]}" "$@" hydra.run.dir="$dir" \
       && bash scripts/save_run.sh "$dir" "${name}_s${SEED}"; then
    rmdir "$lock"
  else
    rmdir "$lock"; echo "FAILED: $name (see $dir)"
  fi
}

# same INIT handling as phase3_sweep.sh stage C
init=(planner.sample_type=randn)
[ "$INIT" = randn0.3 ] && init=(planner.sample_type=randn planner.init_scale=0.3)
base=(planner.optimizer=$OPT planner.lr=$LR "${init[@]}")
prefix=sgd_${OPT}_lr${LR}_${INIT}

sched_args() {  # sched_args <sigma0> <const|geom>: overrides for that schedule
  case $2 in
    const) echo "planner.sigma0=$1 planner.sigma_schedule=const" ;;
    geom)  local smin
           smin=$(python -c "import sys; print(f'{float(sys.argv[1]) / 10:g}')" "$1") || exit 1
           echo "planner.sigma0=$1 planner.sigma_schedule=geom planner.sigma_min=$smin" ;;
    *)     echo "unknown schedule $2 (const or geom)" >&2; exit 1 ;;
  esac
}

case $STAGE in
S1) : "${SIGMAS:?set SIGMAS, e.g. SIGMAS=\"0.1 0.3 1.0\"}"
    for sig in $SIGMAS; do
      for sched in const geom; do
        sa=$(sched_args "$sig" "$sched") || exit 1
        run ${prefix}_K8_S125_sig${sig}_${sched} "${base[@]}" $sa \
            planner.K=8 planner.opt_steps=125
      done
    done ;;
S2) : "${SIG:?set SIG (best S1 sigma0)}" "${SCHED:?set SCHED (best S1 schedule: const or geom)}"
    sa=$(sched_args "$SIG" "$SCHED") || exit 1
    # K=1 cannot be antithetic (needs even K)
    run ${prefix}_K1_S1000_sig${SIG}_${SCHED} "${base[@]}" $sa \
        planner.K=1 planner.opt_steps=1000 planner.antithetic=false
    run ${prefix}_K4_S250_sig${SIG}_${SCHED} "${base[@]}" $sa planner.K=4 planner.opt_steps=250
    run ${prefix}_K16_S62_sig${SIG}_${SCHED} "${base[@]}" $sa planner.K=16 planner.opt_steps=62 ;;
*)  echo "unknown STAGE $STAGE"; exit 1 ;;
esac
