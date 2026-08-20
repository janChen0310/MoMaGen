#!/bin/bash
# usage: run_shard_camc.sh <GPU_ID> <SEED> <NUM_DEMOS> [CORESET e.g. 0-45]
GPU=$1; SEED=$2; NUM=${3:-38}; CORES=${4:-}
NY=/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen
exec > $NY/gen_camc_seed${SEED}.log 2>&1
export PYTHONPATH=$NY:$NY/robomimic:$NY/BEHAVIOR-1K/OmniGibson:$NY/BEHAVIOR-1K/bddl:$PYTHONPATH
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
# Multi-GPU: physical GPU via CUDA_VISIBLE_DEVICES (this tree lacks the old curobo-device
# patch that OMNIGIBSON_GPU_ID needed). Comment warns CUDA_VISIBLE_DEVICES can desync Vulkan;
# it has been the working selector on this box for the camC fleet.
export CUDA_VISIBLE_DEVICES=$GPU
export MOMAGEN_REPLAY_NUM_REPEAT=1
export JC_WORLDCAM=1
# ---------- [JC PERF] max-aggressive speed levers (see momagen-generation-perf) ----------
export JC_DS_RATIO=2            # 30Hz->15Hz replay downsample; halves the dominant replay phase
export JC_MP_INTER_DIST=0.025   # coarser MP interpolation; ~2.5x fewer transit steps
# Tame per-shard BLAS thread pools so 4 shards don't oversubscribe the 192-core box.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONBREAKPOINT=0
cd $NY
echo "START $(date -u) CAMC SHARD gpu=$GPU seed=$SEED num=$NUM cores=${CORES:-all} JC_DS_RATIO=$JC_DS_RATIO JC_MP_INTER_DIST=$JC_MP_INTER_DIST"
PIN=""
[ -n "$CORES" ] && PIN="taskset -c $CORES"   # confine this shard's threads to a disjoint core set
$PIN /home/ubuntu/DATA4/conda/envs/momagen/bin/python -u momagen/scripts/generate_dataset.py \
  --config momagen/datasets/configs/demo_src_tidybot_picking_up_trash_task_D0.json \
  --num_demos $NUM --bimanual --robot_type TidyBot \
  --folder $NY/gen_out_camc/shard_seed${SEED} --seed $SEED --auto-remove-exp
echo "CAMC_SEED${SEED}_RC=$? $(date -u)"
