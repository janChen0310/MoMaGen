#!/bin/bash
# Generate a shard of tidybot_grasp_can demos.
#   usage: run_shard_grasp.sh <GPU_ID> <SEED> <NUM_DEMOS> [CORESET e.g. 0-45]
#
# Modelled on run_shard_camc.sh (the trash-task fleet script that produced 106 demos), with the
# grasp-task specifics layered on:
#   JC_CAN_REGION      the 0.04-inset re-sweep: 105 usable cells, so the can lands anywhere a
#                      grasp station exists rather than only where the source demo happened to sit.
#   JC_BASE_SAMPLE_*   must bracket the region's standoffs, or the base sampler burns all 200 tries
#                      on "Could not find a valid pose near the object". Across the viable cells the
#                      nearest grasp station sits between 0.41 m and 0.86 m, median 0.53 -- so the
#                      first 0.36-0.46 band reached only 27 of 105 cells and generated zero demos.
#                      0.36-0.68, paired with JC_CAN_REQUIRE_SERVABLE (keep only cells whose nearest
#                      station is <= 0.60 m, stamped by refine_can_region.py), keeps the can region
#                      and the sampling band consistent with each other.
set -u
GPU=$1; SEED=$2; NUM=${3:-20}; CORES=${4:-}
NY=${MOMAGEN_REPO:-/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen}
exec > $NY/gen_grasp_seed${SEED}.log 2>&1
export PYTHONPATH=$NY:$NY/robomimic:$NY/BEHAVIOR-1K/OmniGibson:$NY/BEHAVIOR-1K/bddl:${PYTHONPATH:-}
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=$GPU          # physical GPU selector; never with OMNIGIBSON_GPU_ID
export MOMAGEN_REPLAY_NUM_REPEAT=1
export JC_WORLDCAM=1
export MOMAGEN_REPO=$NY
# ---------- task shape ----------
export JC_CAN_REGION=$NY/can_region2.json
export JC_CAN_REQUIRE_SERVABLE=1
export JC_BASE_SAMPLE_LO=0.36
export JC_BASE_SAMPLE_HI=0.68
# ---------- [JC PERF] speed levers (see momagen-generation-perf) ----------
export JC_DS_RATIO=2            # 30Hz->15Hz replay downsample; halves the dominant replay phase
export JC_MP_INTER_DIST=0.025   # coarser MP interpolation; ~2.5x fewer transit steps
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1 PYTHONBREAKPOINT=0
cd $NY
echo "START $(date -u) GRASP SHARD gpu=$GPU seed=$SEED num=$NUM cores=${CORES:-all}"
PIN=""
[ -n "$CORES" ] && PIN="taskset -c $CORES"   # disjoint core sets keep 4 shards from thrashing
$PIN /home/ubuntu/anaconda3/envs/momagen/bin/python -u momagen/scripts/generate_dataset.py \
  --config $NY/momagen/datasets/configs/demo_src_tidybot_grasp_can_task_D0.json \
  --num_demos $NUM --bimanual --robot_type TidyBot \
  --folder $NY/gen_out_grasp/shard_seed${SEED} --seed $SEED --auto-remove-exp
echo "GRASP_SEED${SEED}_RC=$? $(date -u)"
