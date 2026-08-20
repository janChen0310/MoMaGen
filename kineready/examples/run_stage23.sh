#!/usr/bin/env bash
# Stages 2 and 3 end to end, once Stage 1 has written its shards.
#
# Ordered so that the cheapest check that can invalidate everything runs first: training and the
# offline metrics need no simulator and finish in minutes, and if the model is bad there is no
# point booting Isaac for the ranking evaluation.
set -euo pipefail

REPO=${REPO:-/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen}
DATA=${DATA:-$REPO/kineready_data/full}
MODEL=${MODEL:-$REPO/kineready_models/kineready.pt}
GPU=${GPU:-5}

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate momagen
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMNIGIBSON_HEADLESS=1
unset DISPLAY

echo "=================== STAGE 2: train ==================="
python -m kineready.train --data "$DATA" --out "$MODEL" --members 5 --epochs 30

echo
echo "=================== STAGE 3a: offline metrics vs baselines ==================="
python -m kineready.eval_offline --data "$DATA" --model "$MODEL" \
    --out "$REPO/kineready_eval_offline.json"

echo
echo "=================== STAGE 3b: latency ==================="
python kineready/examples/latency.py

echo
echo "=================== STAGE 3c: ranking vs exact oracle ==================="
python -m kineready.eval_ranking --model "$MODEL" --n-configs 20 --n-poses 64 \
    --out "$REPO/kineready_eval_ranking.json"

echo
echo "=================== STAGE 3d: trash-task symmetry augmentation ==================="
python kineready/examples/trash_task.py --model "$MODEL" --n-poses 256 --k 8

echo
echo "ALL STAGES DONE"
