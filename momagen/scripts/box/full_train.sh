#!/bin/bash
set -x
NY=/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen
OP=/home/ubuntu/DATA4/backup_root_home/yhu/openpi
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate openpi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
ROOT=~/.cache/huggingface/lerobot/local/tidybot_trash_camc

# --- verify the converted dataset before training (fail-fast) ---
python - << 'PEOF' || { echo "DATASET_VERIFY_FAILED — aborting before train"; exit 1; }
import json, os, glob
import numpy as np
root=os.path.expanduser("~/.cache/huggingface/lerobot/local/tidybot_trash_camc")
info=json.load(open(f"{root}/meta/info.json"))
print("VERIFY episodes:", info["total_episodes"], "frames:", info["total_frames"], "fps:", info["fps"])
feats=info["features"]
print("state shape:", feats["observation.state"]["shape"], "action:", feats["action"]["shape"])
print("base img:", feats["observation.images.base"]["shape"], "wrist:", feats["observation.images.wrist"]["shape"])
assert info["fps"]==15, "fps must be 15"
assert feats["observation.state"]["shape"]==[11], feats["observation.state"]["shape"]
assert tuple(feats["observation.images.base"]["shape"])==(224,224,3)
print("DATASET_VERIFY_OK")
PEOF

# --- norm stats on FULL dataset (CPU: single-device mesh avoids batch%ndev crash) ---
cd $OP
JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py --config-name pi05_tidybot_trash
echo "NORMSTATS_FULL_RC=$?"

# --- full finetune: 20k steps, FSDP-8 ---
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,9,10
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python scripts/train.py pi05_tidybot_trash --exp-name=trash_camC_full_v1 --overwrite 2>&1 | tail -60
echo "FULL_TRAIN_RC=${PIPESTATUS[0]}"
