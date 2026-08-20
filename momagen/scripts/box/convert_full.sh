#!/bin/bash
set -e
NY=/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen
TASKDIR=demo_src_tidybot_picking_up_trash_task_D0
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate openpi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import ast; ast.parse(open('$NY/momagen/scripts/convert_to_lerobot.py').read()); print('converter parse OK')"
ROOT=~/.cache/huggingface/lerobot/local/tidybot_trash_camc
rm -rf $ROOT
# seeds 3009-3012 = per-episode tmp files (paused shards); 3013-3016 = merged demo.hdf5 (completed)
INPUTS=""
for s in 3009 3010 3011 3012; do INPUTS="$INPUTS $NY/gen_out_camc/shard_seed$s/$TASKDIR/tmp"; done
for s in 3013 3014 3015 3016; do INPUTS="$INPUTS $NY/gen_out_camc/shard_seed$s/$TASKDIR/demo.hdf5"; done
echo "INPUTS:$INPUTS"
python $NY/momagen/scripts/convert_to_lerobot.py \
  --inputs $INPUTS \
  --root $ROOT --repo-id local/tidybot_trash_camc \
  --task "pick up the trash and put it in the trash can" --fps 15
echo "CONVERT_FULL_RC=$?"
ls $ROOT/meta/ 2>/dev/null
python -c "import json;i=json.load(open('$ROOT/meta/info.json'));print('episodes',i.get('total_episodes'),'frames',i.get('total_frames'),'fps',i.get('fps'))"
