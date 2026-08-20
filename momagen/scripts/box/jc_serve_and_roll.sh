#!/bin/bash
# usage: jc_serve_and_roll.sh <SERVE_GPU> <ROLL_GPU> <CKPT_DIR> <NUM_EP> <PORT>
SERVE_GPU=$1; ROLL_GPU=$2; CKPT=$3; NUM_EP=${4:-3}; PORT=${5:-8321}
NY=/home/ubuntu/DATA4/backup_root_home/yhu/MoMaGen
OP=/home/ubuntu/DATA4/backup_root_home/yhu/openpi
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh

# ---------- policy server (openpi env, JAX) ----------
conda activate openpi
cd $OP
CUDA_VISIBLE_DEVICES=$SERVE_GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 HF_HUB_OFFLINE=1 \
  setsid nohup python scripts/serve_policy.py --port $PORT \
  policy:checkpoint --policy.config pi05_tidybot_trash --policy.dir $CKPT \
  > $NY/jc_serve_policy.log 2>&1 < /dev/null &
SPID=$!
echo "SERVER_PID=$SPID (gpu $SERVE_GPU, port $PORT), loading 12GB params..."

# wait until the port is actually listening (model load can take a few min), or server dies
ready=0
for i in $(seq 1 90); do
  sleep 5
  if ! kill -0 $SPID 2>/dev/null; then echo "SERVER_DIED"; tail -30 $NY/jc_serve_policy.log; exit 1; fi
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then ready=1; echo "SERVER_READY after ~$((i*5))s"; break; fi
done
[ $ready -eq 0 ] && { echo "SERVER_TIMEOUT"; tail -30 $NY/jc_serve_policy.log; kill $SPID; exit 1; }
grep -E "metadata|Loading|Serving|ready|error|Error" $NY/jc_serve_policy.log | tail -5

# ---------- rollout client (momagen env, Isaac) ----------
export PYTHONPATH=$NY:$NY/robomimic:$NY/BEHAVIOR-1K/OmniGibson:$NY/BEHAVIOR-1K/bddl:$PYTHONPATH
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export CUDA_VISIBLE_DEVICES=$ROLL_GPU
export MOMAGEN_REPLAY_NUM_REPEAT=1 JC_WORLDCAM=1 PYTHONUNBUFFERED=1 PYTHONBREAKPOINT=0
cd $NY
echo "=== ROLLOUT START ($NUM_EP episodes, gpu $ROLL_GPU) ==="
/home/ubuntu/DATA4/conda/envs/momagen/bin/python -u momagen/scripts/rollout_pi05.py \
  --config momagen/datasets/configs/demo_src_tidybot_picking_up_trash_task_D0.json \
  --host localhost --port $PORT --num-episodes $NUM_EP --out-dir $NY/rollout_out
echo "ROLLOUT_RC=$?"
echo "=== stopping policy server ==="
kill $SPID 2>/dev/null
