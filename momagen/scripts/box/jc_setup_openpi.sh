#!/bin/bash
set -x
D=/home/ubuntu/DATA4/backup_root_home/yhu
cd $D
rm -rf openpi
git clone ssh://git@ssh.github.com:443/Physical-Intelligence/openpi.git
cd $D/openpi || { echo CLONE_FAILED; exit 1; }
git log -1 --format="%H %cs"
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda env list | grep -q "envs/openpi" || conda create -y -n openpi python=3.11
conda activate openpi
python -V
cd $D/openpi
pip install -e . 2>&1 | tail -8
python -c "import openpi; print('OPENPI_IMPORT_OK', openpi.__file__)"
echo "OPENPI_SETUP_RC=$?"
