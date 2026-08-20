#!/bin/bash
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate openpi
python - << 'PEOF'
import gcsfs, os, time
fs = gcsfs.GCSFileSystem(token="anon")
src = "openpi-assets/checkpoints/pi05_base/params"
dst = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_base/params")
files = fs.find(src)
print("files:", len(files), "total GB: %.2f" % (sum(fs.info(f)["size"] for f in files) / 1e9), flush=True)
t0 = time.time()
for i, f in enumerate(files):
    rel = f[len(src):].lstrip("/")
    out = os.path.join(dst, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == fs.info(f)["size"]:
        continue
    fs.get(f, out)
    print("[%d/%d] %s (%.0fs)" % (i + 1, len(files), rel, time.time() - t0), flush=True)
print("PREWARM_FULL_DONE %.0fs" % (time.time() - t0), flush=True)
PEOF
echo "PREWARM_RC=$?"
du -sh ~/.cache/openpi/openpi-assets/checkpoints/pi05_base
