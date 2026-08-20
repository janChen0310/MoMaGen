# Box-only runner scripts

These ran on the remote GPU box and existed **only there** — on a tree with no version control.
They are committed here so the pipeline is reproducible, not because they are portable.

**They contain hardcoded absolute paths** (`/home/ubuntu/DATA4/backup_root_home/yhu/...`,
`/home/ubuntu/anaconda3/...`). Adjust before running anywhere else.

Note the box has **two trees**: code runs from `DATA4/backup_root_home/yhu/MoMaGen`, but the
installed `omnigibson` package and therefore `get_dataset_path("custom_dataset")` resolve to
`/home/ubuntu/yhu/MoMaGen`. Asset edits made in the DATA4 copy are silently inert.

| Script | Stage | What it does |
|---|---|---|
| `jc_setup_openpi.sh` | setup | Creates the `openpi` conda env and installs it |
| `jc_prewarm_pi05.sh` | setup | Downloads the 12 GB pi05_base weights file-by-file over anonymous gcsfs. `maybe_download` had cached an 8 KB marker as a fake success |
| `run_shard_camc.sh` | generate | One generation shard. Cap at 4 concurrent per box — 8 drove the 192-core box to load-average 452 and produced zero demos in 1.5 h |
| `convert_full.sh` | convert | Merges shard outputs into one LeRobot v3.0 dataset under the HF cache root, where openpi looks for it |
| `full_train.sh` | train | The FSDP-8 finetune. `fsdp_devices=4` OOMs at init (~26.8 GB needed vs 22.8 GB usable per 4090) |
| `jc_serve_and_roll.sh` | evaluate | Starts the openpi policy server on one GPU, waits for the port, runs `rollout_pi05.py` in the momagen env on another, then tears the server down |

Two gotchas these encode, both learned the hard way:

- **Norm stats must be computed on CPU** (`JAX_PLATFORMS=cpu`). A multi-GPU mesh imposes
  `batch_size % ndev` and crashes.
- **`HF_HUB_OFFLINE=1` always** — the box cannot reach huggingface.co, though GCS is reachable.
