# π0.5 finetuning — the openpi side

Three pieces live here. None of them fork openpi; they patch a clean upstream checkout so it stays
easy to update.

| File | Purpose |
|---|---|
| `tidybot_policy.py` | Deployed to `openpi/src/openpi/policies/`. Maps our observations onto PI0/PI05's image slots and slices the model's 32-dim action back to TidyBot's 11 |
| `patch_openpi_config.py` | Idempotently inserts the `LeRobotTidybotDataConfig` and the `TrainConfig` entry into `openpi/src/openpi/training/config.py` |
| `openpi-lerobot-0.4.4-bridge.patch` | The lerobot version bridge — see below |

## The lerobot bridge

openpi pins the lerobot git rev only in `pyproject.toml`'s uv sources, so `pip install -e .` pulls
PyPI **lerobot 0.3.3 = dataset v2.1**, which cannot read the **v3.0** datasets `convert_to_lerobot.py`
writes. The failure is not obviously a version problem when you hit it.

Fix, against openpi commit `15a9616`:

```bash
pip install lerobot==0.4.4 h5py
git apply momagen/pi05/openpi-lerobot-0.4.4-bridge.patch
```

The patch does two things: moves the import from `lerobot.common.datasets.lerobot_dataset` to
`lerobot.datasets.lerobot_dataset` (the module moved in 0.4), and shims `meta.tasks` back to
`dict[int, str]` — 0.4 returns a DataFrame, and `PromptFromLeRobotTask` expects the dict.

These were hand edits on the box for a long time, with no patch file anywhere. If openpi is ever
re-cloned they have to be redone, so they are captured here.

## Two constraints that are not optional

- **Norm stats must be computed on CPU**: `JAX_PLATFORMS=cpu python scripts/compute_norm_stats.py`.
  A multi-GPU mesh imposes `batch_size % ndev` and crashes. There is no CLI escape hatch — both
  `transform_dataset` and `transform_iterable_dataset` raise if the stats file is missing.
- **`HF_HUB_OFFLINE=1`** on the box, which cannot reach huggingface.co (GCS is reachable).

## Known stale, fixed in the next task's finetune

- `patch_openpi_config.py` still says `fsdp_devices=4`, which **OOMs at init** (~26.8 GB needed vs
  22.8 GB usable per 24 GB 4090, no NVLink). The run that produced the 65 %-success checkpoint
  used **8**.
- `action_horizon=20` dates from when the data was 20 Hz. After `JC_DS_RATIO=2` the data is 15 Hz,
  so the chunk is 1.33 s rather than the intended ~1 s. `rollout_pi05.py --execute-horizon`
  inherits the same mismatch.
