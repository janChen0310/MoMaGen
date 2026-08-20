"""Insert the TidyBot data-config factory + pi05_tidybot_trash TrainConfig into openpi's
src/openpi/training/config.py (commit 15a9616). Idempotent: skips if already present.

Design (mirrors LeRobotLiberoDataConfig / pi05_libero, the maintained custom-finetune
reference):
- repack maps OUR LeRobot v3 keys -> the policy-server keys used by TidybotInputs;
- action_sequence_keys=("action",) because our dataset stores the LeRobot-standard
  singular "action" (LIBERO's conversion used plural "actions");
- NO delta-action transform: base dims are velocities (delta-ifying them against a
  velocity state would be wrong) and pi0.5-DROID's own guidance finetunes on absolute
  joint-position actions;
- Pi0Config(pi05=True, action_horizon=20): ~1 s chunk at our 20 Hz (PI used 50 @ 50 Hz);
  discrete_state_input=False copies pi05_libero;
- batch_size 32 + fsdp_devices 4 for 4x RTX 4090 (24 GB, no NVLink);
- weights from gs://openpi-assets/checkpoints/pi05_base/params (the finetuning base).
"""
import re

P = "/home/ubuntu/DATA4/backup_root_home/yhu/openpi/src/openpi/training/config.py"
s = open(P).read()

if "pi05_tidybot_trash" in s:
    print("already patched")
    raise SystemExit(0)

# 1) import the policy module alongside the other policy imports
anchor = "import openpi.policies.libero_policy as libero_policy"
assert anchor in s
s = s.replace(anchor, anchor + "\nimport openpi.policies.tidybot_policy as tidybot_policy", 1)

# 2) the data-config factory, inserted right before RLDSDroidDataConfig
factory = '''
@dataclasses.dataclass(frozen=True)
class LeRobotTidybotDataConfig(DataConfigFactory):
    """TidyBot (OmniGibson MoMaGen) LeRobot v3 dataset -> pi0.5. See tidybot_policy.py."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Map our LeRobot dataset keys to the keys the inference client sends.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/base_image": "observation.images.base",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[tidybot_policy.TidybotInputs(model_type=model_config.model_type)],
            outputs=[tidybot_policy.TidybotOutputs()],
        )
        # NO delta-action transform: dims 0-2 are base VELOCITIES (delta vs a velocity
        # state would corrupt them) and the arm uses absolute joint-position targets,
        # which is pi0.5-DROID's own recommended finetuning action space.

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


'''
anchor = "@dataclasses.dataclass(frozen=True)\nclass RLDSDroidDataConfig(DataConfigFactory):"
assert anchor in s
s = s.replace(anchor, factory + anchor, 1)

# 3) the TrainConfig, inserted right before the pi05_libero entry
train_cfg = '''    TrainConfig(
        name="pi05_tidybot_trash",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=20, discrete_state_input=False),
        data=LeRobotTidybotDataConfig(
            repo_id="local/tidybot_trash_camc",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        fsdp_devices=4,
    ),
'''
anchor = '    TrainConfig(\n        name="pi05_libero",'
assert anchor in s
s = s.replace(anchor, train_cfg + anchor, 1)

open(P, "w").write(s)
import ast
ast.parse(s)
print("PATCH OK: pi05_tidybot_trash + LeRobotTidybotDataConfig inserted")
