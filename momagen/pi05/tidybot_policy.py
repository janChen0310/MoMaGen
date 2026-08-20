"""TidyBot input/output transforms for openpi (pi0.5 finetuning).

Deployed into openpi as src/openpi/policies/tidybot_policy.py. Modeled directly on
libero_policy.py at commit 15a9616 (the maintained custom-LeRobot-finetune reference).

Robot: TidyBot++-style holonomic mobile manipulator in OmniGibson.
  action (11) = [base_vx, base_vy, base_wz, arm_j1..j7 (joint POSITION targets), gripper]
  state  (11) = [base_vx, base_vy, base_wz, arm_j1..j7 (qpos), gripper_qpos]
Cameras: base_0_rgb  <- rear-left mast camera (mount C, 1.20 m, the pi0.5 third-person
         view); left_wrist_0_rgb <- Kinova wrist camera; right wrist = zeros + mask False
         (single-arm robot; mask False is the PI0/PI05 convention -- PI0_FAST does not mask).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# TidyBot action/state dimensionality (before zero-padding to the model's 32).
TIDYBOT_DOF = 11


def make_tidybot_example() -> dict:
    """Random input example (norm-stats sanity / policy server warmup)."""
    return {
        "observation/state": np.random.rand(TIDYBOT_DOF),
        "observation/base_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "pick up the trash and put it in the trash can",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TidybotInputs(transforms.DataTransformFn):
    """Dataset/inference inputs -> model inputs. Used for both training and inference."""

    # Determines which model will be used. Do not change.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/base_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # single-arm robot: no right wrist camera -> zero-pad
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # padding images are masked for pi0/pi0.5 but NOT for pi0-FAST
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TidybotOutputs(transforms.DataTransformFn):
    """Model outputs -> environment actions (inference only): unpad 32 -> 11."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :TIDYBOT_DOF])}
