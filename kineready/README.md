# KineReady — a learned IK-feasibility head for base-pose scoring

`base_pose_metric/` scores a candidate base pose on four components. Three of them are cheap
(distance + visibility + collision ≈ 1 ms for a whole batch). The fourth, exact CuRobo IK, costs
**0.34 s per pose** and is 99.7% of the total — which is what stops the metric from being used as a
dense online reward for training a navigation policy's endpoint selection.

KineReady replaces that one term with a small learned model, and leaves the other three exact.

## What the model is (and is not)

The model answers exactly one question:

> given an end-effector target expressed **in the robot's base frame**, does an IK solution exist?

It is **object- and task-agnostic**. It has never seen a soda can, a kitchen, or a scene. The
upstream planner supplies whatever target it wants at query time. Task-specific logic — the trash
can's grasp pose, the K=8 symmetry orbit — lives in `examples/`, above both the model and the
reward core.

This works because of one coordinate change:

```
T_E^B = T_B^W(b)^-1 @ T_O^W @ T_E^O
```

Expressing the target in the base frame absorbs *where the base stands* into the transform, turning
mobile-base placement into a fixed-arm reachability query. What remains is a fixed property of the
arm, so the model input is 9 numbers (xyz + 6D rotation) and the model is a ~59k-parameter MLP.
`tests/test_kineready_frames.py` checks this transform against an independent implementation,
because a wrong transform yields a model that fits its own wrong labels perfectly and only fails
once the base moves.

## Layout

| file | role | needs a simulator? |
|---|---|---|
| `frames.py` | the base-frame transform, 6D rotation encoding, noisy-OR | no |
| `robot_env.py` | minimal **empty-scene** TidyBot env for labeling | yes |
| `teacher.py` | batched exact CuRobo IK labeler + control suite | yes |
| `datagen.py` | fk / uniform / boundary-concentrated robust samplers, resumable shards | yes |
| `dataset.py` | shard reader, **geometric** splits, duplicate guard | no |
| `model.py` | MLP, ensemble, temperature calibration, ECE / FPR@recall | no |
| `train.py` | training loop and metrics | no |
| `reward.py` | `ReadinessReward` — the inference API a policy calls | no |
| `eval_offline.py` | metrics vs baselines (distance band, cylinder, RM4D histogram, kNN) | no |
| `eval_ranking.py` | ranking vs the exact oracle, behind a **frame gate** | yes |
| `examples/` | Stage-0 benchmark, Stage-1 runner, latency, the trash-task worked example | mixed |

## Running it

```bash
# Stage 0 — prove the labeler is correct, then measure its throughput
OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=5 python kineready/examples/stage0_benchmark.py

# Stage 1 — generate (pilot first, then full). Shards resume if the session dies.
OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=5 \
  python kineready/examples/stage1_datagen.py --pilot --out kineready_data/pilot

# Stage 2 — train (no simulator needed; run wherever the data is)
python -m kineready.train --data kineready_data/full --out kineready_models/kineready.pt

# Stage 3 — evaluate
python -m kineready.eval_offline --data kineready_data/full --model kineready_models/kineready.pt
OMNIGIBSON_HEADLESS=1 python -m kineready.eval_ranking --model kineready_models/kineready.pt
python kineready/examples/latency.py
```

Using it in the existing metric:

```python
from kineready.reward import ReadinessReward
metric = BasePoseMetric(robot, ik_predictor=ReadinessReward.from_checkpoint("kineready.pt"))
results = metric.evaluate(base_poses, target)      # p_kin instead of a 0.34 s solve
```

**One caveat that is not a detail:** a probability is not a joint configuration, so with a learned
predictor there is nothing to run the reach-collision check on. It is reported as `None` — "not
checked" — never as `False`. Re-verify the top candidates with exact IK whenever reach-collision
matters.

## Measured

| | |
|---|---|
| teacher throughput | 398 labels/s (batch 64, one RTX 4090) — 135× serial exact IK |
| full ~2.3 M-solve dataset | ≈ 1.6 h |
| inference, 4096 poses × 8 grasps | 51 ms GPU / 136 ms CPU |
| speedup vs exact IK | 10,000× (CPU) to 27,000× (GPU) |

## Why the controls are not ceremony

Every teacher run embeds known-answer checks and **aborts the run** if any fails, because a
silently-wrong labeler poisons the dataset and everything trained on it, and unlike a rendering bug
it is invisible in every artifact downstream.

Three controls, each catching a class the others cannot:

- **identity** — forward kinematics of the robot's own rest pose must label reachable 100%.
- **fk samples** — poses from random valid configurations, reachable by construction.
- **far target** — 6 m away, unreachable for any tabletop arm.

The identity control earns its place. A quaternion-ordering bug (`compute_trajectories` applies its
xyzw→wxyz permutation *outside* the `is_local` guard, so callers must pass xyzw even for local
targets) scrambled every target's rotation while leaving positions correct. IK then succeeded
whenever the scrambled orientation happened to be reachable, producing a plausible 72%
"reachability" rate that no geometric story explains. The far-target control is blind to it —
nothing at 6 m is reachable at any orientation. Only the identity control could see it, and it did,
immediately: 50% where 100% is the only correct answer.

The general rule that came out of it: **at least one control must be orientation-sensitive.**

## The frame trap (read this before reusing the teacher)

For a holonomic base, `is_local=True` is a **misnomer**. Measured on TidyBot: the link curobo
treats as its base, `base_footprint_x`, sits at the **world origin with identity orientation no
matter where the robot is** — the world pose lives entirely in the base joints. So the wrapper's
world→base conversion is the identity, and "local" targets are really WORLD targets.

The teacher's documented frame — the target in the arm's own base frame — is therefore correct
only while the arm is at the origin. That is exactly how every shard was generated (empty scene,
robot at the origin), so **the dataset is sound**. But build a teacher in a furnished scene with
the robot at (4.79, −1.28) and ask about a target 0.3 m in front of it, and curobo is asked about a
point 0.3 m from the origin, 4.8 m from the arm: everything returns unreachable, silently.

`IKTeacher` now asserts the base joints are at zero and refuses to construct otherwise.

Nothing already in place could have caught this. The controls are self-consistent round-trips
through the same frame, so a shared wrong frame is invisible to them. The frame gate ran in the
empty scene at the origin, where the correct and incorrect frames coincide — **a gate that cannot
fail is not a gate.** What did catch it: comparing against a ground truth built the other way
round (set the base joints, give the target in world coordinates, physically move the robot and
re-solve), plus a negative control that scores a target against the *wrong* base pose and must
do measurably worse.

Verified after the fix, against that frame-independent ground truth:

| | teacher | model | negative control |
|---|---|---|---|
| broad targets, base up to ±2.5 m from origin | 100% | 100% | 60% |
| kitchen-grid regime (target at 1.02 m, tilted 23°) | 100% | 93.8% | 75% |
