# Integrating a New Robot Embodiment (worked example: TidyBot++)

This guide explains how to add a brand-new robot embodiment to MoMaGen so you can
**generate** and **collect** demonstration trajectories on it, using
[TidyBot++](https://tidybot2.github.io) (Wu et al., CoRL 2024) — a single-arm
*holonomic* mobile manipulator — as the worked example.

> **Read first:** [Generating Data](generating-data.md) and [Creating Custom
> Tasks](custom-tasks.md). This guide assumes you understand the
> phase → arm → subtask task spec and the source-demo → `datagen_info` →
> generation pipeline.

---

## 0. TL;DR

MoMaGen is architecturally built around the **bimanual R1** robot, in
**OmniGibson / Isaac Sim**. Adding a new embodiment means threading it through
**four** layers:

| Layer | Where | What you provide |
|------|-------|------------------|
| **A. Simulator robot + asset** | `BEHAVIOR-1K/OmniGibson/omnigibson/robots/` | A `TidyBot` robot class + a **USD** model + CuRobo configs |
| **B. MoMaGen env-interface** | `momagen/env_interfaces/omnigibson.py` | A subclass mapping eef-pose ↔ action, object/term signals |
| **C. Robot/task config** | `momagen/utils/robot_config.py`, `momagen/configs/`, `momagen/datasets/base_configs/` | Controller config, reset pose, task spec JSON |
| **D. Source demo + annotation** | `momagen/datasets/source_og/`, `prepare_src_dataset.py` | One teleop demo → robomimic HDF5 with `datagen_info` |

**The single biggest assumption to break:** MoMaGen's generation core
(`DataGenerator.generate`) is implemented **bimanual-only** — it iterates
`range(2)` over `arm_left`/`arm_right` and stacks a per-arm `4×4` eef pose into an
`8×4` block. TidyBot++ has **one** arm. You must either (a) thread a *phantom*
second arm whose `object_ref` is always `None` (fast, hacky), or (b) generalize
the core to N-arm (clean, more work). See §4.

**The single biggest asset gap:** neither tidybot repo ships a **USD** model —
only URDF (`tidybot_platform/.../urdf/tidybot_isaac.urdf`) and MuJoCo
(`tidybot2/models/stanford_tidybot/tidybot.xml`). OmniGibson requires USD, so you
must run the URDF→USD importer. See §5-A.

**Effort estimate:** the OmniGibson robot-class + USD/CuRobo asset pipeline is
well-trodden (importer + `R1Pro` template) — call this the *bounded* part
(~1-2 days). The *unbounded* part is generalizing MoMaGen's bimanual core to
single-arm and validating motion planning + replay for the new kinematics.

---

## 1. How MoMaGen binds to a robot

A single trajectory is produced like this (see
`momagen/scripts/generate_dataset.py:144` → `momagen/datagen/data_generator.py:371`):

1. `env.reset()` randomizes object poses and repositions the robot.
2. For each **phase × arm × subtask**: slice the source `eef_pose` window for that
   subtask, apply the MimicGen object-centric rigid retarget
   (`momagen/utils/pose_utils.py:207`) using the *current* reference-object pose,
   and build a `WaypointTrajectory`.
3. A **reachability/visibility gate** decides whether to prepend a navigation
   phase (the mobile-manipulation part).
4. `WaypointTrajectory.execute` (`waypoint.py:1186`) runs: CuRobo **base** motion
   plan → CuRobo **arm** free-space motion plan up to `MP_end_step` →
   **open-loop replay** of the retargeted contact-rich suffix via
   `target_pose_to_action`.
5. Success = `env.is_success()["task"]` (BDDL goal predicates).
6. Per-attempt **720×720 third-person video** is written from `external_sensor2`
   inside `EnvOmniGibson.step` (`robomimic/.../env_omnigibson.py:382-391`).

Every step that says **"the robot"** is where embodiment assumptions live:
DOF layout, arm/base/torso joints, EEF link names, the action vector, the
controllers, and the kinematic model used by CuRobo.

---

## 2. TidyBot++ vs R1 — morphology

| | **R1 (MoMaGen native)** | **TidyBot++** |
|---|---|---|
| Arms | **2** (left + right), 7-DOF each | **1**, 7-DOF Kinova Gen3 |
| Base | Holonomic (3 planar DOF) | Holonomic (3 planar DOF) — *clean match* |
| Trunk/torso | Articulated trunk (4 DOF) | **None** |
| Head | Actuated (the `eyes` link drives visibility) | **None** (base + wrist cameras only) |
| Gripper | 2-finger ×2 | Robotiq 2F-85 (1 coupled DOF) |
| Arm control | per-arm Joint controller (MoMaGen solves a CVXPY QP IK) | native **EE-space** `arm_pos`+`arm_quat` via IK |
| Sim model | USD (Isaac) | **MuJoCo** + URDF (no USD) |

Native TidyBot++ action dict (`tidybot2/mujoco_env.py:29-32`):
`{ base_pose[3], arm_pos[3], arm_quat[4 (x,y,z,w)], gripper_pos[1] }`, EE expressed
in the **base/local** frame. Kinova retract pose
`[0.0, -0.349, 3.1416, -2.548, 0.0, -0.873, 1.571]`. Base joints `joint_x`(slide),
`joint_y`(slide), `joint_th`(hinge) — exactly OmniGibson's `HolonomicBaseRobot`
planar joints.

**Good news:** the holonomic base and EE-space arm control map *directly* onto
OmniGibson's `HolonomicBaseRobot` + `InverseKinematicsController`.
**Bad news:** no trunk, no head, one arm — three assumptions R1 bakes in.

---

## 3. The four hard-coded assumptions you must touch

Verified locations (R1/bimanual specifics):

1. **Bimanual `8×4` packing & `range(2)` loops** —
   `data_generator.py:211,317,451,554,721`; `waypoint.py` `merge_wp` stacks two
   `4×4` arms; `gripper_action` length 2.
2. **`arm_left`/`arm_right` / `gripper_left`/`gripper_right` keys** indexed
   directly — `waypoint.py:576-610,717-841`, env-interface `generate_action`.
3. **`isinstance(env.robot, R1)` + torso special-cases** that raise
   *"Robot type not supported"* — `data_generator.py:598,734`;
   `waypoint.py:550-557`; `momagen/utils/robot_config.py:25-41`. TidyBot++ has
   **no trunk**, so these must be made absent-trunk-safe.
4. **Visibility / head-tracking is R1-only** — gated to `isinstance(env.robot, R1)`
   (`waypoint.py`), base nav targets the R1 `eyes` link so the head tracks the
   object. TidyBot++ has no actuated head → set
   `hard_visibility_constraint=False` / `soft_visibility_constraint=False` (as
   Tiago does) — meaning MoMaGen's *visibility guarantee is lost*; base placement
   falls back to reachability + collision only.

---

## 4. Strategy decision: phantom arm vs N-arm generalization

| | **(a) Phantom second arm** | **(b) Proper N-arm generalization** |
|---|---|---|
| Idea | Keep the bimanual code path; define the OG `TidyBot` class with a real arm + a *dummy/frozen* arm, and set `object_ref["arm_right"]=None` in **every** subtask so the phantom arm just holds still | Parameterize arm count: replace `range(2)` / `arm_left|right` with a loop over `robot.arm_names`; make `merge_wp` handle 1..N arms; make eef-pose `N*4 × 4` |
| Touches | OG robot class, base_configs (always-`None` right arm), a few `None`-guards | `data_generator.py`, `waypoint.py`, both env-interface methods, `merge_trajs`, source `datagen_info` shape |
| Pros | Fast; reuses the *only* fully-supported `generate()` path (bimanual) | Clean; correct `8→4` data, no wasted DOF, reusable for other single-arm robots |
| Cons | Wastes a phantom arm; you must model a 2nd arm in USD or fake it; brittle | More upfront work; must bring the unimanual `generate()` path to parity |

> **Recommendation:** prototype with **(a)** to get end-to-end generation working
> and validate the asset/IK/replay stack, then refactor to **(b)** for a clean,
> reusable single-arm path. The non-bimanual `OmniGibsonInterface` /
> `from_json` path already exists (`env_interfaces/omnigibson.py:30`,
> `configs/task_spec.py`), but `DataGenerator.generate()` is only *fully* wired
> for `bimanual=True`, so (b) requires finishing the unimanual branch.

---

## 5. Step-by-step integration recipe

### A. OmniGibson robot class + USD asset

**A.1 — Convert URDF → USD** (the missing asset). Use the OmniGibson importer:

```bash
conda activate momagen
python BEHAVIOR-1K/OmniGibson/omnigibson/examples/robots/import_custom_robot.py
```

Author an import config (see the top of `import_custom_robot.py:56`) pointing at
`tidybot_platform/src/tidybot_description/urdf/tidybot_isaac.urdf`, with:

- `base_motion.use_holonomic_joints: true` — injects the six `base_footprint_*`
  virtual joints + world rootJoint that `HolonomicBaseRobot` asserts on
  (`import_custom_robot.py:56`, `964`).
- a `curobo:` block — the importer also emits
  `models/tidybot/curobo/tidybot_description_curobo_{default,base,arm,arm_no_torso}.yaml`
  (`create_curobo_cfgs`, `import_custom_robot.py:695`), required for motion
  planning. Validate the collision spheres and the Robotiq 4-bar coupling.

Output USD must land at `models/tidybot/usd/tidybot.usda` — the path
`robot_base.py:622` resolves from `model_name`.

**A.2 — Robot class.** Create `omnigibson/robots/tidybot.py`. Because TidyBot++ has
no trunk, subclass `HolonomicBaseRobot + ManipulationRobot` (drop
`ArticulatedTrunkRobot`). The `R1Pro` class (`robots/r1pro.py`) is the minimal
"override only the naming patterns" template — mirror it with single-arm
(`arm_names = ['0']`) Kinova/Robotiq names:

```python
class TidyBot(HolonomicBaseRobot, ManipulationRobot):
    @property
    def model_name(self): return "tidybot"

    @cached_property
    def arm_names(self): return ["0"]                     # single arm

    @cached_property
    def arm_joint_names(self):  return {"0": [f"joint_{i}" for i in range(1, 8)]}   # Gen3
    @cached_property
    def eef_link_names(self):   return {"0": "bracelet_with_vision_link"}
    @cached_property
    def finger_joint_names(self): return {"0": ["left_driver_joint", "right_driver_joint"]}  # 2F-85

    @cached_property
    def base_footprint_link_name(self): return "base_link"
    @cached_property
    def floor_touching_base_link_names(self): return ["caster_link_0", ...]

    @property
    def _default_controllers(self):
        return {"base": "HolonomicBaseJointController",
                "arm_0": "InverseKinematicsController",     # matches EE-space control
                "gripper_0": "MultiFingerGripperController"}

    @property
    def _default_joint_pos(self):
        # base at origin, Gen3 retract, gripper open
        ...
```

Register it by adding the import to `omnigibson/robots/__init__.py` (auto-registers
into `REGISTERED_ROBOTS`). Verify it loads standalone before touching MoMaGen.

> **Controller choice matters for replay.** MoMaGen's open-loop replay turns
> planned configs into actions via `q_to_action` (`holonomic_base_robot.py:374`).
> If you use an `InverseKinematicsController` for the arm, the MoMaGen env-interface
> can use the simpler single-arm IK path (§B) instead of R1's CVXPY QP.

### B. MoMaGen env-interface

The abstract contract (`momagen/env_interfaces/base.py:84-167`):
`get_robot_eef_pose`, `target_pose_to_action`, `action_to_target_pose`,
`action_to_gripper_action`, `get_object_poses`, `get_subtask_term_signals`,
`get_datagen_info`. A metaclass auto-registers any subclass that sets
`INTERFACE_TYPE`, once imported in `env_interfaces/__init__.py`.

Subclass the **single-arm** base `OmniGibsonInterface` (`omnigibson.py:30`), which
already returns a `4×4` eef pose for `default_arm` and resolves the action-vector
arm slice via `_setup_arm_controller`:

```python
class OmniGibsonInterfaceTidyBot(OmniGibsonInterface):
    INTERFACE_TYPE = "omnigibson_tidybot"

    def target_pose_to_action(self, target_pose, relative=True):
        # arm is an IK controller → use the inherited single-arm preprocess
        # (NOT R1's per-arm CVXPY QP from eef_jacobian_relative)
        ...

    def action_to_gripper_action(self, action):
        return action[self._gripper_slice]      # single gripper, not [2]

    def get_datagen_info(self, action=None):
        # record base_pose if you want navigation phases (R1 does, omnigibson.py:291)
        ...
```

Then per-task interface classes (mirror `MG_R1PickCup`) build a `TaskConfig`
(`omnigibson.py:20`) listing the tracked objects and termination signals for your
task, and add a `'tidybot'` key to `robot_specific_objects` (keyed by
`type(robot).__name__.lower()`, `omnigibson.py:104`).

**Remove R1-hardcoded indices** when adapting: the literal `action[11]=0` and the
`action[5:12]`/`action[12:19]` arm slices in the bimanual `generate_action`
(`omnigibson.py:468`), and the cycle-consistency assert ranges.

### C. Robot config & task config

`momagen/utils/robot_config.py`:

- add `ROBOT_TIDYBOT = "TidyBot"`, import the OG class, add a `ROBOT_LINK_NAMES`
  entry (handle the **absent torso**);
- add `isinstance` branches to `get_torso_link_name` / `get_robot_type_from_instance`
  (`robot_config.py:25,34`) — they currently `raise ValueError` for unknown robots
  (and torso should return `None`/skip for tidybot);
- add `get_tidybot_config()` + `configure_tidybot_env_meta(env_meta)` mirroring
  `get_tiago_config` / `configure_tiago_env_meta` (`robot_config.py:43,112`):
  a `reset_joint_pos` tensor + a `controller_config` (arm IK, gripper MultiFinger,
  base HolonomicBaseJointController; **no** `arm_left`/`arm_right`/`trunk`).
  *(R1 has no `configure_r1_env_meta` — it reuses the source dataset's stored
  env_meta; tidybot++ needs an explicit configure fn.)*

Wire `configure_tidybot_env_meta` into `generate_dataset.py:215` (currently only
`if robot_type == "Tiago"`; add an `elif`) and extend the `--robot_type` choices
(`generate_dataset.py:884`).

Config classes & JSON:

- register an `MG_Config` subclass in `momagen/configs/omnigibson.py` (like
  `R1PickCup`, an empty `phase1=dict()` placeholder);
- author the real spec in `momagen/datasets/base_configs/tidybot_<task>.json`
  (phase → arm → subtask; `r1_clean_pan.json` is the concrete reference). For
  **strategy (a)** set the `arm_right` subtask's `object_ref` to `null` everywhere;
- add the base-config path + task name to `BASE_CONFIGS` /
  `TASK_NAMES_MOMAGEN_ONLY` in `momagen/scripts/generate_configs.py` (the order is
  asserted, `generate_configs.py:109`) and run
  `python momagen/scripts/generate_configs.py`.

### D. Source demo + annotation

1. **Collect one teleop demo** on TidyBot++ (the bimanual path hardcodes
   `selected_src_demo_ind=0`, so one annotated source demo is enough). TidyBot++
   ships a phone-teleop interface and `tidybot2/convert_to_robomimic_hdf5.py` to
   produce a robomimic HDF5. *(If you collect in OmniGibson instead, reuse
   BEHAVIOR-1K's teleop — keyboard/spacemouse/VR via telemoma, or JoyLo.)*
2. The source HDF5 must expose, per timestep:
   `data/<demo>/datagen_info/{eef_pose, object_poses/<name>, gripper_action}` and
   `actions`. **Match the eef-pose convention** to your interface's
   `get_robot_eef_pose` (single `4×4`, *not* R1's `8×4`).
3. **Reconcile frames:** TidyBot++ expresses `arm_pos`/`arm_quat` in the base/local
   frame with `(x,y,z,w)` quats; OmniGibson/MoMaGen work in **world** frame with
   OG's quat convention. Convert in `get_datagen_info` / `target_pose_to_action`.
4. **Annotate** subtask boundaries by replaying:
   ```bash
   python momagen/scripts/prepare_src_dataset.py \
     --dataset momagen/datasets/source_og/tidybot_<task>.hdf5 \
     --env_interface MG_TidyBot<Task> --env_interface_type omnigibson_tidybot \
     --replay_for_annotation
   ```
   Note the `MP_end_step` (MP→replay split) and `subtask_term_step` (subtask end),
   then generate the processed HDF5:
   ```bash
   python momagen/scripts/prepare_src_dataset.py \
     --dataset momagen/datasets/source_og/tidybot_<task>.hdf5 \
     --env_interface MG_TidyBot<Task> --env_interface_type omnigibson_tidybot \
     --generate_processed_hdf5
   ```
   This writes the `env_interface_name`/`env_interface_type` attrs
   (`prepare_src_dataset.py:247`) that `generate_dataset.py:327` reads back.

---

## 6. Generating & collecting trajectories on TidyBot++

Once A-D are in place, generation uses the **same** entry point as R1:

```bash
export OMNIGIBSON_HEADLESS=1
TASK=<your_task>; DR=0; NUM_DEMOS=10; WORKER_ID=0; FOLDER=/path/to/out
python momagen/scripts/generate_dataset.py \
  --config momagen/datasets/configs/demo_src_tidybot_${TASK}_task_D${DR}.json \
  --num_demos $NUM_DEMOS --robot_type TidyBot \
  --folder $FOLDER/$TASK/tidybot_${TASK}_worker_$WORKER_ID --seed $WORKER_ID \
  --auto-remove-exp \
  ${BIMANUAL_FLAG}   # --bimanual ONLY for strategy (a) phantom-arm; omit for (b)
```

Outputs (in `<folder>/<experiment.name>/`): `demo.hdf5` (merged successes),
`demo_failed.hdf5`, `videos/<NNNN>.mp4` (per-attempt 720×720 third-person), and
`important_stats.json`. Set `experiment.generation.guarantee=true` in the config to
run until N **successes** (default is N **attempts**).

**Collecting more / scaling:** run multiple workers with different `--seed`
(`WORKER_ID`) into per-worker folders, then merge their `demo.hdf5` with
`MG_FileUtils.merge_all_hdf5` (or robomimic's dataset-merge utilities).

---

## 7. Validation checklist (do these in order)

1. ✅ `TidyBot` USD loads standalone in OmniGibson; joints/links named as expected.
2. ✅ CuRobo warmup succeeds with the generated `tidybot_*_curobo_*.yaml`
   (IK + TrajOpt, no collision-sphere errors).
3. ✅ The env-interface round-trips: `target_pose_to_action(get_robot_eef_pose())`
   keeps the arm still (cycle-consistency assert passes).
4. ✅ `prepare_src_dataset.py --replay_for_annotation` reproduces your teleop demo
   in sim (frames match) — confirms the frame/convention reconciliation.
5. ✅ A single generation attempt completes: base MP → arm MP → replay, with a
   readable `videos/0000.mp4`.
6. ✅ `env.is_success()["task"]` fires on a good attempt (BDDL goal satisfied).

---

## 8. File-by-file change summary

| File | Change |
|------|--------|
| `omnigibson/robots/tidybot.py` *(new)* | `TidyBot(HolonomicBaseRobot, ManipulationRobot)` class |
| `omnigibson/robots/__init__.py` | import → auto-register |
| `models/tidybot/usd/…`, `models/tidybot/curobo/…` *(new assets)* | from `import_custom_robot.py` |
| `momagen/env_interfaces/omnigibson.py` | `OmniGibsonInterfaceTidyBot` + per-task `MG_TidyBot*` + `TaskConfig`s |
| `momagen/env_interfaces/__init__.py` | import the new interface |
| `momagen/utils/robot_config.py` | `ROBOT_TIDYBOT`, link map, `get_/configure_tidybot_*`, isinstance branches |
| `momagen/configs/omnigibson.py` | `MG_Config` task subclasses |
| `momagen/datasets/base_configs/tidybot_<task>.json` *(new)* | phase/arm/subtask spec |
| `momagen/scripts/generate_configs.py` | register base config + task name |
| `momagen/scripts/generate_dataset.py` | `--robot_type TidyBot` branch + `configure_tidybot_env_meta` |
| `momagen/datagen/data_generator.py`, `momagen/datagen/waypoint.py` | **strategy (b) only:** N-arm generalization; **both:** absent-trunk + visibility-off guards |
| `momagen/datasets/source_og/tidybot_<task>.hdf5` *(new)* | teleop source demo |

---

*Cited line numbers reflect the repository state at the time of writing; grep the
named symbols if they have drifted.*
