# TidyBot Task Pipelines: Collect, Generate, Replay

This tutorial documents the complete TidyBot++ pipeline built on this fork: collecting
source trajectories with the browser-forwarded GUI, the custom task definitions
(**pick-and-dispose**, **make-coffee**, **pack-and-deliver**), generating expert
trajectories from a single source demo, and replaying the collected data.

Everything here was validated end-to-end on the `datagen_picking_up_trash`
(pick-and-dispose) task in the `house_single_floor` scene, producing 106 expert
trajectories (60 normal-speed + 46 fast).

---

## 1. Collecting source trajectories with the forwarded GUI

MoMaGen needs **one** human (or scripted) source demonstration per task. On a headless
GPU server, use the web teleop — it streams an interactive MJPEG view to your browser
and takes keyboard commands over HTTP.

### 1.1 Launch the web teleop on the server

```bash
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
python momagen/scripts/collect_source_web_teleop.py
```

The script:

- builds the task environment from the sampled scene instance (see §2),
- spawns TidyBot at a validated, collision-free pose **at load time**
  (`cfg["robots"][0]["position"]` — never teleport the holonomic base at runtime with
  a placement search; see §5 gotchas),
- serves an MJPEG stream + control endpoint on **port 8890**.

### 1.2 Forward the GUI to your browser

```bash
ssh -L 8890:localhost:8890 <user>@<server>
# then open http://localhost:8890 locally
```

Controls: WASD/QE drive the holonomic base; IJKL/UO jog the end-effector via
damped-least-squares IK (`CartesianTeleop`); space toggles the Hand-E gripper;
`1/2` orbit, `3/4` tilt, `5/6` zoom the camera. The wrapper records every step through
`DataCollectionWrapper` into `momagen/datasets/source_og/tidybot_<task>.hdf5` and tags
the episode with the `mask/use` filter key.

### 1.3 Scripted collection (alternative)

For tasks with simple structure, a scripted source is more repeatable than teleop and
produces cleaner replay segments. `momagen/scripts/collect_source_scripted_trash.py` is
the script that produced the shipped pick-and-dispose source demo: trav-map navigation,
a tilted near-top-down DLS-IK grasp, and a lateral reach-over drop. It also prints the
**phase boundaries** (`MP_end_step` / `subtask_term_step`) you need for the base config
(§3.2), so no manual annotation replay is required.

> **Rule of thumb learned the hard way:** whatever collects the source, keep the
> contact-rich segments *monotone* (no closed-loop correction wiggles) — MoMaGen replays
> them open-loop re-anchored to new object poses, and baked-in corrections do not
> transfer. And make sure objects are at their final poses **before** `env.reset()` is
> recorded: a post-reset nudge leaves a stale object pose in `datagen_info` that shifts
> every generated grasp by the nudge distance.

---

## 2. The custom tasks

All three tasks live in `house_single_floor` and follow the same recipe:
**BDDL activity → sampled scene instance → env interface → base config**.
Success is always the BDDL `:goal` predicate, evaluated by `env.is_success()["task"]`.

### 2.1 Pick-and-dispose (`datagen_picking_up_trash`) — implemented ✅

Pick a soda can off the kitchen countertop, drive to the trash can on the floor, drop
it in.

**BDDL** (`BEHAVIOR-1K/bddl/bddl/activity_definitions/datagen_picking_up_trash/problem0.bddl`):

```lisp
(:objects  ashcan.n.01_1 - ashcan.n.01
           can__of__soda.n.01_1 - can__of__soda.n.01
           countertop.n.01_1 - countertop.n.01 ...)
(:init     (ontop ashcan.n.01_1 floor.n.01_1)
           (ontop can__of__soda.n.01_1 countertop.n.01_1)
           (inroom countertop.n.01_1 kitchen) ...)
(:goal     (and (inside ?can__of__soda.n.01_1 ?ashcan.n.01_1)))
```

Key implementation facts:

- The kitchen has no `table.n.02` — the can sits on `countertop.n.01`.
- `can_of_soda_595` is **scaled `[0.5, 0.5, 0.38]`** in the scene instance (~32 mm
  diameter): the Hand-E stroke is 50 mm, so full-size cans are ungraspable, and a squat
  can resists tipping under the near-top-down grasp. (22 mm slips; 32 mm is validated.)
- Scene instance JSON (spawns + scaling):
  `momagen/scene_instances/house_single_floor/house_single_floor_task_datagen_picking_up_trash_0_0_template.json`
  (mirror it into `BEHAVIOR-1K/datasets/2025-challenge-task-instances/scenes/...`).
- Env interface: `MG_TidyBotPickingUpTrash` + `TASK_CONFIGS["tidybot_picking_up_trash"]`
  in `momagen/env_interfaces/omnigibson.py`, tracking `can_of_soda_595` and
  `trash_can_596`.
- Two phases: grasp (object_ref = can) and drop (object_ref = trash, attached_obj =
  can). The drop is out of arm reach from the grasp standoff, so MoMaGen auto-inserts a
  **navigation phase** carrying the can.

### 2.2 Make-coffee (`datagen_make_coffee`) — designed, not yet implemented

Pour milk, then coffee, into a cup. Liquids are represented as **BDDL cube tokens**
(real dataset objects, so the goal predicate sees them — `PrimitiveObject`s would not
be visible to `is_success()`):

```lisp
(:objects coffee_cup, milk vessel (open mug), coffee vessel (open mug),
          sugar_cube (= milk token), bouillon_cube (= coffee token), countertop)
(:init    tokens inside their vessels; vessels on the countertop)
(:goal    (and (inside sugar_cube coffee_cup) (inside bouillon_cube coffee_cup)))
```

~4 phases: grasp milk vessel → pour over cup (`object_ref=coffee_cup`,
`attached_obj=milk_vessel`; the tilt lives in the contact-rich replay segment, carried
through by 6-DOF object-centric re-anchoring) → grasp coffee vessel → pour. Highest
risk: token must land in the cup; use a wide cup and verify `inside` fires in the
source demo before generating.

### 2.3 Pack-and-deliver (`datagen_pack_deliver`) — designed, not yet implemented

Put 2–3 small objects into a basket, then carry the basket to a surface in another room:

```lisp
(:goal (and (inside obj1 basket) (inside obj2 basket) (ontop basket delivery_surface)))
```

Phases: per-object grasp → place-in-basket, then grasp basket → deliver
(`object_ref=delivery_surface`, `attached_obj=basket`). The delivery surface being out
of reach triggers the auto-navigation phase while carrying. Register the delivery
surface in `tracked_objects` or the reachability gate never fires. TidyBot has no head
camera, so navigation is gated on IK reachability only.

---

## 3. Generating expert trajectories from a source demo

### 3.1 Process the source demo

```bash
python momagen/scripts/prepare_src_dataset.py \
  --dataset momagen/datasets/source_og/tidybot_picking_up_trash.hdf5 \
  --env_interface MG_TidyBotPickingUpTrash \
  --env_interface_type omnigibson_tidybot \
  --generate_processed_hdf5
```

This bakes `datagen_info` (eef/base/object poses, gripper actions) into
`momagen/datasets/processed_source_demos/tidybot_picking_up_trash.hdf5`.
Isaac may segfault at shutdown *after* a successful write — check that the file has
`datagen_info` **and** `data.attrs["env_args"]` (re-add `env_args` from the `config`
attr if the segfault landed between the two writes).

### 3.2 Base config with phase boundaries

`momagen/datasets/base_configs/tidybot_picking_up_trash.json` — fill each phase's
`object_ref`, `attached_obj`, `MP_end_step`, `subtask_term_step` with the boundaries
printed by the collection script (single-arm TidyBot uses the bimanual format with a
phantom `arm_right`: `object_ref: null`). Register the config path and the task name in
`momagen/scripts/generate_configs.py` (`BASE_CONFIGS`, `TIDYBOT_TASK_NAMES`) and add an
`MG_Config` subclass in `momagen/configs/omnigibson.py`, then:

```bash
python momagen/scripts/generate_configs.py
```

> Generated configs bake **absolute paths**. Re-run `generate_configs.py` whenever the
> repo moves; a stale path exits silently with rc=0.

### 3.3 Run generation with the validated environment stack

```bash
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
# validated TidyBot generation stack:
export MOMAGEN_REPLAY_NUM_REPEAT=1          # QP replay steps per waypoint
export JC_NAV_DIST_THRESH=0.05 JC_NAV_ANGLE_THRESH=0.10   # nav endpoint tolerance (arm compensates ±3cm)
export JC_REANCHOR_DROP=1                   # re-aim release if the bin was nudged during nav
export JC_BASE_SAMPLE_LO=0.55 JC_BASE_SAMPLE_HI=0.65 \
       JC_BASE_SAMPLE_YAWC=-1.56 JC_BASE_SAMPLE_YAWS=0.6  # drop-approach sampling band (source geometry)
export JC_BASE_COUPLED=1 JC_OBJ_MAG=0.05    # diversity: can ±5cm, base follows per episode
export JC_MP_INTER_DIST=0.025 JC_DS_RATIO=2 # 2.1x faster execution (validated, precision intact)

python momagen/scripts/generate_dataset.py \
  --config momagen/datasets/configs/demo_src_tidybot_picking_up_trash_task_D0.json \
  --num_demos 20 --bimanual --robot_type TidyBot \
  --folder <output_dir> --seed 9300 --auto-remove-exp
```

What the knobs are for (each fixed a measured failure mode):

| Variable | Why |
|---|---|
| `JC_BASE_SAMPLE_*` | stock sampling draws the drop standoff 0.4–1.0 m at any angle; the tilt-constrained executor only tracks the release from ~0.6 m on the source's approach side. Biggest single win (→ 83% on the fixed-base config). |
| `JC_REANCHOR_DROP` | the release trajectory is anchored to the bin pose at phase start; the base can nudge the bin en route. This shifts the replay targets by the measured displacement. |
| `JC_NAV_*_THRESH` | nav's default `low_precision` accepts 10 cm/11° base error; TidyBot's arm can only absorb ±3 cm. |
| `JC_BASE_COUPLED`/`JC_OBJ_MAG` | per-episode diversity. The object randomizes ±`JC_OBJ_MAG`; the base is teleported (direct x/y/yaw joint write + zeroed velocities + controller reset — statically clean) to the source standoff relative to the object, so grasp geometry is preserved. |
| `JC_MP_INTER_DIST`/`JC_DS_RATIO` | speed. Note the CLI `--ds_ratio` flag is dead code — use `JC_DS_RATIO`. 1031 → ~490 steps per episode, drop precision unchanged. |

**Multi-GPU:** one worker per GPU via `CUDA_VISIBLE_DEVICES=<gpu>` **alone** (do not
combine with `OMNIGIBSON_GPU_ID` — that breaks Omniverse's Vulkan device enumeration on
non-zero GPUs). Give each worker its own `--seed` and `--folder`, then merge or split
(§4).

Successful episodes are written per-attempt under `<output>/.../tmp/*.hdf5` and merged
into `demo.hdf5` at run end — a killed run loses nothing; merge manually with
`momagen.utils.file_utils.merge_all_hdf5`.

**Always verify a rendered video, not just the logs.** BDDL happily labels a lucky
1-meter can drop as a success; watching one episode per config change caught every bug
the metrics missed.

### 3.4 What generation does per attempt

Randomize object poses (D0: xy ± mag, z-rot) → couple the base → CuRobo-plan the
free-space reach to the transformed `MP_end` pose → replay the contact-rich grasp
segment re-anchored to the new object pose (QP eef tracking) → verify the object is
attached → auto-navigate to the drop → replay the re-anchored release → check the BDDL
goal → keep or discard.

---

## 4. Using the collected data

Each demo is a self-contained hdf5 (one per file in the delivered
`demos_60_split/`, `demos_fast_split/` layout; `data/demo_0` inside):

| Field | Contents |
|---|---|
| `data.attrs["env_args"]` | full env construction config (scene, robot, task) |
| `actions` | (T, 11) @ 20 Hz: base velocity (x, y, yaw) + 7 arm joint targets + gripper |
| `states` | serialized per-step sim states (`og.sim.load_state(..., serialized=True)`) |
| `obs` | proprio + **wrist cam** (`arm_camera_link`) + **base cam** (`base_camera_link`): 256×256 rgb / depth / seg + camera poses |
| `datagen_info` | eef/base/object poses per step (for re-generation) |

### Replay (simple demonstration)

```bash
export OMNIGIBSON_HEADLESS=1
python momagen/scripts/replay_demo.py demos_60_split/tidybot_picking_up_trash_demo_000.hdf5 \
       --video replay.mp4
# or exact state-sync playback instead of action playback:
python momagen/scripts/replay_demo.py <demo.hdf5> --states
```

The script builds the env from the demo's own `env_args`, restores the recorded first
sim state, then steps the recorded actions (or loads every recorded state with
`--states`), optionally rendering an overview video.

For a fully portable setup (code + robot model + scene + all assets + decryption key +
this replay script in one archive), see the self-contained bundle produced by the
pipeline (`tidybot_trash_replay_bundle.tar.gz`): extract, `pip install -e` the four
packages, and replay — no other downloads needed.

---

## 5. Gotchas index (hard-won)

1. **Never runtime-teleport the holonomic base with a placement search** — single
   teleports with velocity-zeroing + controller reset are clean (probe-verified), but
   rapid-fire search loops corrupt the articulation (chassis sinks, base oscillates).
   Prefer load-time placement; use the `JC_BASE_COUPLED` code path for per-episode moves.
2. **`self_collisions=False` for TidyBot** (overlapping imported collision meshes
   otherwise vibrate the robot at rest). Baked into `configure_tidybot_env_meta`.
3. **Phantom-arm gripper aliasing**: TidyBot's single gripper backs *both* action
   channels; any code writing the phantom arm's gripper command clobbers the real one.
   Guarded (with `object_ref` checks) in the MP builder and all replay loops of
   `momagen/datagen/waypoint.py` — mind this when touching those loops.
4. **`waypoint.py` has twin execute methods** — `execute_baseline` (mimicgen/skillgen)
   and `execute` (MoMaGen). Instrument/patch the right one.
5. Source objects must be settled **before** `env.reset()` records the initial state
   (stale `object_poses` shift every generated grasp).
6. Debug env vars: `JC_REPLAY_DEBUG=1` (per-waypoint finger-vs-object trace),
   `JC_DROP_DEBUG=1` (end-of-episode object-vs-goal geometry per trial).
