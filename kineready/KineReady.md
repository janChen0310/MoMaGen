# KineReady

**A 59k-parameter surrogate for inverse kinematics, so base-pose quality can be a dense online
reward instead of a 0.34-second blocking solve.**

| | |
|---|---|
| Package | `MoMaGen/kineready` |
| Robot | TidyBot++ · Kinova Gen3, 7-DOF arm on a 3-DOF holonomic base |
| Teacher | CuRobo via OmniGibson |
| Status | trained and verified |

| Result | Value |
|---|---|
| AUROC, held-out IID | **0.9997** |
| False positives at 95% recall | **0.14 %** |
| Top-1 ranked pose truly reachable | **100 %** |
| Latency per base pose (GPU) | **3.6 µs** |

> *Measured* — 1.1 M-row dataset · 81,391-row IID holdout · 19 discriminative ranking configs ·
> one RTX 4090.

> **Viewing this file.** Math is KaTeX, rendered natively by the VS Code Markdown preview
> (`Ctrl/Cmd+Shift+V`). It requires `markdown.math.enabled`, which is on by default since
> VS Code 1.55. Diagrams are box-drawing inside fenced blocks, so they need no extension.

---

## Contents

- [1. One term dominates the cost](#1-one-term-dominates-the-cost)
- [2. Architecture](#2-architecture)
- [3. Training pipeline and recipe](#3-training-pipeline-and-recipe)
- [4. Inference pipeline](#4-inference-pipeline)
- [5. Evaluation results](#5-evaluation-results)
- [6. Failure modes and why the obvious checks missed them](#6-failure-modes-and-why-the-obvious-checks-missed-them)
- [7. Reproduction](#7-reproduction)

---

## 1. One term dominates the cost

`base_pose_metric` scores a candidate base pose on four components. Three are effectively free.
The fourth is why none of it could be used inside a training loop.

| Component | Cost | How it is computed |
|---|---:|---|
| Distance to target | ~0 ms | Closed form |
| Visibility | 0.26 ms | Frustum test on AABB corners + occlusion raycast |
| Collision (static) | 0.50 ms | Batched sphere check, all poses at once |
| **IK feasibility** | **340 ms** | CuRobo solve, 512 seeds, **one pose at a time** |

Per candidate base pose. Exact IK costs roughly 440 ms when it *fails*, which is the common case
during a sweep.

A navigation policy choosing where to stop needs to score many candidate endpoints per step, and
to do so without a live simulator in the rollout worker. At 0.34 s per pose with CuRobo and Isaac
Sim resident in memory, that is not a reward function — it is a batch job.

KineReady replaces **only** the IK term with a learned model and leaves the other three exact.
Environment effects stay where they are already cheap and correct.

---

## 2. Architecture

### 2.1 The factorization

The model is small because the input is already in the right coordinates. One change of frame
turns a mobile-base placement problem into a fixed-arm reachability query:

$$
T_E^B \;=\; \bigl(T_B^W(b)\bigr)^{-1} \; T_O^W \; T_E^O
$$

where $b=(x,y,\theta)$ is the candidate base pose, $T_O^W$ the object's world pose, and
$T_E^O$ the grasp in the object's frame.

```
 WORLD FRAME — what the planner has          BASE FRAME — what the model sees
 ─────────────────────────────────           ────────────────────────────────

                ● grasp E                                ● target
                ▲                                       ╱
                │ T_E^O                                ╱
             ┌──┴──┐                                  ╱   one query
     T_O^W   │ obj │                                 ╱
   ┌────────►└─────┘                                ╱
   │                                          ┌────┴────┐
   │  T_B^W(b)  ┌────────┐                    │  base   │
   ├───────────►│  base  │                    └─────────┘
   │            └────────┘                         │
   ●─────────────────────── W               ●──────┴──────────── B
 world origin    b = (x, y, θ)            base frame origin

 three transforms, world coordinates       9 numbers:  [ x y z | r₁…r₆ ]
 depends on b AND the world layout         a fixed property of the arm
```

Where the base stands is absorbed into the transform, not fed to the network. What remains is a
fixed property of the arm — which is why a small MLP suffices, and why the model transfers to any
scene without retraining.

This is the load-bearing piece of the whole approach. Get it wrong and the model learns a coherent
function of the *wrong quantity*, which fits its own labels perfectly and only fails once the base
moves. It is therefore tested two ways: against an independent matrix inverse, and against the
invariance property it exists to provide —

$$
T_E^B\bigl(b,\;T\bigr) \;=\; T_E^B\bigl(G \cdot b,\;\; G\,T\bigr)
\qquad \forall\, G \in \mathrm{SE}(2)
$$

Move the base and the target together by the same rigid transform and the query must not change.
A transform that ignored yaw entirely would pass that test alone, so the counterpart is checked
too: rotating the base *alone* must change the query.

The base inverse is computed in closed form rather than with `np.linalg.inv`. This is not premature
optimization — it runs $N\times K$ times per policy step, and a rigid transform's inverse is exact
in closed form where the general inverse is not:

$$
\bigl(T_B^W\bigr)^{-1} =
\begin{bmatrix} R_z(\theta)^{\!\top} & -R_z(\theta)^{\!\top} t \\[2pt] \mathbf{0}^{\!\top} & 1 \end{bmatrix}
$$

### 2.2 Input encoding

Nine numbers — position, plus the first two columns of the rotation matrix (Zhou et al.'s
continuous 6D representation):

$$
\mathbf{x} \;=\;
\bigl[\, t_x,\; t_y,\; t_z,\;\; r_{11}, r_{21}, r_{31},\;\; r_{12}, r_{22}, r_{32} \,\bigr]
\;\in\; \mathbb{R}^{9}
$$

Quaternions and Euler angles are discontinuous as functions on $SO(3)$, so a network consuming
them must represent a jump that is not there. The 6D form has no such seam, and $q$ and $-q$ —
the same rotation — encode identically. The third column is recovered when needed by Gram–Schmidt,
$\mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2$, which also re-orthonormalizes noisy input.

Per the proposal's MVP, the arm's current configuration $q_0$ is dropped from the input: TidyBot
navigates with the arm at its home pose and has no torso, so $q_0$ is a constant.

### 2.3 Network

| Stage | Shape | Params | Notes |
|---|---|---:|---|
| Input block | 9 → 128 → 128 | 17,792 | SiLU |
| Residual block | 128 → 128 → 128 | 33,024 | Added to input, then SiLU |
| Trunk | 128 → 64 | 8,256 | Shared by both heads |
| Head — `exist` | 64 → 1 | 65 | $P(\exists\, q \text{ solving the IK})$ |
| Head — `robust` | 64 → 1 | 65 | Fraction surviving pose jitter |
| **Per member** | — | **59,202** | Ensemble of 5 → 296,010 total |

**Two heads, because a bit is not enough.** A target deep in the workspace and one on its very
edge both have $y_{\text{ex}} = 1$. Only the margin distinguishes them, and choosing between those
two base poses is the entire job. The robust label is

$$
y_{\mathrm{rob}}(T) \;=\; \frac{1}{J}\sum_{j=1}^{J}
y_{\mathrm{ex}}\bigl(T \oplus \xi_j\bigr),
\qquad
\xi_j \sim \mathcal{N}\!\bigl(0,\sigma_{\mathrm{pos}}^{2}\bigr) \times
        \mathcal{N}\!\bigl(0,\sigma_{\mathrm{rot}}^{2}\bigr)
$$

with $\sigma_{\mathrm{pos}} = 2\,\mathrm{cm}$, $\sigma_{\mathrm{rot}} = 5^\circ$, $J = 8$ — taken from
the pipeline's real object jitter and base coupling noise, not invented. Rotation perturbations
are applied through the exponential map (Rodrigues), so every perturbed pose stays a valid
rotation.

**Five members, because the errors are asymmetric.** A false positive sends the policy to a base
pose the planner then fails on, wasting a real episode; a false negative discards one candidate out
of many. Consumers therefore get a lower confidence bound:

$$
P^{\mathrm{LCB}}_{\mathrm{kin}} \;=\; \mathrm{clip}\bigl(\mu - \beta\,\sigma,\; 0,\; 1\bigr),
\qquad
\mu = \tfrac{1}{M}\textstyle\sum_m p_m, \quad
\sigma = \operatorname{std}_m\, p_m
$$

This moves toward zero exactly where the members disagree, instead of averaging uncertainty away
into a confident-looking number. A temperature $T^\star$ fitted on validation NLL keeps the
probabilities honest enough to threshold:

$$
T^\star \;=\; \arg\min_{T}\; \mathrm{BCE}\!\left(\frac{\bar{z}}{T},\; y\right)
\qquad\Rightarrow\qquad T^\star = 0.851
$$

### 2.4 Aggregating over grasp candidates

A grasp on a cylindrical object is not one pose. Rotating the gripper about the can's vertical
axis gives a different end-effector pose that grasps it equally well, and a base pose that cannot
reach the demo's exact grasp may comfortably reach the one 45° around. The $K$ candidates combine
by noisy-OR:

$$
P_{\mathrm{kin}}(b) \;=\; 1 - \prod_{k=1}^{K}\bigl(1 - \pi_k\, p_k\bigr)
$$

Preferred over $\max_k p_k$ because $K$ near-misses genuinely are better evidence than one
near-miss, and the result stays smooth for reward shaping. The priors $\pi_k$ weight the
demonstrated grasp above its inferred siblings — the symmetry is only approximately true once a
can has a tab or a label ($\pi_{\text{demo}} = 1.0$, $\pi_{\text{other}} = 0.85$).

### 2.5 Module map

| File | Role | Simulator? |
|---|---|---|
| `frames.py` | Base-frame transform, 6D rotation encoding, noisy-OR, perturbation | no |
| `robot_env.py` | Minimal empty-scene TidyBot environment for labeling | yes |
| `teacher.py` | Batched exact CuRobo IK labeler + control suite | yes |
| `datagen.py` | Samplers, boundary selection, resumable shards | yes |
| `dataset.py` | Shard reader, geometric splits, duplicate guard | no |
| `model.py` | MLP, ensemble, calibration, ECE / FPR@recall | no |
| `train.py` | Training loop and metrics | no |
| **`reward.py`** | **`ReadinessReward` — the API a policy calls** | **no** |
| `eval_offline.py` | Metrics against four baselines | no |
| `eval_ranking.py` | Ranking against the exact oracle | yes |
| `examples/` | Stage runners, latency, worked example, verification scripts | mixed |

Everything the deployed reward path touches is simulator-free. Isaac Sim is needed to *produce*
labels, never to *consume* the model.

### 2.6 Scope boundary

The model is **object- and task-agnostic**. It has never seen a soda can, a kitchen, or a scene.
Its input is an arbitrary end-effector target in the base frame; the upstream planner supplies
whatever target it wants at query time.

Task-specific logic — the trash can's grasp pose, the $K=8$ symmetry orbit — lives in `examples/`,
above both the model and the reward core.

The labels are also **scene-free**: joint limits and self-collision are enforced, world geometry
is not. Environment effects stay with the exact terms.

---

## 3. Training pipeline and recipe

A silently-wrong labeler poisons the dataset and every number derived from it, and unlike a
rendering bug it is invisible in every artifact. So the pipeline spends its first stage proving the
teacher is correct, and refuses to proceed otherwise.

```
 ┌──────────── SIMULATOR REQUIRED ────────────┐  ┌───────── RUNS ANYWHERE ─────────┐

   Stage 0                  Stage 1               Stage 2              Stage 3
 ┌──────────────────┐    ┌──────────────────┐   ┌────────────────┐  ┌────────────────────┐
 │ EMPTY-SCENE      │    │ SHARDS (hdf5)    │   │ ENSEMBLE ×5    │  │ ReadinessReward    │
 │ TEACHER          │───►│ 1.1 M rows       │──►│ 59,202 par. ea.│─►│ numpy + torch only │
 │ CuRobo, 512 seeds│    │ 90 MB            │   │ 225 s to train │  │ 3.6 µs / pose      │
 │ ARM_NO_TORSO     │    │ fk / uniform /   │   │ T* = 0.851     │  │ → navigation policy│
 │ 398 labels/s     │    │ robust / screen  │   │                │  │                    │
 └──────────────────┘    └──────────────────┘   └────────────────┘  └────────────────────┘
         │                        │
         └── controls must pass, or the run aborts ──┘
```

The stage boundary is also a dependency boundary: nothing to the right of it imports OmniGibson,
CuRobo, or Isaac.

### Stage 0 — the teacher, and proving it works

Labels come from CuRobo through OmniGibson's `CuRoboMotionGenerator`, called with `ik_only=True`.
Three properties of that wrapper make it cheap and scene-free:

- Targets are passed in the robot's own base frame, so a candidate base pose becomes a transformed
  target and **the robot never moves** — no teleporting, no state save/restore.
- `compute_trajectories` chunks `num_targets` internally, so one call labels an arbitrary number
  of poses.
- `ik_world_collision_check=False` disables world geometry for the solve, leaving joint limits and
  self-collision enforced — exactly the validity set $Q_{\text{valid}}$ the proposal specifies.

The solver already runs 512 IK seeds per target, so the proposal's "multi-seed solver-robust
label" needs no seed loop: one call **is** the multi-seed answer.

| Setting | Value | Why |
|---|---|---|
| Embodiment | `ARM_NO_TORSO` | TidyBot has no torso; plain `ARM` fails every solve while looking exactly like "unreachable" |
| Active cspace | `joint_1 … joint_7` | CuRobo strips locked joints; base and finger joints are held fixed |
| IK seeds | 512 | Solver-robust by construction |
| Position tolerance | 0.01 m | Recorded in shard metadata as the label's $\varepsilon$ |
| Rotation tolerance | 0.12 rad | Same |
| Batch size | 64 | Largest that fits beside Isaac on a 24 GB card |

#### The control suite

Every run embeds known-answer checks and aborts loudly on failure. Three controls, each catching a
class the others cannot, plus one assertion:

| Control | Expected | Observed | Catches |
|---|---|---:|---|
| Identity | 100 % | 100 % | Frame **and orientation** errors — FK of the robot's own rest pose |
| FK samples | ≥ 95 % | 100 % | Errors away from the rest pose; solver pessimism across the workspace |
| Far target | 0 % | 0 % | A target frame being ignored entirely (6 m away) |
| Base at origin | assert | pass | The world-frame trap — see §6.2 |

The identity control earns its place: it is the only one sensitive to orientation, and it is what
caught a quaternion-ordering bug that the other two passed for two full runs.

#### Why the scene is empty

The label is scene-free by definition, so loading the furnished kitchen bought nothing and cost a
great deal: ~16 GB of the 24 GB card went to RTX geometry, leaving so little for CuRobo that the
motion generator ran out of memory during its own constructor warm-up at batch size 8. The
throughput cap was self-inflicted.

An empty scene with the robot at the origin and `obs_modalities=[]` — no cameras, therefore no
render products — fixed the memory, cut boot time to ~30 s, and removed the dependency on one
task's demo file, which the task-agnostic scope required anyway.

A second self-inflicted cost: `CuRoboMotionGenerator` builds *every* embodiment in the robot's
config by default — four of them, each with 512 IK seeds and a full trajectory-optimization
warm-up in its constructor. Only one is ever called. Passing `embodiment_types` builds just that
one, plus `DEFAULT`, which the batched collision checker hardcodes.

### Stage 1 — dataset generation

Three samplers, each covering something the others miss, plus a fourth slice that falls out free.

| Slice | Rows | Positive | What it contributes |
|---|---:|---:|---|
| `fk` | 200,000 | 100.0 % | The reachable manifold's true *orientation* distribution, which uniform sampling badly misrepresents |
| `uniform` | 500,000 | 25.4 % | The negatives and the decision boundary |
| `robust` | 100,000 | 43.7 % | Graded margin — 27.2 % land strictly inside $(0,1)$ |
| `screen` | 300,000 | 19.1 % | By-product of boundary selection, kept as ordinary labels |
| **total** | **1,100,000** | **38.9 %** | 90 MB · ~2.0 M solves · 1.6 h on one 4090 |

#### Orientation coverage

Uniform positions in a box of $\pm1.1$ m horizontally and $0$–$1.5$ m vertically around the base.
Orientations are a deliberate mix:

- **70 % uniform on $SO(3)$** (Haar measure, via QR of a Gaussian matrix) — the model must answer
  whatever the upstream planner produces.
- **30 % biased downward** within a 75° cone — mobile-manipulation targets cluster there.

The bias is kept a minority on purpose: a straight-down assumption already produced a wrong answer
once in this project, since the real trash grasp is tilted about 23°. The two slices are reported
separately in evaluation so the bias cannot hide a coverage gap.

#### FK samples are labeled, not assumed

The proposal treats forward-kinematics samples as costless positives — a configuration reaching
them exists by construction. But that configuration may be in self-collision, which the solver
enforces and forward kinematics does not; about 8 % of uniformly-sampled configurations are. An
unverifiable label in the largest slice of the dataset is a bad trade for solve time we now know we
have, so the FK sampler is treated as a *distribution* and its poses go through the teacher like
everything else.

#### Boundary-concentrated robust anchors

The pilot exposed a real problem: at physically-grounded perturbation scales, **92.5 % of robust
anchors returned exactly 0 or 1**. All perturbations agreed with the nominal, so those rows taught
the margin head nothing that `exist` did not already say — while costing $1+J = 9$ solves each.

Widening $\sigma$ would have manufactured a gradient by measuring uncertainty the pipeline does not
have. The fix is *where* anchors are drawn, not how hard they are shaken. A cheap one-solve screen
labels $4n$ candidates; for each, the disagreement of its $k$ nearest neighbours in feature space

$$
d_i \;=\; \frac{1}{k}\sum_{j \,\in\, \mathcal{N}_k(i)} \mathbf{1}\bigl[\,y_j \neq y_i\,\bigr]
$$

identifies anchors sitting near the decision boundary. Because $d_i$ is quantized to multiples of
$1/k$, the top of the ranking is a large block of ties; breaking them toward the tightest
neighbourhoods (smallest mean neighbour distance) alone lifted boundary enrichment from ~2× to
~3.3×. A 20 % random floor keeps the slice from being exclusively hard cases.

| Robust slice | Graded in $(0,1)$ | Positive rate |
|---|---:|---:|
| Pilot — anchors drawn uniformly | 7.5 % | 63.2 % |
| **Final — boundary-concentrated** | **27.2 %** | **43.7 %** |

3.6× more usable margin signal, and a slice that rebalanced toward the decision surface it exists
to describe.

Two subtleties that are easy to get wrong, both of which were:

- Screen candidates must come from `uniform` **only**. FK poses are interior positives, ineligible
  for the boundary by construction; mixing them in spends half the screen budget on candidates
  that can never be selected, and measurably dilutes the result (1.6× instead of 3.3×).
- Promoted anchors must be **removed** from the screen rows. The robust slice is drawn *from* the
  screen pool, so returning the pool whole puts every anchor in the dataset twice — and the two
  copies can land on opposite sides of a train/test split.

#### Shards

Written atomically and skipped when already complete, because box sessions get torn down mid-run
and a two-million-solve job that cannot resume is a job that never finishes. Completeness means the
file opens **and** holds the expected row count — checking existence alone would silently accept a
file truncated by a killed session, and a short shard looks exactly like a distribution shift
during training.

Each shard carries its provenance: script md5, seed, solver tolerances, seed count, embodiment, and
the control report from the run that produced it.

### Stage 2 — splits and the training recipe

Splits are deliberately **not** random row shuffles. A random split over a dense sample of a smooth
function is nearly free to fit — every test point sits between two training points — so IID
accuracy would look excellent while saying nothing. The splits are geometric, and a guard refuses
to build them at all if any target pose appears twice.

| Split | Rows | What it measures |
|---|---:|---|
| `train` | 651,137 | — |
| `val` | 81,391 | Temperature calibration only |
| **`test_iid`** | **81,391** | **Deployment-relevant.** The shipped model trains on the whole workspace, so every query is in-distribution |
| `test_region` | 286,081 | A held-out workspace quadrant. A *structure probe*, worse by construction: memorization collapses here, learned geometry degrades gently |
| `test_orientation` | 133,625 | The region slice restricted to near-uniform orientations, so the 30 % downward bias cannot hide a gap |

#### Loss

$$
\mathcal{L} \;=\;
\underbrace{\mathrm{BCE}\bigl(z_{\mathrm{ex}},\, y_{\mathrm{ex}}\bigr)}_{\text{primary label}}
\;+\;
\underbrace{\frac{\sum_i m_i\,\mathrm{BCE}\bigl(z_{\mathrm{rob},i},\, y_{\mathrm{rob},i}\bigr)}{\sum_i m_i}}_{\text{masked margin}}
\;+\;
\underbrace{\frac{\sum_i m_i\bigl(\sigma(z_{\mathrm{rob},i}) - y_{\mathrm{rob},i}\bigr)^{2}}{\sum_i m_i}}_{\text{Brier, masked}}
$$

where $m_i \in \{0,1\}$ marks the rows that actually carry a robust label. Without the mask, the
other 91 % of rows would train the margin head toward a fabricated zero. Each ensemble member sees
the same batch and its losses are summed, so the only thing coupling them is the data.

The proposal also lists a pairwise ranking term over same-target/different-base groups. That term
belongs to a task-conditioned dataset; this model is task-agnostic and its rows are independent
targets, so there are no natural ranking groups to form. Ranking quality is measured instead where
it is actually defined — against the exact oracle in Stage 3.

#### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, lr $2\times10^{-3}$, weight decay $10^{-4}$ |
| Schedule | Cosine annealing over all steps |
| Batch size | 4,096 |
| Epochs | 30 |
| Ensemble | 5 members, independent initialization |
| Perturbation | $\sigma_{\mathrm{pos}} = 2$ cm, $\sigma_{\mathrm{rot}} = 5^\circ$, $J = 8$ |
| Calibration | Temperature on validation NLL → $T^\star = 0.851$ |
| **Wall time** | **225 s on one RTX 4090** |

---

## 4. Inference pipeline

One module, importing numpy and torch and nothing else. No OmniGibson, no Isaac, no CuRobo — which
is the point of having trained a surrogate at all.

```python
from kineready.reward import ReadinessReward

reward = ReadinessReward.from_checkpoint("kineready.pt")

# N candidate base poses (x, y, yaw) × K world grasp targets → (N,) in [0, 1]
scores = reward.score(base_poses, targets_world, priors=grasp_priors(8))
best_i, best_score = reward.best_pose(base_poses, targets_world)
```

Internally that is four steps:

1. Compute the $N\times K$ base-frame transforms analytically (§2.1).
2. Encode to 9-vectors.
3. Run **one** batched forward pass.
4. Aggregate the $K$ by noisy-OR (§2.4).

Per-target probabilities, ensemble spread, and margin scores are all available via
`return_components=True`. The default is the **lower confidence bound**, not the mean, because the
asymmetry of the errors is real; pass `use_lcb=False` for the raw calibrated probability.

### 4.1 Dropping into the existing metric

```python
metric = BasePoseMetric(robot, ik_predictor=reward)
results = metric.evaluate(base_poses, target)   # p_kin instead of a 0.34 s solve
```

> ⚠ **One caveat that is not a detail.** A probability is not a joint configuration, so with a
> learned predictor there is nothing to run the reach-collision check against. It is reported as
> `None` — "not checked" — never as `False`. Those are different claims, and conflating them would
> quietly turn an unverified pose into a cleared one. Re-verify the top candidates with exact IK
> whenever reach-collision matters; that hybrid costs one solve instead of $N$.

### 4.2 As a reward

Two composition helpers ship with it. `readiness()` combines the learned term with the three exact
ones as a weighted geometric mean, with collision as a hard gate:

$$
\Phi(b) \;=\;
\mathbf{1}\bigl[\,C_B(b) = 0\,\bigr]\;\cdot\;
\exp\!\left(
\frac{w_d \ln S_d + w_v \ln S_v + w_k \ln P^{\mathrm{LCB}}_{\mathrm{kin}}}
     {w_d + w_v + w_k}
\right)
$$

with defaults $w_d = w_v = 1$, $w_k = 2$. Geometric rather than arithmetic because a base pose that
cannot see the object is not redeemed by being well-placed for IK, and an arithmetic mean would let
three good terms outvote one fatal one.

`potential_shaping()` wraps it in Ng et al.'s potential-based form:

$$
r \;=\; \gamma\,\Phi(s') - \Phi(s)
$$

That form provably leaves the optimal policy unchanged, so a miscalibrated readiness estimate can
slow learning but cannot teach the policy to prefer a worse endpoint. That guarantee is why the
reward enters this way rather than as a raw bonus.

### 4.3 Latency

| Device | Base poses | $K$ | Wall | Per pose | vs exact IK |
|---|---:|---:|---:|---:|---:|
| CPU | 64 | 8 | 6.7 ms | 104 µs | 3,271× |
| CPU | 512 | 8 | 16.7 ms | 32.6 µs | 10,436× |
| CPU | 4,096 | 8 | 86.6 ms | 21.1 µs | 16,091× |
| GPU | 64 | 8 | 1.1 ms | 17.8 µs | 19,084× |
| GPU | 512 | 8 | 2.5 ms | 4.8 µs | 70,475× |
| **GPU** | **4,096** | **8** | **14.6 ms** | **3.6 µs** | **95,624×** |

Each row scores $N$ base poses against $K=8$ grasp candidates, so the last line is 32,768 model
queries in 14.6 ms. Exact IK for those 4,096 poses would take 23 minutes.

The CPU column matters independently: an RL rollout worker rarely has a spare GPU, and a reward
that needs one is not deployable.

---

## 5. Evaluation results

### What each number is measured on

The deployed score is $P_{\mathrm{kin}}$ — a lower confidence bound, noisy-OR'd over $K$ grasp
candidates. Two things follow, and they are easy to trip over.

**There is no ground-truth LCB, and there cannot be.** The bound describes the *model's own
epistemic uncertainty*, not a property of the world; ground truth has no uncertainty to report.
Noisy-OR is different — it estimates $P(\exists k:\ \text{candidate } k \text{ feasible})$, whose
truth value is the boolean OR $y_{\text{any}} = \max_k y_k$, obtainable by solving exact IK on all
$K$ candidates.

Neither blocks evaluation, because AUROC, FPR@recall and top-1 are **rank- and threshold-based**:
they require a score and a binary label, never a ground-truth score.

| Result | Model output scored | Ground truth | $K$ |
|---|---|---|---:|
| AUROC · AUPRC · Brier · ECE · FPR@95 (§5.1) | `p_exist` | teacher's `exist` bit | 1 |
| Top-1 · Spearman (§5.2) | `p_lcb` via `score()` | teacher's `exist` bit | 1 |
| Teacher / model vs moved robot (§5.3) | `p_lcb` via `score()` | physically-moved-robot solve | 1 |
| Symmetry augmentation (§5.4) | **full LCB + noisy-OR** | $\max_k y_k$ over the orbit | 8 |
| Latency (§4.3) | full path | — (timing only) | 8 |

#### Does the deployed transform change the offline numbers?

$\mathrm{LCB} = p_{\mathrm{exist}} - \beta\sigma$ is **not** a monotone function of
$p_{\mathrm{exist}}$ — $\sigma$ varies independently — so it can reorder pairs and move a
rank-based metric. Measured rather than assumed:

| Slice | AUROC `p_exist` | AUROC `p_lcb` | $\Delta$ | FPR@95 `p_exist` | FPR@95 `p_lcb` | $\Delta$ | pairs reordered |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_iid` | 0.9997 | 0.9997 | −0.00001 | 0.0014 | 0.0014 | −0.00006 | 0.33 % |
| `test_region` | 0.9972 | 0.9951 | −0.00215 | 0.0135 | **0.0128** | −0.00070 | 0.66 % |
| `test_orientation` | 0.9971 | 0.9949 | −0.00219 | 0.0138 | **0.0133** | −0.00046 | 0.80 % |

On the deployment-relevant slice the headline figures are unchanged, so §5.1 stands for the shipped
score as well as for `p_exist`. Under distribution shift the bound costs ~0.002 AUROC and *buys* a
lower false-positive rate — the intended trade, and it scales with uncertainty (mean LCB shift
0.0048 on IID versus 0.0144 on the held-out quadrant).

> **Do not read ECE on `p_lcb`.** It rises from 0.0018 to 0.0043 on IID, which is correct
> behaviour, not a regression: the bound is *deliberately* biased low. Calibration is a property of
> `p_exist`; demanding it of the LCB would defeat the bound's purpose.

### 5.1 Against the baselines it has to beat

A learned model is only worth its complexity if it beats the cheap thing. All four baselines are
fit on the **same** labels from the **same** training split, so the comparison isolates the model
rather than the data.

- `distance_band` — is the target within $[r_{\text{lo}}, r_{\text{hi}}]$ of the base? The rule the
  metric used before any of this.
- `cylinder` — radial distance **and** height both in range; the standard hand-drawn workspace
  approximation for a shoulder-mounted arm.
- `rm4d_histogram` — an RM4D-style reachability map: bin the labels by
  $(r,\; z,\; \text{tool tilt},\; \text{azimuth relative to the reaching direction})$ and look up
  the empirical positive rate. This is the real competition: same data, no training, genuine
  orientation dependence.
- `knn` — $k$-nearest-neighbour vote in feature space. Not deployable (it needs the whole dataset
  at query time) but it bounds how much signal the features carry at all.

**`test_iid`** — 81,391 rows, 36.5 % positive:

| Model | AUROC | AUPRC | Brier | ECE | FPR @ 95 % recall |
|---|---:|---:|---:|---:|---:|
| **KineReady MLP** | **0.9997** | **0.9995** | **0.0059** | **0.0018** | **0.0014** |
| k-NN (k=16) | 0.9961 | 0.9927 | 0.0307 | 0.0475 | 0.0241 |
| RM4D histogram | 0.9793 | 0.9643 | 0.0663 | 0.1133 | 0.0913 |
| Cylinder bound | 0.7932 | 0.5888 | 0.3293 | 0.3843 | 0.4327 |
| Distance band | 0.6192 | 0.4326 | 0.5103 | 0.5339 | 0.7275 |

The gap that matters is the last column. At a threshold tuned to rarely miss a reachable target,
the histogram green-lights an unreachable one 9.1 % of the time and the MLP 0.14 % — **65× fewer**.

$$
\mathrm{FPR@}R \;=\; \Pr\bigl[\,\hat{p} \ge \tau_R \;\big|\; y = 0\,\bigr],
\qquad
\tau_R = \mathrm{quantile}_{1-R}\bigl(\{\hat{p}_i : y_i = 1\}\bigr)
$$

$$
\mathrm{ECE} \;=\; \sum_{b=1}^{B}\frac{|\mathcal{B}_b|}{n}\,
\Bigl|\;\mathrm{conf}(\mathcal{B}_b) - \mathrm{acc}(\mathcal{B}_b)\;\Bigr|
$$

**Generalization slices** — 286,081 and 133,625 rows:

| Model | Region AUROC | Region FPR@95 % | Orientation AUROC | Orientation FPR@95 % |
|---|---:|---:|---:|---:|
| **KineReady MLP** | **0.9972** | **0.0135** | **0.9971** | **0.0138** |
| RM4D histogram | 0.9771 | 0.1116 | 0.9760 | 0.1074 |
| k-NN (k=16) | 0.9632 | 0.1585 | 0.9600 | 0.1803 |
| Cylinder bound | 0.8109 | 0.4259 | 0.8144 | 0.4405 |
| Distance band | 0.6292 | 0.7025 | 0.6247 | 0.7204 |

The MLP loses 0.0025 AUROC on a workspace quadrant it never trained on, while k-NN — which can only
interpolate — drops *below* the histogram it beat on IID data. That reordering is the evidence that
the network learned the arm's geometry rather than memorizing samples.

### 5.2 Ranking against the exact oracle

Offline AUROC answers "is each individual prediction right", which is not the question the metric
exists for. A navigation policy picks **one** endpoint out of many, so what matters is whether the
pose it ranks first actually works. A model can have excellent AUROC and still rank badly if its
errors concentrate among the top candidates — precisely where they hurt.

| Metric | Value | Reading |
|---|---:|---|
| **Top-1 feasible** | **100 %** | In every discriminative config, the highest-ranked pose was genuinely reachable |
| Top-5 hit rate | 100 % | — |
| Mean Spearman $\rho$ | 0.877 | Correlation with the oracle's full ordering |
| Configs used | 19 / 20 | One config had every candidate infeasible and cannot discriminate any ranker; averaging it in would have inflated the score with a free win |
| Speedup, same call | 86.5× | 1.95 ms vs 169 ms for 64 poses, including the oracle's own batching |

Each config is one world target with 64 candidate base poses — the decision an endpoint selector
actually faces. The plan's target was ≥ 90 % top-1.

### 5.3 Are the labels right in the first place?

Every number above is measured against the teacher. So the teacher itself was checked against a
ground truth that does not depend on its frame convention at all:

> Choose a base pose $p$ and a target $L$ expressed in the arm's frame. The world target is
> $W = T(p)\,L$. Solve IK with the base joints **actually set to** $p$ and $W$ given in world
> coordinates — the formulation independently validated against a *physically moved* robot,
> 12 / 12 and 10 / 10.

A negative control runs alongside it: score the same world target against the **wrong** base pose.
If that agrees as well as the correct pairing, the test cannot see frame errors and proves nothing.

| Regime | Feasible | Teacher | Model | Negative control |
|---|---:|---:|---:|---:|
| **Broad targets, base ±2.5 m from origin** | 20 / 48 | **100 %** | **100 %** | 60 % |
| Kitchen-grasp regime (1.02 m, tilted 23°) | 13 / 48 | **100 %** | 93.8 % | 75 % |

The control sitting well below the correct pairing is what makes the 100 % meaningful.

### 5.4 Symmetry augmentation, on the real task

The worked example scores base poses for the pick-trash-and-dispose grasp, comparing a single demo
grasp against its 8-member symmetry orbit.

| Scoring | Base poses that can grasp | AUROC vs orbit oracle |
|---|---:|---:|
| Demo grasp only | 15.2 % | 0.9540 |
| **$K=8$ symmetry orbit, noisy-OR** | **21.9 %** | **0.9996** |

256 candidate base poses. Symmetry augmentation recovers a 44 % larger set of usable standing
positions — poses a single-grasp metric would have discarded outright.

The orbit is generated by rotating about the **object's** symmetry axis, not the gripper's:

$$
T_k \;=\; G_k\, T_{\text{grasp}},
\qquad
G_k = \begin{bmatrix} R_{\hat{a}}(\alpha_k) & p - R_{\hat{a}}(\alpha_k)\,p \\[2pt] \mathbf{0}^{\!\top} & 1\end{bmatrix},
\qquad
\alpha_k = \frac{2\pi k}{K}
$$

with $p$ a point on the axis and $\hat{a}$ its direction. Rotating in the gripper frame instead
would spin the tool in place and produce grasps that miss the object entirely — an easy mistake to
make and impossible to see in an aggregate score, so every orbit member is checked to preserve its
distance to the axis and its height.

---

## 6. Failure modes and why the obvious checks missed them

Each of these produced confident, plausible, completely wrong output. None was visible in any
rendering. They are recorded because the mitigations are the reason the numbers above can be
trusted.

### 6.1 The quaternion that looked like physics **[FIXED]**

The FK control sat at 72 % and no geometric story explained it. The cause: `compute_trajectories`
applies its xyzw → wxyz permutation **outside** the `is_local` guard, so callers must pass xyzw
even for local targets. Passing wxyz permuted it twice:

$$
(w,\,x,\,y,\,z) \;\longmapsto\; (z,\,w,\,x,\,y)
$$

— a valid-looking unit quaternion naming a completely different rotation.

Positions passed through correctly, so targets landed in the right place with a scrambled
orientation, and IK succeeded whenever that scramble happened to be reachable. The result was a
believable reachability rate. The 6-metre control was blind to it, because nothing at 6 m is
reachable at any orientation.

> **The rule that came out of it: at least one control must be orientation-sensitive.**

The identity control caught it immediately, at 50 % where 100 % is the only correct answer.

### 6.2 The gate that could not fail **[FIXED]**

For a holonomic base, `is_local=True` is a **misnomer**. Measured directly: the link CuRobo treats
as its base (`base_footprint_x`) sits at the **world origin with identity orientation no matter
where the robot is** — the world pose lives entirely in the base joints. So the wrapper's
world → base conversion is the identity, and "local" targets are really *world* targets.

The teacher's documented frame is therefore correct only while the arm is at the origin — which is
exactly how every shard was generated, so **the dataset is sound**. But build a teacher in a
furnished scene with the robot at $(4.79,\,-1.28)$ and ask about a target 0.3 m in front of it, and
CuRobo is asked about a point 4.8 m from the arm. Everything returns unreachable, silently.
Measured target-frame error: **6.93 m**.

Nothing already in place could have caught it:

- The controls are self-consistent round-trips through the same frame, so a shared wrong frame is
  invisible to them.
- The frame gate ran in the empty scene at the origin, where the correct and incorrect frames
  coincide. **A gate that cannot fail is not a gate.**

What caught it was building ground truth the other way round (§5.3) and adding the negative
control. `IKTeacher` now refuses to construct unless the base joints are at zero.

The correct formulation when the robot is elsewhere is `base_pose_metric._solve_ik`'s: **target in
world coordinates, base joints locked to the candidate** via `initial_joint_pos`.

### 6.3 Five more, in brief

| Bug | Consequence | Status |
|---|---|---|
| Four embodiments built where one was needed | Each with 512 seeds and a trajopt warm-up; made batch size 8 exhaust a 24 GB card | **[FIXED]** |
| A constant reported as a measurement | The FK sampler's "% rejected as self-colliding" was computed *after* truncating to $n$, making it arithmetically equal to $1 - 1/\text{oversample}$ regardless of the checker. 0.667 was quoted for two runs; the real value is ~8 % | **[FIXED]** |
| Train/test leak by duplication | Robust anchors are selected *from* the screen pool; returning the pool whole put every anchor in twice, and the copies could land on opposite sides of a split | **[FIXED]** |
| Irreproducible seeding | `hash()` on strings is randomized per process, so a resumed run drew a different stream and the recorded seed did not reproduce the dataset. Now CRC32 | **[FIXED]** |
| Double forward pass | The ensemble indexed `m(x)[0]` and `m(x)[1]` in two comprehensions, running the whole trunk twice per member — doubling the inference cost of the thing whose entire purpose is being cheap | **[FIXED]** |

### 6.4 Open **[UNRESOLVED]**

`base_pose_metric.evaluate()` reported all 25 kitchen base poses IK-feasible where frame-independent
ground truth puts the rate near 27 %. `_solve_ik` called directly is correct (12/12 and 10/10
against a physically moved robot), so the over-reporting lives in `evaluate()`'s wrapper path rather
than the solver.

This predates KineReady and is not yet diagnosed — worth chasing before trusting that metric's
feasibility counts.

---

## 7. Reproduction

```bash
# Stage 0 — prove the labeler is correct, then measure its throughput
OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=5 \
  python kineready/examples/stage0_benchmark.py

# Stage 1 — pilot first, then the full run. Shards resume if the session dies.
OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=5 \
  python kineready/examples/stage1_datagen.py --pilot --out kineready_data/pilot
OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=5 \
  python kineready/examples/stage1_datagen.py --out kineready_data/full

# Stages 2 and 3 — train, then evaluate. No simulator needed for training.
GPU=5 bash kineready/examples/run_stage23.sh

# Correctness, against frame-independent ground truth
python kineready/examples/verify_model_vs_truth.py --regime broad
python kineready/examples/verify_model_vs_truth.py --regime grid
```

Offline tests — frame maths against scipy, sampler validity, split disjointness, the invariance
property, the world-frame trap — run in under a second with no simulator:

```bash
pytest tests/test_kineready_frames.py tests/test_kineready_pipeline.py \
       tests/test_base_pose_geometry.py      # 46 passed
```

### Deferred, and recorded as such

- **MotionReady** and point-cloud conditioning — environment effects stay with the exact terms.
- **Learned collision fields** — the batched exact check already costs 0.5 ms for a whole sweep.
- **Boundary / active retraining round** — the plan made it conditional on ranking falling short of
  90 % top-1. It reached 100 %.
- **RL training of the navigation policy itself** — `reward.py` ships the potential-shaping helper
  ready for that loop.

---

*Measured on one RTX 4090 with Isaac Sim resident. Dataset 1.1 M rows / 90 MB; checkpoint 1.2 MB.
Package quick-reference lives alongside this file in `README.md`.*
