# KineReady: Efficient State-Based Manipulation Readiness for Navigation-to-Manipulation Handoff

## Learning Robust Kinematic Reachability and Collision-Free Motion Feasibility as High-Frequency Navigation Rewards

## 1. Abstract

Long-horizon mobile manipulation requires a navigation policy to terminate at a base pose from which the subsequent manipulation task is actually executable. Conventional navigation objectives based primarily on distance to the target are insufficient: a robot may be close to and facing an object while the desired end-effector pose remains kinematically unreachable or while no collision-free arm motion exists.

This proposal introduces **KineReady**, a lightweight manipulation-readiness framework that explicitly separates semantic perception, geometric localization, kinematic feasibility, and motion safety. Rather than predicting manipulation feasibility directly from images with a large vision-language model, KineReady assumes that an upstream perception or navigation module has already localized the target object or actionable part. The desired object-relative interaction poses are transformed into the robot arm-base frame, after which a compact state-based neural network predicts robust inverse-kinematics reachability.

The core model therefore implements

$$
F_{\mathrm{kin},\theta}:
T_E^A
\rightarrow
P(\text{robust IK feasibility}),
$$

or, when current robot configuration matters,

$$
F_{\mathrm{kin},\theta}:
(T_E^A,q_0)
\rightarrow
P(\text{local IK feasibility}).
$$

This formulation removes vision from the high-frequency kinematic-scoring pathway. Distance-to-object, object visibility, and mobile-base collision can continue to be computed analytically or directly from simulator/perception state, while only the expensive IK-related component is approximated by a learned surrogate.

Task-space reachability has already been represented successfully using compact neural networks and support-vector models in differentiable reachability-map research, while inverse reachability maps have been explicitly applied to mobile-manipulator base placement [1,2].

We additionally propose **MotionReady**, a scene-conditioned extension that predicts whether a collision-free motion from the current arm configuration to a desired interaction pose is likely to be found:

$$
F_{\mathrm{motion},\phi}:
(T_E^A,q_0,\mathcal E)
\rightarrow
P(\text{collision-free motion feasibility}),
$$

where $\mathcal E$ denotes local environment geometry represented by a point cloud, voxel map, or distance field.

The learned models are intended for high-frequency reward shaping and candidate filtering, **not as safety certificates**. Before handing control to the manipulation policy, the top-ranked terminal base poses are verified using exact IK, collision checking, and motion planning. GPU-accelerated packages such as cuRobo provide batched IK, collision checking, geometric planning, and trajectory optimization suitable for both offline label generation and terminal verification [3].

The primary research hypothesis is that a carefully factorized state-based feasibility model can provide most of the useful manipulation awareness of a substantially heavier vision-language critic while offering significantly lower latency, higher RL-training throughput, stronger interpretability, and simpler generalization analysis.

---

# 2. Motivation

Consider a mobile manipulator navigating toward an object at world pose $T_O^W$.

A conventional navigation policy may optimize

$$
d(b,O)
=
\|p_B^W(b)-p_O^W\|,
$$

where $b$ is the mobile-base pose.

However,

$$
d(b,O)\rightarrow 0
$$

does not imply

$$
\exists q:
FK(q)=T_E^{A(b)}.
$$

Likewise,

$$
\exists q^\star:
FK(q^\star)=T_E^{A(b)}
$$

does not imply that a collision-free trajectory exists from the current arm configuration $q_0$ to $q^\star$.

The relevant hierarchy is therefore

$$
\boxed{
\text{Object Proximity}
\neq
\text{IK Reachability}
\neq
\text{Collision-Free Goal}
\neq
\text{Collision-Free Motion}
}
$$

The navigation policy should ideally terminate in states satisfying all manipulation prerequisites while avoiding the computational expense of invoking a full IK solver or motion planner at every RL simulation step.

This motivates a learned approximation of only the expensive components.

---

# 3. Key Design Principle: Factorize Semantics from Kinematics

The proposed system deliberately avoids solving the following problem:

$$
(\text{RGB image},\text{text instruction})
\rightarrow
\text{manipulation feasibility}.
$$

Instead, it factorizes the pipeline as

```text
Task instruction
       |
       v
Semantic perception / upstream navigation
       |
       v
Target object or actionable-part pose
       |
       v
Interaction / grasp pose generation
       |
       v
Target end-effector pose
       |
       +----------------------------+
       |                            |
       v                            v
State-based KineReady        Exact cheap metrics
       |                    distance / visibility
       |                      / base collision
       +-------------+--------------+
                     |
                     v
           Manipulation readiness
                     |
                     v
             Navigation reward
```

The semantic module answers:

> **What should the robot manipulate, and where is it?**

KineReady answers:

> **Given the resulting end-effector target relative to this candidate base pose, is it kinematically reachable?**

MotionReady answers:

> **Given the local geometry and current arm state, is the target likely reachable through a collision-free motion?**

This separation is central to the proposal.

For a fixed robot, IK reachability is fundamentally a low-dimensional task-space property. A model should not need to rediscover robot kinematics from RGB pixels every time it is queried.

---

# 4. Problem Formulation

## 4.1 Coordinate Frames

We define:

- $W$: world frame;
- $B$: mobile-base frame;
- $A$: arm-root frame;
- $O$: target object or actionable-part frame;
- $E$: desired end-effector frame.

The mobile-base configuration is

$$
b=(x_b,y_b,\psi_b)\in SE(2).
$$

If the robot includes an actuated torso, lift, or other mechanism affecting the arm root, denote this state by $h$.

The arm-root pose in the world is

$$
T_A^W(b,h)
=
T_B^W(b)T_A^B(h).
$$

---

# 5. From Object Pose to Manipulation Target Pose

The upstream semantic/localization module estimates

$$
\hat T_O^W.
$$

However, object pose alone is generally insufficient for manipulation.

For task $\tau$, define a set of object-relative interaction poses

$$
\mathcal G_\tau
=
\left\{
(T_{E,k}^O,\pi_k)
\right\}_{k=1}^{K},
$$

where:

- $T_{E,k}^O$ is candidate end-effector pose $k$ relative to the object or actionable-part frame;
- $\pi_k$ is an optional prior probability or quality score;
- $K$ is the number of grasp, approach, placement, or interaction candidates.

For candidate base pose $b$, interaction pose $k$ is transformed into the arm-root frame as

$$
\boxed{
T_{E,k}^{A(b)}
=
\left(T_A^W(b,h)\right)^{-1}
\hat T_O^W
T_{E,k}^{O}
}
$$

This transformation is important because it converts mobile-base placement into a conventional manipulator reachability query.

---

# 6. Core Model: KineReady

## 6.1 Existential IK Reachability

The most fundamental feasibility definition is

$$
Y_{\mathrm{IK}}^{\mathrm{exist}}(T_E^A)
=
\mathbf 1
\left[
\exists q\in\mathcal Q_{\mathrm{valid}}
:
d_{SE(3)}(FK(q),T_E^A)<\epsilon
\right],
$$

where $\mathcal Q_{\mathrm{valid}}$ additionally satisfies:

$$
q_{\min}\le q\le q_{\max},
$$

$$
C_{\mathrm{self}}(q)=0,
$$

and, optionally,

$$
C_{\mathrm{robot-body}}(q)=0.
$$

Thus, a positive example corresponds to at least one configuration satisfying:

1. the desired end-effector pose;
2. joint limits;
3. self-collision constraints;
4. collisions between the arm and fixed robot-body geometry such as the torso or mobile base.

This definition **does not require the current arm configuration $q_0$**.

It asks:

> Does any valid configuration exist?

This is closely related to conventional reachability maps and the recently proposed differentiable reachability maps [1,2].

---

## 6.2 Robust IK Reachability

A binary IK label is undesirable near workspace boundaries.

Suppose the perception system estimates

$$
\hat T_E^A,
$$

but the true target has uncertainty

$$
\xi
\sim
\mathcal N(0,\Sigma_T).
$$

Generate perturbed target poses

$$
\tilde T_{E,j}^{A}
=
\hat T_E^A
\exp
\left(
\xi_j^\wedge
\right),
$$

with

$$
\xi_j
\sim
\mathcal N(0,\Sigma_T).
$$

Define robust feasibility as

$$
\boxed{
Y_{\mathrm{IK}}^{\mathrm{robust}}
=
\frac{1}{J}
\sum_{j=1}^{J}
Y_{\mathrm{IK}}^{\mathrm{exist}}
\left(
\tilde T_{E,j}^{A}
\right)
}
$$

so that

$$
Y_{\mathrm{IK}}^{\mathrm{robust}}
\in[0,1].
$$

This provides a much more useful navigation signal.

For example:

- a target deep inside the workspace may have score $0.99$;
- a target on a workspace boundary may have score $0.55$;
- an unreachable target may have score $0.01$.

This captures **reachability margin** rather than merely nominal reachability.

---

# 7. Solver-Robust Labels

A numerical IK failure does not necessarily prove mathematical infeasibility.

For every perturbed target pose, use multiple seeds

$$
q_1^{\mathrm{seed}},
\dots,
q_S^{\mathrm{seed}}.
$$

Define

$$
Y_j
=
\mathbf 1
\left[
\exists s\in\{1,\dots,S\}:
IK(\tilde T_{E,j}^A;q_s^{\mathrm{seed}})
\text{ succeeds}
\right].
$$

The robust label becomes

$$
Y_{\mathrm{IK}}^{\mathrm{robust}}
=
\frac{1}{J}
\sum_{j=1}^{J}Y_j.
$$

The training pipeline should categorize samples as:

### Positive

At least one sufficiently diverse solver seed produces a verified valid solution.

### Negative

All seeds fail despite a sufficiently large solver budget.

### Ambiguous

Results vary substantially across:

- IK seeds;
- solver settings;
- solvers;
- pose perturbations.

Ambiguous samples can initially be removed or assigned reduced loss weight.

cuRobo is particularly suitable as a teacher because it provides massively parallel GPU implementations of inverse kinematics and motion generation [3].

---

# 8. Local IK Feasibility

Although current arm state $q_0$ is unnecessary for existential reachability, it becomes relevant if the question changes to:

> Can the solver reach a convenient solution from the robot's current state?

Define

$$
F_{\mathrm{local}}
(T_E^A,q_0)
=
P(
\text{IK succeeds from states near }q_0
).
$$

Generate perturbed seeds

$$
\tilde q_s
=
q_0+\delta q_s
$$

and define

$$
Y_{\mathrm{local}}
=
\frac{1}{S}
\sum_{s=1}^{S}
\mathbf 1
\left[
IK(T_E^A;\tilde q_s)
\text{ succeeds}
\right].
$$

This quantity captures effects such as:

- redundant IK branches;
- proximity to joint limits;
- large joint displacement;
- solver initialization sensitivity.

For the first implementation, if the arm remains in one standardized stow configuration while navigating, $q_0$ should be removed entirely from the KineReady input.

This produces the simplest and fastest model.

---

# 9. KineReady Input Representation

The target pose consists of position and orientation.

Position:

$$
p_E^A
=
[x,y,z].
$$

For orientation, use the continuous 6D rotation representation proposed by Zhou et al. rather than directly using Euler angles [4].

Thus the basic feature vector is

$$
x_{\mathrm{kin}}
=
[
x,y,z,r_1,r_2,r_3,r_4,r_5,r_6
].
$$

The core model therefore receives only nine scalar values.

Optional extensions include

$$
x_{\mathrm{kin}}
=
[
p_E^A,
R_{6D},
\sin q_0,
\cos q_0,
h,
\rho,
\operatorname{diag}(\Sigma_T)
],
$$

where:

- $q_0$: current arm joint state;
- $h$: torso/lift configuration;
- $\rho$: robot morphology encoding;
- $\Sigma_T$: target-pose uncertainty.

For a fixed robot and fixed navigation arm posture, the MVP should use only

$$
\boxed{
x_{\mathrm{kin}}
=
[p_E^A,R_{6D}]
}.
$$

---

# 10. KineReady Architecture

No Transformer, CNN, or vision backbone is required.

A suitable initial architecture is

```text
9-dimensional target-pose state
              |
              v
        Linear(9, 128)
              |
             SiLU
              |
        Linear(128, 128)
              |
             SiLU
              |
        Residual MLP block
              |
        Linear(128, 64)
              |
             SiLU
              |
       +------+-------+
       |              |
       v              v
 p_exist head   p_robust head
```

A concrete specification is:

```python
Input: 9 dimensions

MLP:
    Linear(9, 128)
    SiLU

    Linear(128, 128)
    SiLU

    Linear(128, 128)
    SiLU
    Residual connection

    Linear(128, 64)
    SiLU

Heads:
    Linear(64, 1)  # existential reachability
    Linear(64, 1)  # robust reachability
```

Optional auxiliary heads can predict:

$$
\hat d_q
=
\min_{q^\star}
\|q^\star-q_0\|_W,
$$

joint-limit margin,

$$
m_{\mathrm{joint}},
$$

or manipulability,

$$
m_{\mathrm{manip}}.
$$

The purpose of these auxiliary tasks is representation regularization rather than direct reward construction.

---

# 11. Multiple Interaction Poses

For every candidate base pose, evaluate all interaction candidates:

$$
p_k
=
F_{\mathrm{kin},\theta}
\left(
T_{E,k}^{A(b)}
\right).
$$

The simplest aggregation is

$$
P_{\mathrm{kin}}(b)
=
\max_k p_k.
$$

A smoother alternative is a weighted noisy-OR:

$$
\boxed{
P_{\mathrm{kin}}(b)
=
1-
\prod_{k=1}^{K}
(1-\pi_kp_k)
}
$$

which approximates the probability that at least one candidate interaction is feasible.

This formulation naturally handles an object with multiple valid grasp or manipulation directions.

---

# 12. Offline KineReady Dataset Generation

Training data should come from several complementary distributions.

## 12.1 Forward-Kinematics Positives

Randomly sample

$$
q\sim p(q)
$$

subject to robot validity constraints and compute

$$
T_E^A=FK(q).
$$

These samples provide inexpensive guaranteed positive examples.

---

## 12.2 Uniform Task-Space Queries

Sample candidate poses from a large volume surrounding the robot:

$$
p_E^A
\sim
\mathcal U(\mathcal V),
$$

$$
R_E^A
\sim
p(R).
$$

Run the exact teacher IK solver to label each pose.

This generates both reachable and unreachable examples.

---

## 12.3 Task-Distribution Samples

Generate realistic mobile-manipulation scenes and sample:

- object positions;
- object orientations;
- base poses;
- grasp candidates;
- torso states;
- target-pose noise.

Transform each candidate into the arm-root frame and query the teacher.

This distribution is particularly important because it matches states likely to be encountered during navigation.

---

## 12.4 Workspace-Boundary Sampling

Uniform samples waste substantial compute on trivial examples.

After training an initial model, sample densely where

$$
P_{\mathrm{kin}}\in[0.3,0.7],
$$

or where an ensemble exhibits high disagreement.

Useful regions include:

- workspace boundaries;
- singularity regions;
- joint-limit boundaries;
- self-collision boundaries;
- transitions between different redundant IK branches.

This active-sampling phase should consume a significant fraction of the expensive IK-label budget.

---

# 13. KineReady Training Objective

The core loss is

$$
\begin{aligned}
\mathcal L_{\mathrm{kin}}
=&
\lambda_e
\mathcal L_{\mathrm{BCE}}
(
\hat p_{\mathrm{exist}},
y_{\mathrm{exist}}
)
\\
&+
\lambda_r
\mathcal L_{\mathrm{BCE}}
(
\hat p_{\mathrm{robust}},
y_{\mathrm{robust}}
)
\\
&+
\lambda_b
\mathcal L_{\mathrm{Brier}}
(
\hat p_{\mathrm{robust}},
y_{\mathrm{robust}}
)
\\
&+
\lambda_{\mathrm{rank}}
\mathcal L_{\mathrm{rank}}
\\
&+
\lambda_{\mathrm{aux}}
\mathcal L_{\mathrm{aux}}.
\end{aligned}
$$

For two base poses $b_i$ and $b_j$ for the same target, if

$$
y_i>y_j,
$$

the pairwise ranking loss can be

$$
\mathcal L_{\mathrm{rank}}
=
-\log
\sigma
\left(
z_i-z_j
\right),
$$

where $z_i$ and $z_j$ are unnormalized readiness logits.

This is useful because navigation primarily needs to determine:

> Is candidate $b_i$ preferable to candidate $b_j$?

rather than perfectly reconstructing an arbitrary scalar score.

---

# 14. Uncertainty-Aware Reachability

For navigation handoff, false-positive reachability predictions are more dangerous than conservative false negatives.

A straightforward implementation is a small ensemble

$$
\{
F_{\theta_1},\dots,F_{\theta_M}
\}.
$$

Define

$$
\mu_{\mathrm{kin}}
=
\frac{1}{M}
\sum_{m=1}^{M}
p_m,
$$

and

$$
\sigma_{\mathrm{kin}}^2
=
\frac{1}{M}
\sum_{m=1}^{M}
(p_m-\mu_{\mathrm{kin}})^2.
$$

Use a lower-confidence score:

$$
\boxed{
P_{\mathrm{kin}}^{\mathrm{LCB}}
=
\operatorname{clip}
\left(
\mu_{\mathrm{kin}}
-
\beta\sigma_{\mathrm{kin}},
0,1
\right)
}
$$

where $\beta$ controls conservativeness.

An exact IK query can be triggered when

$$
\sigma_{\mathrm{kin}}>\tau_\sigma.
$$

---

# 15. Manipulation-Readiness Reward

The current proposal assumes that the following quantities are inexpensive:

1. distance to target;
2. target visibility;
3. base collision.

Therefore, they should remain exact rather than being replaced by learned predictions.

The only learned core term is kinematic feasibility.

---

## 15.1 Distance Score

Raw distance is

$$
d(b)
=
\|p_O^W-p_B^W(b)\|.
$$

Closer is not always better.

Define a preferred distance band:

$$
S_d(b)
=
\sigma
\left(
\frac{d(b)-d_{\min}}{\tau_d}
\right)
\sigma
\left(
\frac{d_{\max}-d(b)}{\tau_d}
\right).
$$

This produces

$$
S_d(b)\approx 0
$$

when the robot is either too far from or too close to the manipulation target.

---

# 16. Visibility Score

For the current actually observed robot pose,

$$
S_v
=
\frac{
N_{\mathrm{visible,target}}
}{
N_{\mathrm{projected,target}}
}.
$$

Whenever possible, visibility should refer to the **actionable region**, such as:

- drawer handle;
- grasp surface;
- receptacle opening;

rather than the full object.

A distinction is important:

- at the robot's **current state**, visibility can be computed from the actual observation;
- for an **unvisited candidate pose**, exact visibility requires a geometric map, renderer, or learned predictor.

The MVP navigation reward only needs the current-state score, avoiding this complication.

---

# 17. Exact Base Collision

Let

$$
C_B(b)\in\{0,1\}
$$

indicate whether the mobile-base footprint is in collision.

Since this check is assumed inexpensive, it remains an exact hard constraint.

---

# 18. Core Readiness Potential

A simple arithmetic weighted sum can allow an excellent distance score to compensate for near-zero reachability.

Instead, define

$$
\boxed{
\Phi_A(b)
=
\mathbf 1[C_B(b)=0]
\exp
\left(
\frac{
w_d\log(S_d+\epsilon)
+
w_v\log(S_v+\epsilon)
+
w_k\log(P_{\mathrm{kin}}^{\mathrm{LCB}}+\epsilon)
}{
w_d+w_v+w_k
}
\right)
}
$$

This is a weighted geometric mean.

It has the useful property that a near-zero necessary condition strongly suppresses the total readiness.

An even more conservative alternative is

$$
\Phi_A(b)
=
\min
\left\{
S_d,
S_v,
P_{\mathrm{kin}}^{\mathrm{LCB}}
\right\}
$$

after enforcing base-collision constraints.

Both variants should be evaluated experimentally.

---

# 19. RL Reward Shaping

Using

$$
r_t=\Phi_A(s_t)
$$

directly may encourage an agent to remain inside a high-reward region.

Instead, use manipulation readiness as a potential:

$$
\boxed{
r_t^{\mathrm{ready}}
=
\gamma\Phi_A(s_{t+1})
-
\Phi_A(s_t)
}
$$

and define the complete reward as

$$
r_t
=
r_t^{\mathrm{nav}}
+
\lambda_{\mathrm{ready}}
r_t^{\mathrm{ready}}
-
\lambda_c C_t.
$$

This encourages improvement in manipulation readiness rather than simply rewarding occupancy of a high-score region.

Potential-based reward shaping originates from the policy-invariance formulation of Ng, Harada, and Russell [5].

The corresponding theoretical guarantees rely on the underlying assumptions of that formulation; empirical validation is still required for partially observable, truncated, or approximate learned-reward settings.

---

# 20. Handoff to Manipulation

Navigation can nominate a state for handoff when

$$
S_v>\tau_v,
$$

$$
P_{\mathrm{kin}}^{\mathrm{LCB}}>\tau_k,
$$

$$
C_B=0.
$$

Instead of immediately executing manipulation, maintain the best $K_{\mathrm{terminal}}$ recent candidate poses.

At handoff:

1. run exact multi-seed IK for all top-$K$ candidates;
2. reject candidates with invalid endpoint configurations;
3. perform exact environment-collision checks;
4. run an exact motion planner;
5. select the lowest-cost verified candidate;
6. reposition if necessary;
7. execute the manipulation policy.

Thus,

$$
\boxed{
\text{Learned model}
=
\text{fast scorer/filter}
}
$$

while

$$
\boxed{
\text{Exact planner}
=
\text{terminal verifier}
}.
$$

---

# 21. Variant: MotionReady

## 21.1 Motivation

KineReady evaluates kinematic reachability.

It does **not** solve the following problem:

$$
\exists\tau:
q_0
\xrightarrow{\tau}
q^\star
$$

subject to

$$
C(q(t),\mathcal E)=0
\qquad
\forall t.
$$

Consider two environments with identical:

$$
(T_E^A,q_0),
$$

but different obstacle placements.

Their IK feasibility can be identical while their motion feasibility differs completely.

Therefore external collision-free motion cannot generally be predicted from pure robot state alone.

It requires environment geometry.

---

# 22. MotionReady Formulation

Let

$$
\mathcal E
$$

denote the local three-dimensional environment.

MotionReady predicts

$$
\boxed{
F_{\mathrm{motion},\phi}
:
(T_E^A,q_0,\mathcal E)
\rightarrow
P_{\mathrm{motion}}.
}
$$

An important distinction is that this probability should not initially be interpreted as

$$
P(
\exists\text{ mathematically valid trajectory}
).
$$

Instead, define a specific teacher planner $\mathcal P$ and planning budget $B_{\mathrm{plan}}$.

The supervised target is

$$
\boxed{
Y_{\mathrm{motion}}
=
\mathbf 1
\left[
\mathcal P
(q_0,T_E^A,\mathcal E;B_{\mathrm{plan}})
\text{ returns a verified trajectory}
\right].
}
$$

Thus MotionReady predicts

> **the probability that the chosen teacher planner succeeds under a fixed computational budget.**

This definition produces reproducible labels and avoids incorrectly treating planner failure as a formal proof of infeasibility.

Neural Feasibility Checking previously demonstrated the broader principle of learning inexpensive feasibility classifiers from simulated IK/planning supervision to reduce expensive planning calls in manipulation pipelines [6].

---

# 23. Separating Goal Collision and Trajectory Collision

MotionReady should predict at least two probabilities.

### Collision-Free Goal Feasibility

$$
P_{\mathrm{goal-safe}}
=
P
\left[
\exists q^\star:
FK(q^\star)=T_E^A
\land
C(q^\star,\mathcal E)=0
\right].
$$

### Collision-Free Motion Feasibility

$$
P_{\mathrm{motion}}
=
P
\left[
\exists\tau:
q_0
\xrightarrow{\tau}
q^\star,
\;
C(q(t),\mathcal E)=0
\;\forall t
\right].
$$

A safe endpoint does not imply a safe trajectory.

This distinction is fundamental in trajectory planning and motivates learned collision representations such as Fastron, DiffCo, configuration-space distance fields, and CSSDF-Net [7–10].

---

# 24. Scene Representation

The initial MotionReady implementation should use a local obstacle point cloud.

Let

$$
P_{\mathcal E}
=
\{p_i\}_{i=1}^{N},
$$

with

$$
p_i\in\mathbb R^3.
$$

Transform the cloud into the arm-root frame:

$$
p_i^A
=
(T_A^W)^{-1}p_i^W.
$$

Recommended initial setting:

$$
N=1024
$$

or

$$
N=2048.
$$

The point set should be cropped to the manipulation workspace.

Sampling should prioritize points near:

- the robot arm;
- target object;
- approach corridor;
- likely swept volume.

Alternative representations for later ablations are:

1. voxel occupancy;
2. TSDF;
3. ESDF;
4. learned configuration-space distance field.

---

# 25. MotionReady Architecture

A lightweight architecture is:

```text
Local obstacle point cloud
          |
          v
   PointNet-style encoder
          |
    global feature
          |
      z_scene
          |
          +----------------------+
                                 |
Target EE pose -----------------+
Current arm q0 -----------------+
Torso state --------------------+
                                 |
                                 v
                           Query encoder
                                 |
                              z_query
                                 |
                    +------------+
                    |
                    v
              Fusion network
                    |
      +-------------+-------------+-------------+
      |             |             |             |
      v             v             v             v
 p_goal_safe    p_motion    min_clearance    path_cost
```

Formally,

$$
z_{\mathrm{scene}}
=
E_{\mathrm{scene}}
(P_{\mathcal E}),
$$

$$
z_{\mathrm{query}}
=
E_{\mathrm{query}}
(T_E^A,q_0,h),
$$

and

$$
z
=
E_{\mathrm{fusion}}
(
z_{\mathrm{scene}},
z_{\mathrm{query}}
).
$$

Outputs are

$$
[
\hat p_{\mathrm{goal-safe}},
\hat p_{\mathrm{motion}},
\hat c_{\min},
\hat L_\tau
].
$$

Here:

- $\hat c_{\min}$ predicts minimum trajectory clearance;
- $\hat L_\tau$ predicts joint-space path length or trajectory cost.

---

# 26. Recommended MotionReady Network Size

An initial implementation can use:

### Scene encoder

```text
Point MLP:
3 -> 64 -> 128 -> 256

Global max/attention pooling:
256-dimensional scene feature
```

### Query encoder

```text
Target pose + q0:
(9 + nq) -> 128 -> 128
```

### Fusion

```text
Concat(256, 128)
    -> 256
    -> 128
    -> output heads
```

This is still substantially smaller than a vision-language model.

The model operates on geometry already extracted from the environment rather than solving semantics and motion feasibility simultaneously.

---

# 27. MotionReady Teacher Planner

The primary teacher should be a strong exact or optimization-based motion-generation system.

cuRobo provides:

- GPU-accelerated IK;
- robot-world collision checking;
- geometric planning;
- trajectory optimization;
- integrated motion generation.

Its original paper reports a parallelized framework combining collision-free IK, geometric planning, and trajectory optimization [3].

The current `MotionGen` API provides the relevant planning interface and can share world collision representations with the IK solver.

A sample labeling pipeline is:

```python
for scene in scenes:

    world = build_collision_world(scene)

    for start_q in sampled_robot_states:

        for target_pose in sampled_targets:

            ik_result = exact_collision_aware_ik(
                target_pose,
                world,
                seeds=num_ik_seeds,
            )

            goal_safe = ik_result.success

            if goal_safe:
                plan_result = exact_motion_plan(
                    start_q=start_q,
                    target_pose=target_pose,
                    world=world,
                    planning_budget=budget,
                )

                motion_success = verify(plan_result)

            save(
                scene_geometry,
                start_q,
                target_pose,
                goal_safe,
                motion_success,
                trajectory,
                clearance,
                planning_time,
            )
```

---

# 28. MotionReady Loss

Use a multi-task objective:

$$
\begin{aligned}
\mathcal L_{\mathrm{motion}}
=&
\lambda_g
\mathcal L_{\mathrm{BCE}}
(
\hat p_{\mathrm{goal-safe}},
y_{\mathrm{goal-safe}}
)
\\
&+
\lambda_m
\mathcal L_{\mathrm{BCE}}
(
\hat p_{\mathrm{motion}},
y_{\mathrm{motion}}
)
\\
&+
\lambda_c
\operatorname{Huber}
(
\hat c_{\min},
c_{\min}
)
\\
&+
\lambda_l
\operatorname{Huber}
(
\hat L_\tau,L_\tau
)
\\
&+
\lambda_r
\mathcal L_{\mathrm{rank}}
+
\lambda_{\mathrm{cal}}
\mathcal L_{\mathrm{calibration}}.
\end{aligned}
$$

False-positive motion predictions should receive stronger penalty than false negatives.

Useful hard negatives include:

1. reachable target but goal configuration intersects a table;
2. collision-free endpoint but direct path collides;
3. only one IK branch permits a valid motion;
4. narrow passages;
5. target next to cabinet walls;
6. elbow motion obstructed by nearby geometry;
7. incomplete or noisy point clouds.

---

# 29. Motion-Aware Readiness Potential

Define an uncertainty-aware motion score

$$
P_{\mathrm{motion}}^{\mathrm{LCB}}
=
\operatorname{clip}
\left(
\mu_{\mathrm{motion}}
-
\beta_m\sigma_{\mathrm{motion}},
0,1
\right).
$$

Then extend the core readiness potential:

$$
\boxed{
\Phi_B(b)
=
\mathbf 1[C_B=0]
\exp
\left(
\frac{
w_d\log(S_d+\epsilon)
+
w_v\log(S_v+\epsilon)
+
w_k\log(P_{\mathrm{kin}}^{\mathrm{LCB}}+\epsilon)
+
w_m\log(P_{\mathrm{motion}}^{\mathrm{LCB}}+\epsilon)
}{
w_d+w_v+w_k+w_m
}
\right).
}
$$

The RL shaping reward becomes

$$
r_t^{\mathrm{ready}}
=
\gamma\Phi_B(s_{t+1})
-
\Phi_B(s_t).
$$

---

# 30. Hierarchical Evaluation Frequency

KineReady and MotionReady do not need to run at the same frequency.

A practical hierarchy is:

### Every RL step

Compute:

$$
S_d,
\quad
S_v,
\quad
C_B,
\quad
P_{\mathrm{kin}}.
$$

### Only near the manipulation region

Compute:

$$
P_{\mathrm{motion}}.
$$

For example, activate MotionReady when

$$
d<d_{\mathrm{activation}}
$$

and

$$
P_{\mathrm{kin}}>\tau_{\mathrm{kin-pre}}.
$$

### Only at handoff

Run:

- exact IK;
- exact collision checking;
- exact motion planning.

This gives the computational hierarchy

$$
\boxed{
\text{Cheap analytic}
\rightarrow
\text{tiny state model}
\rightarrow
\text{scene-conditioned model}
\rightarrow
\text{exact planner}
}
$$

with progressively increasing fidelity and computational cost.

---

# 31. Optional Extension: Learned Collision Fields

Instead of predicting planner success directly, MotionReady can be extended with a learned collision-distance function.

Define

$$
D_\psi(q,\mathcal E)
\rightarrow
\text{collision clearance}.
$$

A trajectory

$$
\tau
=
(q_0,\dots,q_H)
$$

can then be optimized with objective

$$
\begin{aligned}
J(\tau)
=&
\lambda_T
d_{SE(3)}
(
FK(q_H),T_E
)^2
\\
&+
\lambda_s
\sum_{t=0}^{H-1}
\|q_{t+1}-q_t\|_2^2
\\
&+
\lambda_c
\sum_{t=0}^{H}
\operatorname{softplus}
\left(
m-D_\psi(q_t,\mathcal E)
\right).
\end{aligned}
$$

Configuration-space distance-field methods explicitly represent collision proximity as a function of joint configuration, and CSSDF-Net extends this concept with learned configuration-space signed distances conditioned on arbitrary obstacle point sets [9,10].

DiffCo provides another reference for a differentiable proxy collision function used inside trajectory optimization [8].

This extension is more ambitious than the direct MotionReady classifier and should not be part of the MVP.

---

# 32. Optional Extension: Learned Motion Generation

A further extension is to generate collision-free motions directly.

A learned policy could take

$$
(q_t,T_E^A,P_{\mathcal E})
$$

and output

$$
\Delta q_t.
$$

Motion Policy Networks provide a relevant reference architecture: M$\pi$Nets condition a neural motion policy on point-cloud observations, current robot configuration, and target pose, and were trained on more than three million planning problems across more than 500,000 procedurally generated environments [11].

This variant is conceptually

$$
\pi_{\mathrm{motion}}
(
\Delta q_t
\mid
q_t,T_E^A,\mathcal E
).
$$

However, it should only be investigated after the simpler feasibility-classification variant demonstrates that motion awareness materially improves base placement.

---

# 33. Implementation Stack

## 33.1 Simulation and Data Generation

**Isaac Lab / Isaac Sim**

Recommended for:

- vectorized simulation;
- robot state sampling;
- scene randomization;
- target-pose sampling;
- ground-truth object geometry;
- collision labels.

Isaac Lab's current SkillGen workflow already integrates cuRobo for kinematically feasible, collision-aware motion generation between manipulation segments, providing a useful deployment reference for large-scale planner-supervised data generation [12].

---

## 33.2 IK and Motion Teacher

**cuRobo**

Recommended for:

- batched multi-seed IK;
- collision-aware IK;
- motion planning;
- trajectory optimization;
- offline labels;
- terminal verification.

References: [3,13].

---

## 33.3 ROS 2 Deployment

**Isaac ROS cuMotion**

The current Isaac ROS cuMotion interface exposes:

- a collision-free IK action;
- a native motion-planning action;
- a MoveIt-compatible planning action.

It also supports integration with environment reconstruction for obstacle-aware motion generation [14].

This makes it suitable for the deployment-side exact verification layer.

---

## 33.4 Independent Collision Validation

**MoveIt 2 PlanningScene**

`PlanningScene` provides explicit self-collision and robot-environment collision checking and can therefore act as an independent exact validator [15].

A useful deployment design is:

```text
KineReady / MotionReady
          |
          v
Candidate ranking
          |
          v
cuRobo / cuMotion planning
          |
          v
MoveIt PlanningScene validation
          |
          v
Execution
```

The independent validator is optional but valuable during development.

---

# 34. Suggested Code Organization

```text
kineready/
|
+-- geometry/
|   +-- frame_transforms.py
|   +-- interaction_pose_generator.py
|   +-- pose_noise.py
|
+-- teacher/
|   +-- curobo_ik_teacher.py
|   +-- curobo_motion_teacher.py
|   +-- exact_collision_validator.py
|
+-- data/
|   +-- fk_positive_sampler.py
|   +-- task_space_sampler.py
|   +-- mobile_base_sampler.py
|   +-- boundary_sampler.py
|   +-- motion_dataset_generator.py
|
+-- models/
|   +-- kineready.py
|   +-- kineready_ensemble.py
|   +-- point_scene_encoder.py
|   +-- motionready.py
|
+-- rl/
|   +-- base_metrics.py
|   +-- readiness_potential.py
|   +-- reward_term.py
|   +-- handoff_manager.py
|
+-- deployment/
|   +-- tensorrt_kineready.py
|   +-- ros2_readiness_node.py
|   +-- terminal_verification.py
|
+-- evaluation/
    +-- reachability_metrics.py
    +-- calibration_metrics.py
    +-- base_ranking_metrics.py
    +-- end_to_end_metrics.py
```

---

# 35. Dataset Schema

## 35.1 KineReady Dataset

Each record should contain:

```text
robot_id
target_pose_arm_frame
target_position
target_rotation_6d

optional:
    current_q
    torso_state
    pose_covariance

labels:
    existential_ik
    robust_ik_probability
    local_ik_probability

teacher metadata:
    successful_joint_solutions
    number_of_seeds
    solver_tolerance
    solver_runtime
    joint_limit_margin
    self_collision_status
    label_confidence

sampling metadata:
    scene_id
    object_id
    base_pose
    sampling_strategy
```

---

# 36. MotionReady Dataset

Additional fields are:

```text
scene_geometry
local_point_cloud
current_q
target_pose

goal_collision_free
planner_success
minimum_clearance
joint_path_length
trajectory_duration
planner_runtime

verified_trajectory

planner_metadata:
    planner_name
    planner_version
    planning_budget
    collision_configuration
```

Recording planner version and compute budget is essential because the MotionReady label is planner-dependent.

---

# 37. Training Distribution Strategy

Training should not rely on a purely random split.

The dataset should deliberately include:

### Easy positive

Clearly reachable states.

### Easy negative

Clearly unreachable states.

### Kinematic boundary

States close to reachability boundaries.

### Orientation boundary

Similar target positions with different target orientations.

### Collision counterfactuals

Same target pose and arm state but different obstacle configurations.

### IK-branch counterfactuals

Same target pose but different initial joint configurations.

### Perception perturbation

Same nominal object pose with translation/orientation noise.

### Mobile-base counterfactual

Nearby base poses on opposite sides of the reachability boundary.

These samples explicitly prevent the model from learning trivial distance heuristics.

---

# 38. Critical Baselines

The proposal should compare against both classical and learned alternatives.

## Kinematic Baselines

1. distance only;
2. distance + visibility;
3. hand-designed spherical/cylindrical workspace;
4. exact multi-seed IK;
5. RM4D reachability lookup [2];
6. differentiable reachability MLP [1]-style baseline;
7. KineReady binary;
8. KineReady robust;
9. KineReady robust + uncertainty;
10. original image/VLM critic concept.

RM4D is especially important because it explicitly targets efficient forward and inverse reachability queries and demonstrates mobile-manipulator grasp/base-placement applications [2].

---

# 39. Motion Baselines

1. KineReady only;
2. collision-free endpoint IK;
3. exact cuRobo motion planning;
4. MotionReady classifier;
5. configuration-space distance field [9];
6. CSSDF-Net-style scene-conditioned collision field [10];
7. DiffCo-style learned collision proxy [8];
8. optional M$\pi$Nets-style learned motion policy [11].

---

# 40. Offline Evaluation Metrics

## 40.1 Kinematic Classification

Report:

- AUROC;
- AUPRC;
- precision;
- recall;
- Brier score;
- Expected Calibration Error.

Because false-positive terminal states are particularly costly, report

$$
\operatorname{FPR}@95\%\operatorname{Recall}.
$$

---

# 41. Base-Pose Ranking

For a set of candidates

$$
\mathcal B
=
\{b_1,\dots,b_N\},
$$

measure:

- Spearman correlation;
- Kendall $\tau$;
- top-1 exact feasibility;
- top-$K$ feasibility recall.

Define oracle regret as

$$
\operatorname{Regret}
=
R^\star(b^\star)
-
R^\star(\hat b),
$$

where

$$
b^\star
=
\arg\max_b R^\star(b)
$$

is the exact oracle pose and

$$
\hat b
=
\arg\max_b \hat R(b)
$$

is the learned selection.

---

# 42. MotionReady Metrics

Report:

- goal-safe AUPRC;
- planner-success AUPRC;
- motion-feasibility calibration;
- false-positive planner-success rate;
- minimum-clearance MAE;
- path-cost MAE;
- top-$K$ planning-success recall;
- fraction of exact planner calls eliminated.

A particularly useful metric is:

$$
\operatorname{PlannerRetention}
=
\frac{
\text{exactly feasible states retained after filtering}
}{
\text{all exactly feasible states}
}.
$$

The model should eliminate many infeasible candidates while keeping this quantity close to $1$.

---

# 43. Runtime Evaluation

Inference speed must be measured under the actual RL workload.

Report:

### KineReady

- single-sample latency;
- batch latency;
- samples/second;
- GPU utilization.

### Exact IK

- single-query latency;
- batched latency;
- label-generation throughput.

### MotionReady

- point-cloud encoder latency;
- query-head latency;
- full inference latency.

### RL System

Most importantly:

$$
\boxed{
\text{environment steps per second}
}
$$

and

$$
\boxed{
\text{wall-clock time to convergence}.
}
$$

The project should not assume a learned IK model is necessary merely because an individual CPU IK query is slow. Modern GPU libraries such as cuRobo already parallelize IK heavily [3].

Therefore the first engineering experiment should compare:

$$
T_{\mathrm{exact,batch}}
$$

against

$$
T_{\mathrm{KineReady,batch}}
$$

in the actual simulator configuration.

---

# 44. End-to-End Mobile-Manipulation Evaluation

The central metric should not be classification accuracy.

The important quantity is

$$
\boxed{
P(
S_{\mathrm{manip}}
\mid
S_{\mathrm{nav}}
)
}
$$

— the probability that the downstream manipulation succeeds given that navigation declared success.

Other metrics include:

- navigation success;
- handoff success;
- terminal exact-IK success;
- terminal exact-motion-planning success;
- downstream manipulation success;
- complete-task success;
- base-collision rate;
- arm-collision rate;
- handoff false-positive rate;
- exact-planner calls per episode;
- episode completion time;
- total RL training wall-clock time.

---

# 45. Ablation Studies

The following ablations are particularly important.

## Kinematic Modeling

- XYZ only vs. full $SE(3)$;
- quaternion vs. 6D rotation;
- with vs. without $q_0$;
- binary vs. robust IK;
- one grasp vs. multiple interaction candidates;
- random vs. boundary-focused sampling;
- single network vs. ensemble;
- probability vs. lower-confidence-bound reward.

## Reward Design

- arithmetic sum;
- geometric mean;
- soft minimum;
- direct state reward;
- potential-based shaping.

## MotionReady

- no environment geometry;
- point-cloud geometry;
- voxel geometry;
- endpoint collision only;
- full trajectory label;
- classification only;
- classification + clearance supervision;
- classification + path-cost supervision.

---

# 46. Exact Verification and Safety

Neither learned model should be interpreted as a formal safety system.

The final execution pipeline is:

```text
Navigation RL
     |
     v
KineReady
     |
     v
MotionReady, if enabled
     |
     v
Top-K candidate base poses
     |
     v
Exact collision-aware IK
     |
     v
Exact motion planning
     |
     v
Exact trajectory collision verification
     |
     v
Manipulation execution
```

MoveIt's `PlanningScene` explicitly supports self-collision and environment-collision queries and is suitable for validation [15].

Isaac ROS cuMotion exposes collision-free IK and motion-planning interfaces for ROS 2 deployment [14].

Thus:

$$
\boxed{
\text{Learned feasibility}
\neq
\text{safety guarantee}.
}
$$

---

# 47. Recommended Development Plan

## Phase 0 — Establish Whether a Surrogate Is Needed

Benchmark:

$$
\text{batched exact IK}
\quad\text{vs.}\quad
\text{MLP inference}.
$$

Measure the fraction of RL wall-clock spent on IK.

**Go criterion:** IK is a meaningful training bottleneck.

---

## Phase 1 — Classical Reachability Baselines

Implement:

1. exact multi-seed IK;
2. simple workspace bounds;
3. RM4D or an equivalent reachability map;
4. simple binary MLP.

Goal:

> Determine how complex the reachability boundary actually is for the target robot.

---

## Phase 2 — Robust KineReady

Add:

- target-pose perturbation;
- multi-seed labels;
- multiple interaction candidates;
- boundary sampling;
- calibration;
- uncertainty.

Goal:

$$
T_E^A
\rightarrow
P_{\mathrm{robust\ IK}}.
$$

---

## Phase 3 — RL Integration

Integrate:

- exact distance;
- exact visibility;
- exact base collision;
- learned robust IK.

Train navigation using

$$
r_t^{\mathrm{ready}}
=
\gamma\Phi_A(s_{t+1})
-
\Phi_A(s_t).
$$

Evaluate terminal manipulation feasibility.

---

## Phase 4 — MotionReady

Generate planner-supervised data:

$$
(T_E^A,q_0,\mathcal E)
\rightarrow
Y_{\mathrm{motion}}.
$$

Train the scene-conditioned feasibility model.

Evaluate whether motion-awareness improves

$$
P(
\text{exact planner success}
\mid
\text{navigation handoff}
).
$$

---

## Phase 5 — Optional Learned Motion Generation

Only if Phase 4 establishes that trajectory feasibility is a major bottleneck:

- learn configuration-space distance fields;
- distill expert planner trajectories;
- investigate reactive learned motion policies.

---

# 48. Expected Contributions

The core scientific contribution should **not** be framed simply as:

> “We use an MLP to approximate IK.”

Learned and precomputed reachability representations already exist [1,2].

Instead, the contribution can be framed around the **navigation-to-manipulation interface**:

### Contribution 1 — Robust Manipulation-Oriented Reachability

A probabilistic reachability representation that accounts for target-pose uncertainty and multiple valid interaction poses:

$$
P_{\mathrm{robust\ IK}}
(
T_E^A
).
$$

### Contribution 2 — Manipulation-Aware Navigation Reward

A factorized reward combining exact inexpensive metrics with learned expensive feasibility:

$$
\boxed{
\Phi
=
G
(
S_d^{\mathrm{exact}},
S_v^{\mathrm{exact}},
C_B^{\mathrm{exact}},
P_{\mathrm{IK}}^{\mathrm{learned}}
).
}
$$

### Contribution 3 — Semantic/Kinematic Factorization

A demonstration that open-vocabulary semantic perception need not remain in the high-frequency reward loop once the manipulation target has been grounded into geometric state.

### Contribution 4 — Scene-Conditioned Motion Feasibility

A hierarchical extension from

$$
\text{reachable}
$$

to

$$
\text{executable through collision-free motion}.
$$

### Contribution 5 — Uncertainty-Aware Filtering with Exact Verification

A deployment architecture in which learned feasibility reduces expensive queries while classical planning remains responsible for terminal verification.

### Contribution 6 — End-to-End Evaluation

Evaluation not merely on IK classification accuracy but on:

$$
P(
S_{\mathrm{manip}}
\mid
S_{\mathrm{nav}}
),
$$

RL training throughput, exact-planner query count, and final long-horizon task success.

---

# 49. Proposed Paper Positioning

A concise positioning statement is:

> Existing mobile-navigation objectives generally optimize geometric proximity or visibility, while conventional manipulation feasibility requires comparatively expensive kinematic and motion-planning queries. We investigate whether manipulation feasibility can be distilled into lightweight state-space models that are inexpensive enough for dense reinforcement-learning rewards. By separating semantic grounding from robot kinematics, we learn a robust task-space reachability field for high-frequency navigation shaping and extend it with scene-conditioned collision-free motion feasibility. Exact geometric planning remains a sparse terminal verifier rather than a dense reward evaluator.

Potential paper titles include:

> **KineReady: State-Based Robust Reachability Rewards for Navigation-to-Manipulation Handoff**

or, for the complete system:

> **From Reachable to Executable: Learning Kinematic and Motion Feasibility for Mobile-Manipulation Base Placement**

---

# 50. References and Deployment Resources

**[1] Murooka, M., Kumagai, I., Morisawa, M., and Kanehiro, F.**  
*Learning Differentiable Reachability Maps for Optimization-based Humanoid Motion Generation.* 2025. arXiv:2508.11275.  
Introduces neural/SVM task-space differentiable reachability maps learned from robot kinematics and applies them to motion-generation problems.

**[2] Rudorfer, M.**  
*RM4D: A Combined Reachability and Inverse Reachability Map for Common 6-/7-axis Robot Arms by Dimensionality Reduction to 4D.* 2024. arXiv:2410.06968.  
Relevant classical baseline for efficient reachability queries and mobile-manipulator base-placement applications.

**[3] Sundaralingam, B. et al.**  
*cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation.* 2023. arXiv:2310.17274.  
GPU-accelerated IK, collision checking, geometric planning, and trajectory optimization; recommended as an offline teacher and exact terminal planner.

**[4] Zhou, Y., Barnes, C., Lu, J., Yang, J., and Li, H.**  
*On the Continuity of Rotation Representations in Neural Networks.* CVPR 2019.  
Reference for continuous 6D rotation representations.

**[5] Ng, A. Y., Harada, D., and Russell, S.**  
*Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML 1999.  
Foundation for potential-based reward shaping.

**[6] Xu, L., Ren, T., Chalvatzaki, G., and Peters, J.**  
*Accelerating Integrated Task and Motion Planning with Neural Feasibility Checking.* 2022. arXiv:2203.10568.  
Relevant precedent for learning inexpensive feasibility classifiers from simulated IK/planning supervision.

**[7] Das, N. and Yip, M.**  
*Learning-Based Proxy Collision Detection for Robot Motion Planning Applications (Fastron).* 2019. arXiv:1902.08164.  
Reference for learned configuration-space proxy collision checking.

**[8] Zhi, Y., Das, N., and Yip, M.**  
*DiffCo: Auto-Differentiable Proxy Collision Detection with Multi-class Labels for Safety-Aware Trajectory Optimization.* 2021. arXiv:2102.07413.  
Differentiable learned collision proxy designed for trajectory optimization.

**[9] Li, Y., Chi, X., Razmjoo, A., and Calinon, S.**  
*Configuration Space Distance Fields for Manipulation Planning.* 2024. arXiv:2406.01137.  
Represents collision distance directly in robot configuration space and includes a neural representation.

**[10] Chen, H., Zhou, Y., Zhou, Y., and Wang, H.**  
*CSSDF-Net: Safe Motion Planning Based on Neural Implicit Representations of Configuration Space Distance Field.* 2026. arXiv:2603.18669.  
Scene-conditioned configuration-space signed-distance learning for self- and external-collision-aware motion planning.

**[11] Fishman, A., Murali, A., Eppner, C., Peele, B., Boots, B., and Fox, D.**  
*Motion Policy Networks.* CoRL 2022 / PMLR 2023.  
Reference architecture for learned collision-free motion generation conditioned on point clouds, current configuration, and target pose.

**[12] NVIDIA Isaac Lab.**  
*SkillGen for Automated Demonstration Generation.*  
Current Isaac Lab workflow integrating cuRobo-based collision-aware motion planning into large-scale demonstration generation; useful implementation reference for planner-supervised data collection.

**[13] cuRobo Documentation.**  
`IKSolver` and `MotionGen` APIs.  
Deployment reference for batched IK, goal-set planning, world collision representations, and motion generation.

**[14] NVIDIA Isaac ROS.**  
*Isaac ROS cuMotion.*  
ROS 2 deployment interface exposing collision-free IK, native motion planning, and MoveIt-compatible planning actions; supports environment-aware planning workflows.

**[15] MoveIt 2.**  
*PlanningScene Documentation.*  
Reference implementation for exact self-collision and robot-environment collision checking and constraint validation.