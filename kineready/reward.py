"""Inference API: score candidate base poses without a simulator.

This is the module a navigation policy actually calls. It imports numpy and torch and nothing
else -- no OmniGibson, no Isaac, no CuRobo -- which is the point of having trained a surrogate at
all. The exact IK term it replaces costs 0.34 s per pose and needs a live simulator; this costs
microseconds per pose and needs a 55k-parameter file.

The layering matters. The MODEL answers one question: given an eef target expressed in the base
frame, does an IK solution exist? It knows nothing about objects, tasks, or scenes. THIS module
composes that answer over the N candidate base poses x K target poses an upstream planner
supplies, and aggregates the K into one number per base pose. Anything task-specific -- how the K
grasp candidates were generated, what object they belong to -- lives in `examples/`, above both.
"""
import numpy as np
import torch as th

from .frames import encode_features, noisy_or, targets_in_base_frame


class ReadinessReward:
    """P_kin(b): probability that at least one supplied target is reachable from base pose b."""

    def __init__(self, model, device=None, beta=1.0, use_lcb=True):
        self.device = device or ("cuda" if th.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.beta = float(beta)
        # A false positive costs a wasted navigation episode; a false negative costs one skipped
        # candidate out of many. The asymmetry is real, so the default is the conservative bound.
        self.use_lcb = bool(use_lcb)

    @classmethod
    def from_checkpoint(cls, path, **kw):
        from .model import KineReadyEnsemble

        ckpt = th.load(path, map_location="cpu", weights_only=False)
        model = KineReadyEnsemble(m=ckpt.get("m", 5))
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, **kw)

    @th.no_grad()
    def score(self, base_poses, targets_world, priors=None, return_components=False):
        """(N, 3) base poses (x, y, yaw) x (K, 4, 4) world targets -> (N,) in [0, 1].

        `priors` optionally weights the K candidates (proposal Sec 11) -- e.g. a grasp sampler's
        own confidence -- so a marginal candidate cannot claim as much credit as a good one.
        """
        b = np.atleast_2d(np.asarray(base_poses, dtype=float))
        T = np.asarray(targets_world, dtype=float)
        if T.ndim == 2:
            T = T[None]

        local = targets_in_base_frame(b, T)                  # (N, K, 4, 4)
        feats = encode_features(local).reshape(-1, 9)        # (N*K, 9)
        out = self.model.predict(th.tensor(feats, dtype=th.float32, device=self.device),
                                 beta=self.beta)
        key = "p_lcb" if self.use_lcb else "p_exist"
        p = out[key].cpu().numpy().reshape(len(b), len(T))

        agg = noisy_or(p, priors=priors, axis=-1)
        if return_components:
            return agg, {"p_per_target": p,
                         "p_std": out["p_std"].cpu().numpy().reshape(len(b), len(T)),
                         "robust": out["robust"].cpu().numpy().reshape(len(b), len(T))}
        return agg

    def best_pose(self, base_poses, targets_world, **kw):
        """-> (index, score) of the highest-scoring candidate. The endpoint-selection primitive."""
        s = self.score(base_poses, targets_world, **kw)
        i = int(np.argmax(s))
        return i, float(s[i])


def potential_shaping(phi_next, phi_now, gamma=0.99):
    """r = gamma*Phi(s') - Phi(s)  (proposal Sec 19).

    Ng et al.'s potential-based form: adding this to a reward provably leaves the optimal policy
    unchanged, so a miscalibrated readiness estimate can slow learning but cannot teach the policy
    to prefer a worse endpoint. That guarantee is why the reward enters this way and not as a raw
    bonus.
    """
    return gamma * np.asarray(phi_next, dtype=float) - np.asarray(phi_now, dtype=float)


def readiness(distance_score, visibility_score, collision_free, p_kin, weights=None):
    """Geometric mean of the four components (proposal Sec 18) -> Phi in [0, 1].

    Geometric, not arithmetic: a base pose that cannot see the object is not redeemed by being
    well-placed for IK, and an arithmetic mean would let three good terms outvote one fatal one.
    Collision-free enters as a hard 0/1 gate for the same reason.
    """
    w = weights or {"distance": 1.0, "visibility": 1.0, "kinematic": 2.0}
    d = np.asarray(distance_score, dtype=float)
    v = np.asarray(visibility_score, dtype=float)
    k = np.asarray(p_kin, dtype=float)
    c = np.asarray(collision_free, dtype=float)

    eps = 1e-6
    total = w["distance"] + w["visibility"] + w["kinematic"]
    logs = (w["distance"] * np.log(np.clip(d, eps, 1.0))
            + w["visibility"] * np.log(np.clip(v, eps, 1.0))
            + w["kinematic"] * np.log(np.clip(k, eps, 1.0)))
    return c * np.exp(logs / total)
