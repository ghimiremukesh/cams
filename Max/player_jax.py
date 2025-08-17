# player_jax.py
# =========================================================
# Flax/JAX implementations of:
#   • CAMSInformed (Player-1 mixed strategy with step-specific heads)
#   • BR          (Player-2 deterministic best response per step)
#
# No pruning logic. Identity logits enforced on the last layer.
# API parity with PyTorch:
#   out = p1.apply(params, obs, k) -> {"A_logits": (B,I,I), "μ": (B,I,d)}
#   u, misc = p1_action_only(p1, params, obs, i_star, k, temperature=1.0)
#   u, misc = p2_action_only(br, params, obs, k)
# =========================================================

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


# =========================================================
# Global switches
# =========================================================
F32 = jnp.float32
DEFAULT_TEMP = 1.0   # for action_only() softmax temperature in misc


# =========================================================
# Feature helper (matches game.belief_coord contract)
#   obs: {"x": (B,STATE_DIM), "p": (B,I)}
#   feat = concat(x, p[..., :I-1])
# =========================================================
def _features_from_obs(obs: Dict[str, jnp.ndarray], I: int) -> jnp.ndarray:
    x = obs["x"]
    p = obs["p"][..., : max(I - 1, 0)]
    return jnp.concatenate([x, p], axis=-1)


# =========================================================
# CAMS step net
#   If last_layer=True, returns identity logits (I×I) per batch.
#   Always returns a full μ-table with I prototypes (B,I,d) scaled to BOX_ACC.
# =========================================================
class _CAMSNet(nn.Module):
    I: int
    d: int
    feat_dim: int
    box_acc: float
    hidden: int
    last_layer: bool = False

    @nn.compact
    def __call__(self, feats: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # feats: (B, feat_dim)
        h = nn.relu(nn.Dense(self.hidden)(feats))
        h = nn.relu(nn.Dense(self.hidden)(h))

        # μ prototypes (B, I, d)
        mu_flat = nn.Dense(self.I * self.d)(h)               # (B, I*d)
        mu_tbl  = mu_flat.reshape(feats.shape[0], self.I, self.d)
        mu_tbl  = jnp.tanh(mu_tbl) * jnp.asarray(self.box_acc, dtype=F32)

        if self.last_layer:
            # fixed identity logits
            eye_logits = _identity_logits(self.I)             # (I,I)
            A_logits = jnp.broadcast_to(eye_logits, (feats.shape[0], self.I, self.I))
        else:
            # learned logits (B, I, I)
            Alog_flat = nn.Dense(self.I * self.I)(h)          # (B, I*I)
            A_logits  = Alog_flat.reshape(feats.shape[0], self.I, self.I)

        return {"A_logits": A_logits, "μ": mu_tbl}


def _identity_logits(I: int) -> jnp.ndarray:
    """Large +diag/−offdiag logits so softmax ≈ identity."""
    off = jnp.full((I, I), -50.0, dtype=F32)
    return off.at[jnp.arange(I), jnp.arange(I)].set(50.0)


# =========================================================
# Player-1: CAMSInformed wrapper with K step-specific heads
#   k=0 uses explicit root params (logits + μ table).
#   k=1..K-1 use small nets; last step uses identity logits.
# =========================================================
class CAMSInformed(nn.Module):
    I: int
    d: int
    feat_dim: int
    K: int
    box_acc: float
    hidden: int = 32

    def setup(self):
        # Root parameters
        self.root_logits = self.param(
            "root_logits", nn.initializers.normal(stddev=1e-2), (self.I, self.I)
        )
        self.root_mu = self.param(
            "root_mu", nn.initializers.normal(stddev=1e-2), (self.I, self.d)
        )

        # Step nets for k = 1 .. K-1
        self.subnets = []
        for k in range(1, self.K):
            self.subnets.append(
                _CAMSNet(
                    I=self.I,
                    d=self.d,
                    feat_dim=self.feat_dim,
                    box_acc=self.box_acc,
                    hidden=self.hidden,
                    last_layer=(k == self.K - 1),
                    name=f"step{k}",
                )
            )

    def __call__(self, obs: Dict[str, jnp.ndarray], k: int) -> Dict[str, jnp.ndarray]:
        """
        Returns:
            {"A_logits": (B,I,I), "μ": (B,I,d)}
        """
        assert 0 <= k < self.K, "k out of range"
        feats = _features_from_obs(obs, self.I)  # (B, feat_dim)

        if k == 0:
            B = feats.shape[0]
            A = jnp.broadcast_to(self.root_logits[None, ...], (B, self.I, self.I))
            mu_tbl = jnp.broadcast_to(self.root_mu[None, ...], (B, self.I, self.d))
            # scale μ to box_acc and squash
            mu_tbl = jnp.tanh(mu_tbl) * jnp.asarray(self.box_acc, dtype=F32)
            return {"A_logits": A, "μ": mu_tbl}

        # k >= 1
        return self.subnets[k - 1](feats)


# =========================================================
# Player-2: Deterministic BR with explicit root μ and small nets later
# =========================================================
class _BRNet(nn.Module):
    feat_dim: int
    d: int
    box_acc: float
    hidden: int

    @nn.compact
    def __call__(self, feats: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.hidden)(feats))
        h = nn.relu(nn.Dense(self.hidden)(h))
        u = nn.tanh(nn.Dense(self.d)(h)) * jnp.asarray(self.box_acc, dtype=F32)
        return u  # (B, d)


class BR(nn.Module):
    d: int
    feat_dim: int
    K: int
    box_acc: float
    hidden: int = 32

    def setup(self):
        self.root_mu = self.param("root_mu", nn.initializers.normal(stddev=1e-2), (self.d,))
        self.subnets = []
        for k in range(1, self.K):
            self.subnets.append(
                _BRNet(
                    feat_dim=self.feat_dim,
                    d=self.d,
                    box_acc=self.box_acc,
                    hidden=self.hidden,
                    name=f"step{k}",
                )
            )

    def __call__(self, obs: Dict[str, jnp.ndarray], k: int) -> jnp.ndarray:
        feats = jnp.concatenate([obs["x"], obs["p"][..., :-1] if obs["p"].shape[-1] > 1 else obs["p"][..., :0]], axis=-1)
        if k == 0:
            B = feats.shape[0]
            mu = jnp.broadcast_to(self.root_mu[None, :], (B, self.d))
            mu = jnp.tanh(mu) * jnp.asarray(self.box_acc, dtype=F32)
            return mu
        return self.subnets[k - 1](feats)


# =========================================================
# Action-only helpers (API parity with PyTorch)
#   - p1_action_only returns u and misc dict with:
#       "A": softmax(A_logits / T), "row": row softmax, "j": indices, "μ": μ-table
#   - p2_action_only returns u and {"μ": u}
# =========================================================
def p1_forward(model: CAMSInformed, params, obs: Dict[str, jnp.ndarray], k: int) -> Dict[str, jnp.ndarray]:
    return model.apply(params, obs, k)


def p1_action_only(
    model: CAMSInformed,
    params,
    obs: Dict[str, jnp.ndarray],
    i_star: jnp.ndarray,   # (B,) int
    k: int,
    temperature: float = DEFAULT_TEMP,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    out = model.apply(params, obs, k)  # {"A_logits": (B,I,I), "μ": (B,I,d)}
    A_logits = out["A_logits"]
    mu_tbl   = out["μ"]

    B, I, _ = A_logits.shape
    temp = jnp.asarray(temperature, dtype=F32)

    # Row for the true type i★
    row_logits = A_logits[jnp.arange(B), i_star]              # (B, I)
    j = jnp.argmax(row_logits, axis=-1)                       # (B,)

    u = mu_tbl[jnp.arange(B), j]                              # (B, d)

    A_prob = jax.nn.softmax(A_logits / temp, axis=-1)         # (B, I, I)
    row    = jax.nn.softmax(row_logits / temp, axis=-1)       # (B, I)

    misc = {
        "A":   A_prob,
        "row": row,
        "j":   j,
        "μ":   mu_tbl,
    }
    return u, misc


def p2_action_only(
    model: BR,
    params,
    obs: Dict[str, jnp.ndarray],
    k: int,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    u = model.apply(params, obs, k)  # (B, d)
    return u, {"μ": u}