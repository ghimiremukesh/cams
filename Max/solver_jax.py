# dsgda_solver_jax.py
# =========================================================
# Clean DSGDA solver (JAX) for the football game.
# - No pruning, no tree visualisation.
# - Exact expectation over all public-message sequences (I^K).
# - Separate optimisers for P1 (descent) and P2 (ascent).
# - JSONL logging + periodic checkpoints compatible with your run layout.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import os, json, time, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import serialization as flax_serial

from game_jax import FootballGame, FootballState
from player_jax import CAMSInformed, BR


# =========================================================
# Global switches
# =========================================================
F32 = jnp.float32
PROFILE_TIMES = True     # set False to skip timing fields (kept for parity)
DEBUG_INFO    = False    # print shapes and sanity stats


# =========================================================
# Helpers
# =========================================================
def _all_paths(I: int, K: int) -> jnp.ndarray:
    """Return tensor of shape (I**K, K) with base-I digit rows."""
    S = I ** K
    idx = jnp.arange(S, dtype=jnp.int32)
    paths = []
    for k in range(K - 1, -1, -1):
        paths.append((idx // (I ** k)) % I)
    return jnp.stack(paths, axis=1).astype(jnp.int32)  # (K, S) → (S, K) after .T
    # NOTE: we stacked from highest power to lowest; transpose for row-major
    # (keeping as (K,S) then transpose for clarity)
def _all_paths(I: int, K: int) -> jnp.ndarray:
    S = I ** K
    idx = jnp.arange(S, dtype=jnp.int32)
    cols = []
    for power in range(K - 1, -1, -1):
        cols.append(((idx // (I ** power)) % I))
    return jnp.stack(cols, axis=1).astype(jnp.int32)   # (S, K)


def _global_grad_norm(grads) -> float:
    sq = 0.0
    for g in jax.tree_util.tree_leaves(grads):
        sq = sq + jnp.sum(jnp.asarray(g, dtype=F32) ** 2)
    return float(jnp.sqrt(sq))


# =========================================================
# Loss factory
# =========================================================
def make_loss_fn(
    game: FootballGame,
    p1_model: CAMSInformed,
    p2_model: BR,
    paths: jnp.ndarray,               # (S, K) int32
):
    I, K = game.I, game.K
    S = paths.shape[0]
    prior = jnp.asarray(game.P0[0], dtype=F32)  # (I,)

    def loss_fn(p1_params, p2_params) -> jnp.ndarray:
        # ----- reset env batch = S sequences --------------------------
        state0, _ = game.reset(batch_size=S)      # PRNG not needed; deterministic lineup
        state = FootballState(
            x=state0.x, t=state0.t, p=state0.p,
            i_star=jnp.zeros((S,), dtype=jnp.int32),
            w_last=state0.w_last
        )

        # path-type probabilities π(seq|i) for all i
        pi = jnp.ones((S, I), dtype=F32)
        running = jnp.zeros((S,), dtype=F32)

        # roll over K steps
        for k in range(K):
            obs = {"x": state.x, "p": state.p, "t": state.t}

            out = p1_model.apply(p1_params, obs, k)           # {"A_logits": (S,I,I), "μ": (S,I,d)}
            A_logits = out["A_logits"]
            mu_tbl   = out["μ"]                               # (S,I,d)

            # soft policy over columns
            A_soft = jax.nn.softmax(A_logits, axis=-1)        # (S,I,I)

            # chosen column per sequence
            j_k = paths[:, k]                                 # (S,)
            j_idx = j_k.reshape(S, 1, 1)
            Aj = jnp.take_along_axis(A_soft, jnp.tile(j_idx, (1, I, 1)), axis=2).squeeze(-1)  # (S,I)
            pi = pi * Aj

            # type-agnostic prototype action for offence
            u1 = mu_tbl[jnp.arange(S), j_k]                   # (S,d)

            # defence action
            u2 = p2_model.apply(p2_params, obs, k)            # (S,d)

            # physics step
            state, p_tackle_now = game.step(state, u1, u2)
            running = running + game._running_loss(u1, u2, p_tackle_now)  # (S,)

            # public belief Bayes update
            state = FootballState(
                x=state.x, t=state.t,
                p=game._bayes_update(state.p, A_soft, j_k),
                i_star=state.i_star, w_last=state.w_last
            )

        # terminal part depends on i★ only through the terminal loss
        total = 0.0
        for i_star in range(I):
            state_i = FootballState(x=state.x, t=state.t, p=state.p,
                                    i_star=jnp.full((S,), i_star, dtype=jnp.int32),
                                    w_last=state.w_last)
            L_term = game._terminal_loss(state_i)             # (S,)
            L_tot  = running + L_term                         # (S,)
            total += prior[i_star] * jnp.sum(pi[:, i_star] * L_tot)

        # scalar
        return total

    # JIT for speed; models captured as static Python objects in closure
    return jax.jit(loss_fn)


# =========================================================
# Solver
# =========================================================
@dataclass
class DSGDASpec:
    lr_p1: float = 3e-3
    lr_p2: float = 1e-2
    momentum: float = 0.6
    C2_p1: float = 10.0   # used as global-norm clip (via sqrt like PyTorch buffer clip)
    C2_p2: float = 10.0


class DSGDASolver:
    """
    JAX DSGDA with separate optax pipelines for P1 (descent) and P2 (ascent).
    Training-only; viz is handled externally on CPU.
    """
    def __init__(
        self,
        game: FootballGame,
        p1_model: CAMSInformed, p1_params,
        p2_model: BR,           p2_params,
        spec: Dict[str, Any],
        *,
        log_root: str = "Max/runs",
    ):
        self.game = game
        self.p1_model, self.p2_model = p1_model, p2_model
        self.p1_params, self.p2_params = p1_params, p2_params

        self.spec = DSGDASpec(**spec)
        self.I, self.K = game.I, game.K

        # Full message grid (I^K × K)
        self.paths = _all_paths(self.I, self.K)   # (S,K)
        self.S_paths = int(self.paths.shape[0])

        # Directories
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.stamp    = stamp
        self.run_dir  = Path(log_root).expanduser() / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "ckpt"
        self.anim_dir = self.run_dir / "anim"   # kept for parity; viz code may write here
        self.ckpt_dir.mkdir(exist_ok=True)
        self.anim_dir.mkdir(exist_ok=True)
        self.log_path = str((self.run_dir / "log.jsonl").resolve())

        self.meta: List[Dict[str, float]] = []

        # Optax transforms
        c1 = float(self.spec.C2_p1) ** 0.5
        c2 = float(self.spec.C2_p2) ** 0.5
        self.opt_p1 = optax.chain(
            optax.clip_by_global_norm(c1),
            optax.sgd(learning_rate=self.spec.lr_p1, momentum=self.spec.momentum, nesterov=False),
        )
        self.opt_p2 = optax.chain(
            optax.clip_by_global_norm(c2),
            optax.sgd(learning_rate=self.spec.lr_p2, momentum=self.spec.momentum, nesterov=False),
        )
        self.opt_state_p1 = self.opt_p1.init(self.p1_params)
        self.opt_state_p2 = self.opt_p2.init(self.p2_params)

        # JIT loss
        self._loss = make_loss_fn(self.game, self.p1_model, self.p2_model, self.paths)

        # Value+grads function
        self._vg = jax.jit(jax.value_and_grad(self._loss, argnums=(0, 1)))

    # -----------------------------------------------------
    def step(self) -> Dict[str, float]:
        if PROFILE_TIMES:
            t0 = time.perf_counter()

        # loss and grads
        loss, (gr_p1, gr_p2) = self._vg(self.p1_params, self.p2_params)

        if PROFILE_TIMES:
            t1 = time.perf_counter()

        # gradient norms (for logging)
        g1 = _global_grad_norm(gr_p1)
        g2 = _global_grad_norm(gr_p2)

        # Apply updates (P1 descent, P2 ascent → minimise -loss)
        updates_p1, self.opt_state_p1 = self.opt_p1.update(gr_p1, self.opt_state_p1, self.p1_params)
        self.p1_params = optax.apply_updates(self.p1_params, updates_p1)

        updates_p2, self.opt_state_p2 = self.opt_p2.update(
            jax.tree_map(lambda g: -g, gr_p2), self.opt_state_p2, self.p2_params
        )
        self.p2_params = optax.apply_updates(self.p2_params, updates_p2)

        if PROFILE_TIMES:
            t2 = time.perf_counter()

        rec = {
            "iter"      : len(self.meta),
            "L"         : float(loss),
            "g_p1"      : float(g1),
            "g_p2"      : float(g2),
            "n_seq"     : int(self.S_paths),
            # Timing (kept keys for plot_run compatibility)
            "t_prune"    : 0.0,                                 # no pruning
            "t_loss"     : (t1 - t0) * 1e3 if PROFILE_TIMES else 0.0,
            "t_backward" : 0.0,                                 # fused in JAX
            "t_momentum" : (t2 - t1) * 1e3 if PROFILE_TIMES else 0.0,  # includes param updates
            "t_step"     : 0.0,
            "wall_ms"    : (t2 - t0) * 1e3 if PROFILE_TIMES else 0.0,
        }

        self.meta.append(rec)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

        if DEBUG_INFO and (rec["iter"] % 50 == 0):
            print(f"[{rec['iter']:04d}] L={rec['L']:+.4f}  g1={rec['g_p1']:.3f}  g2={rec['g_p2']:.3f}  S={rec['n_seq']}")

        return rec

    # -----------------------------------------------------
    def save_checkpoint(self, tag: str | int):
        payload = {
            "iter"        : int(tag) if isinstance(tag, int) else tag,
            "spec"        : dict(self.spec.__dict__),
            "game_spec"   : self.game.spec,     # plain Python types + ndarray ok
            "p1_params"   : flax_serial.to_state_dict(self.p1_params),
            "p2_params"   : flax_serial.to_state_dict(self.p2_params),
            "opt_p1"      : flax_serial.to_state_dict(self.opt_state_p1),
            "opt_p2"      : flax_serial.to_state_dict(self.opt_state_p2),
            "meta_log"    : self.meta,
            "paths"       : jnp.asarray(self.paths).astype(jnp.int32).tolist(),
        }
        fname = (self.ckpt_dir / f"ckpt_{tag}.json").resolve()
        with open(fname, "w") as fh:
            json.dump(payload, fh)
        return str(fname)