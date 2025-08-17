# game_jax.py
# =========================================================
# JAX environment for the 2-team American-football running-play game.
# Training is JAX/functional; visualisation remains CPU/Matplotlib.
# No tree pruning, no tree visualisation.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import jax
import jax.numpy as jnp


# =========================================================
# Global switches (tune here)
# =========================================================
F32 = jnp.float32
DEBUG_CHECKS = False           # set True for shape/NaN checks during dev
DEFAULT_PRNG_KEY = jax.random.PRNGKey(1234)


# =========================================================
# Spec (same fields/names as your PyTorch version)
# =========================================================
def default_football_spec(
    N: int = 11,
    horizon: float = 2.0,
    dt: float = 0.05,
    device: str = "cpu",  # kept for parity; JAX places arrays automatically
    *,
    box_pos: float = 1.6,
    box_vel: float = 5.0,
    box_acc: float = 3.0,
    merge_radius: float = 0.15,
    lineup_off_x: float = -1.2,
    lineup_def_x: float = -0.8,
    n_substeps: int = 4,
) -> Dict[str, Any]:
    twoN = 2 * N
    eye = np.eye(twoN, dtype=np.float32)

    spec: Dict[str, Any] = dict(
        tau=dt,
        T=horizon,
        device=device,
        N_PLAYERS=N,
        n_types=2,
        BOX_POS=box_pos,
        BOX_VEL=box_vel,
        BOX_ACC=box_acc,
        R1=eye,
        R2=eye,
        MERGE_RADIUS=merge_radius,
        LINEUP_OFF_X=lineup_off_x,
        LINEUP_DEF_X=lineup_def_x,
        TACKLE_PENALTY=5.0,
        RB_DEPTH=0.25,
        n_substeps=n_substeps,
        P_OFFSETS=np.array([[-0.8], [+0.8]], dtype=np.float32),
    )

    if N == 11:
        off_x = lineup_off_x
        def_x = lineup_def_x

        y_ol = np.array([-0.80, -0.40, 0.00, 0.40, 0.80], dtype=np.float32)
        te_y = np.float32(1.10)
        wrL_y = np.float32(-1.45)
        wrR_y = np.float32(+1.45)
        QB_x = np.float32(off_x - 0.20)
        FB_x = np.float32(off_x - 0.30)
        RB_x = np.float32(off_x - 0.40)

        off_pos = np.stack(
            [
                np.array([off_x, y_ol[0]], np.float32),  # LT
                np.array([off_x, y_ol[1]], np.float32),  # LG
                np.array([off_x, y_ol[2]], np.float32),  # C
                np.array([off_x, y_ol[3]], np.float32),  # RG
                np.array([off_x, y_ol[4]], np.float32),  # RT
                np.array([off_x, te_y], np.float32),     # TE (right)
                np.array([off_x, wrL_y], np.float32),    # WR-L
                np.array([off_x, wrR_y], np.float32),    # WR-R
                np.array([QB_x, 0.0], np.float32),       # QB
                np.array([FB_x, 0.20], np.float32),      # FB
                np.array([RB_x, 0.00], np.float32),      # RB (ball)
            ],
            axis=0,
        )

        y_dl = np.array([-0.60, -0.20, 0.20, 0.60], dtype=np.float32)
        y_lb = np.array([-0.80, 0.00, 0.80], dtype=np.float32)
        CB_yL, CB_yR = -1.45, 1.45
        FS_y, SS_y = -0.90, 0.90

        DL_x = def_x
        LB_x = def_x - 0.15
        CB_x = def_x + 0.05
        S_x = def_x - 0.45

        def_pos = np.stack(
            [
                np.array([DL_x, y_dl[0]], np.float32),  # LDE
                np.array([DL_x, y_dl[1]], np.float32),  # LDT
                np.array([DL_x, y_dl[2]], np.float32),  # RDT
                np.array([DL_x, y_dl[3]], np.float32),  # RDE
                np.array([LB_x, y_lb[0]], np.float32),  # SLB
                np.array([LB_x, y_lb[1]], np.float32),  # MLB
                np.array([LB_x, y_lb[2]], np.float32),  # WLB
                np.array([CB_x, CB_yL], np.float32),    # CB-L
                np.array([CB_x, CB_yR], np.float32),    # CB-R
                np.array([S_x, FS_y], np.float32),      # FS
                np.array([S_x, SS_y], np.float32),      # SS
            ],
            axis=0,
        )

        # Soft bounds via clipping
        off_pos[:, 0] = np.clip(off_pos[:, 0], -box_pos, box_pos)
        off_pos[:, 1] = np.clip(off_pos[:, 1], -box_pos, box_pos)
        def_pos[:, 0] = np.clip(def_pos[:, 0], -box_pos, box_pos)
        def_pos[:, 1] = np.clip(def_pos[:, 1], -box_pos, box_pos)

        spec["OFF_POS"] = off_pos
        spec["DEF_POS"] = def_pos
        spec["RB_INDEX"] = 10
        spec["PLAY_ROLES"] = {
            "offence": ["LT","LG","C","RG","RT","TE","WR-L","WR-R","QB","FB","RB"],
            "defence": ["LDE","LDT","RDT","RDE","SLB","MLB","WLB","CB-L","CB-R","FS","SS"],
            "notes": "I-formation (21 personnel) vs 4-3 base",
        }

    return spec


# =========================================================
# Env state (pure PyTree)
# =========================================================
@jax.tree_util.register_pytree_node_class
@dataclass
class FootballState:
    x: jnp.ndarray          # (B, STATE_DIM)
    t: jnp.ndarray          # (B,)
    p: jnp.ndarray          # (B, I)
    i_star: jnp.ndarray     # (B,) int32
    # Optional cached merge weights for debug/viz
    w_last: jnp.ndarray     # (B, N, N)

    def tree_flatten(self):
        children = (self.x, self.t, self.p, self.i_star, self.w_last)
        aux = {}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# =========================================================
# Game class: physics + helpers (no models/solver here)
# =========================================================
class FootballGame:
    def __init__(self, spec: Dict[str, Any], batch_size: int = 1):
        self.spec = spec
        self.dt = float(spec["tau"])
        self.T = float(spec["T"])
        self.K = int(round(self.T / self.dt))
        self.N = int(spec["N_PLAYERS"])
        self.I = int(spec["n_types"])
        self.B = int(batch_size)

        self.BOX_POS = float(spec["BOX_POS"])
        self.BOX_VEL = float(spec["BOX_VEL"])
        self.BOX_ACC = float(spec["BOX_ACC"])

        self.R1 = jnp.asarray(spec["R1"], dtype=F32)
        self.R2 = jnp.asarray(spec["R2"], dtype=F32)
        self.P_OFFSETS = jnp.asarray(spec["P_OFFSETS"], dtype=F32)  # (I,1)
        self.tackle_pen = jnp.asarray(spec["TACKLE_PENALTY"], dtype=F32)
        self.RB_DEPTH = float(spec["RB_DEPTH"])
        self.n_substeps = int(spec["n_substeps"])
        self.merge_r2 = float(spec["MERGE_RADIUS"]) ** 2

        self.OFF_POS = spec.get("OFF_POS")
        self.DEF_POS = spec.get("DEF_POS")
        if self.OFF_POS is not None:
            self.OFF_POS = jnp.asarray(self.OFF_POS, dtype=F32)
        if self.DEF_POS is not None:
            self.DEF_POS = jnp.asarray(self.DEF_POS, dtype=F32)
        if (self.OFF_POS is not None) and (self.OFF_POS.shape[0] != self.N):
            self.N = int(self.OFF_POS.shape[0])

        self.BALL_IDX = int(spec.get("RB_INDEX", self.N // 2))

        # Dimensions
        self.ACTION_DIM = 2 * self.N
        self.STATE_DIM = 8 * self.N
        self.BELIEF_DIM = self.I - 1
        self.FEAT_DIM = self.STATE_DIM + self.BELIEF_DIM

        # Opponent mask (2N x 2N) for potential future use; kept for parity
        M = 2 * self.N
        side = jnp.arange(M) < self.N
        self.opp_mask = (side[:, None] ^ side[None, :]).astype(F32)

        # Merge/tackle parameters
        self.k_tackle = 60.0
        self.w_tackle_thr = self.merge_r2

        # Uniform prior
        self.P0 = jnp.full((1, self.I), 1.0 / self.I, dtype=F32)

    # -----------------------------------------------------
    # Resets
    # -----------------------------------------------------
    def _lineup(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return (pos1_row, pos2_row) each (N,2). JAX arrays."""
        if (self.OFF_POS is not None) and (self.DEF_POS is not None) \
           and (self.OFF_POS.shape == (self.N, 2)) and (self.DEF_POS.shape == (self.N, 2)):
            return self.OFF_POS, self.DEF_POS

        # Procedural fallback for arbitrary N
        lanes = jnp.linspace(-0.9 * self.BOX_POS, 0.9 * self.BOX_POS, num=self.N, dtype=F32)

        # Offence
        off_x = float(self.spec["LINEUP_OFF_X"])
        x_off = jnp.full((self.N,), off_x, dtype=F32)
        y_off = lanes
        rb_idx = int(self.BALL_IDX)
        x_off = x_off.at[rb_idx].set(off_x - float(self.spec["RB_DEPTH"]))

        # Defence
        def_x = float(self.spec["LINEUP_DEF_X"])
        x_def = jnp.full((self.N,), def_x, dtype=F32)
        y_def = lanes
        perm = jnp.argsort(jnp.abs(y_def))
        n_dl = min(4, self.N)
        n_lb = min(3, max(self.N - n_dl - 4, 0))
        n_db = self.N - n_dl - n_lb

        x_def = x_def.at[perm[:n_dl]].set(def_x)
        x_def = x_def.at[perm[n_dl:n_dl + n_lb]].set(def_x - 0.20)
        if n_db > 0:
            x_def = x_def.at[perm[n_dl + n_lb:]].set(def_x - 0.50)

        pos1_row = jnp.stack([x_off, y_off], axis=-1)  # (N,2)
        pos2_row = jnp.stack([x_def, y_def], axis=-1)
        return pos1_row, pos2_row

    def reset(self, key: jax.random.KeyArray = DEFAULT_PRNG_KEY, batch_size: Optional[int] = None) -> Tuple[FootballState, jax.random.KeyArray]:
        if batch_size is not None:
            self.B = int(batch_size)

        B = self.B
        pos1_row, pos2_row = self._lineup()
        pos1 = jnp.tile(pos1_row[None, :, :], (B, 1, 1))
        pos2 = jnp.tile(pos2_row[None, :, :], (B, 1, 1))
        vel1 = jnp.zeros_like(pos1)
        vel2 = jnp.zeros_like(pos2)

        x = jnp.concatenate(
            [
                pos1.reshape(B, -1),
                vel1.reshape(B, -1),
                pos2.reshape(B, -1),
                vel2.reshape(B, -1),
            ],
            axis=-1,
        ).astype(F32)

        t = jnp.zeros((B,), dtype=F32)
        p = jnp.tile(self.P0.astype(F32), (B, 1))

        key, k2 = jax.random.split(key)
        i_star = jax.random.randint(k2, (B,), minval=0, maxval=self.I, dtype=jnp.int32)

        w_last = jnp.zeros((B, self.N, self.N), dtype=F32)
        state = FootballState(x=x, t=t, p=p, i_star=i_star, w_last=w_last)
        return state, key

    # -----------------------------------------------------
    # Helpers to split/merge state
    # -----------------------------------------------------
    def _split_state(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B = x.shape[0]
        pos1 = x[:, 0:2 * self.N].reshape(B, self.N, 2)
        vel1 = x[:, 2 * self.N:4 * self.N].reshape(B, self.N, 2)
        pos2 = x[:, 4 * self.N:6 * self.N].reshape(B, self.N, 2)
        vel2 = x[:, 6 * self.N:8 * self.N].reshape(B, self.N, 2)
        return pos1, vel1, pos2, vel2

    def _merge_state(self, pos1, vel1, pos2, vel2) -> jnp.ndarray:
        return jnp.concatenate(
            [pos1.reshape(self.B, -1),
             vel1.reshape(self.B, -1),
             pos2.reshape(self.B, -1),
             vel2.reshape(self.B, -1)],
            axis=-1
        )

    # -----------------------------------------------------
    # Physics: smooth merge/tackle
    # -----------------------------------------------------
    def _merge(self, pos1, pos2, vel1, vel2, acc1, acc2):
        # (B,N,N,2)
        diff = pos1[:, :, None, :] - pos2[:, None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)  # (B,N,N)
        w = jax.nn.sigmoid(self.k_tackle * (self.w_tackle_thr - dist2))  # (B,N,N)

        # attackers
        w_sum_a = jnp.sum(w, axis=2, keepdims=True)  # (B,N,1)
        vel1_new = (vel1 + jnp.sum(w[:, :, :, None] * vel2[:, None, :, :], axis=2)) / (1.0 + w_sum_a)
        acc1_new = (acc1 + jnp.sum(w[:, :, :, None] * acc2[:, None, :, :], axis=2)) / (1.0 + w_sum_a)

        # defenders
        w_t = jnp.swapaxes(w, 1, 2)
        w_sum_d = jnp.sum(w_t, axis=2, keepdims=True)
        vel2_new = (vel2 + jnp.sum(w_t[:, :, :, None] * vel1[:, None, :, :], axis=2)) / (1.0 + w_sum_d)
        acc2_new = (acc2 + jnp.sum(w_t[:, :, :, None] * acc1[:, None, :, :], axis=2)) / (1.0 + w_sum_d)

        return vel1_new, vel2_new, acc1_new, acc2_new, w

    # -----------------------------------------------------
    # Running/terminal costs and Bayes
    # -----------------------------------------------------
    def _running_loss(self, u1: jnp.ndarray, u2: jnp.ndarray, p_tackle_now: jnp.ndarray) -> jnp.ndarray:
        # u1/u2 shape: (B, 2N)
        cost_u1 = jnp.einsum("bi,ij,bj->b", u1, self.R1, u1)
        cost_u2 = jnp.einsum("bi,ij,bj->b", u2, self.R2, u2)
        ctrl = 0.1 * 0.5 * (cost_u1 - cost_u2) * self.dt
        tack = self.tackle_pen * p_tackle_now
        return ctrl + tack

    def _terminal_loss(self, state: FootballState) -> jnp.ndarray:
        B = state.x.shape[0]
        pos1, _, _, _ = self._split_state(state.x)
        batch = jnp.arange(B, dtype=jnp.int32)

        eta = 0.6
        beta = 5.0

        x_ball = pos1[batch, self.BALL_IDX, 0]
        y_ball = pos1[batch, self.BALL_IDX, 1]

        alpha = self.P_OFFSETS[state.i_star, 0]  # (B,)
        base = -(x_ball + alpha * jnp.abs(y_ball))

        # add forward-drive term ONLY for sweep (type-1)
        bonus = eta * jax.nn.softplus(-x_ball, beta=beta)
        base = jnp.where(state.i_star == 1, base - bonus, base)
        return base

    @staticmethod
    def _tackle_flag_from_w(w_merge: jnp.ndarray, ball_idx: int) -> jnp.ndarray:
        # w_merge: (B,N,N), rb row is attackers axis=1 against all defenders
        w_rb = w_merge[:, ball_idx, :]  # (B,N)
        p_tackle = 1.0 - jnp.prod(1.0 - w_rb, axis=-1)
        return p_tackle

    @staticmethod
    def _bayes_update(p: jnp.ndarray, A: jnp.ndarray, j: jnp.ndarray) -> jnp.ndarray:
        # p: (B,I); A: (B,I,I); j: (B,)
        B, I = p.shape
        j_idx = j.reshape(B, 1, 1)
        Aj = jnp.take_along_axis(A, jnp.tile(j_idx, (1, I, 1)), axis=2).squeeze(-1)  # (B,I)
        numer = Aj * p
        denom = jnp.clip(jnp.sum(numer, axis=-1, keepdims=True), 1e-8, None)
        return numer / denom

    def belief_coord(self, p: jnp.ndarray) -> jnp.ndarray:
        return p[..., : self.BELIEF_DIM]

    # -----------------------------------------------------
    # One simulation step (semi-implicit Euler + smooth merges)
    # -----------------------------------------------------
    def step(self, state: FootballState, u1: jnp.ndarray, u2: jnp.ndarray) -> Tuple[FootballState, jnp.ndarray]:
        """
        Inputs:
            state: FootballState
            u1: (B, 2N) clipped to ±BOX_ACC
            u2: (B, 2N)
        Returns:
            new_state, p_tackle_now
        """
        B = state.x.shape[0]
        dt_s = self.dt / float(self.n_substeps)

        def _clip_act(u):
            # Accept either (B,2) (per-player acceleration prototype) or (B,2N)
            def expand_if_needed(x):
                return jnp.tile(x, (1, self.N)) if x.shape[-1] == 2 else x
            u = expand_if_needed(u)
            u = jnp.clip(u, -self.BOX_ACC, self.BOX_ACC)
            return u.reshape(B, self.N, 2)

        u1_mat = _clip_act(u1)
        u2_mat = _clip_act(u2)

        def body(carry, _):
            pos1, vel1, pos2, vel2 = carry
            # merge with zero "control" acc for the sticky component
            zeros = jnp.zeros_like(vel1)
            vel1_m, vel2_m, acc1_c, acc2_c, w = self._merge(pos1, pos2, vel1, vel2, zeros, zeros)

            # blend controls vs sticky accel
            p_merge_a = 1.0 - jnp.exp(-jnp.sum(w, axis=2, keepdims=True))
            p_merge_d = 1.0 - jnp.exp(-jnp.sum(w, axis=1, keepdims=True))
            acc1_tot = p_merge_a * acc1_c + (1.0 - p_merge_a) * u1_mat
            acc2_tot = p_merge_d * acc2_c + (1.0 - p_merge_d) * u2_mat

            # semi-implicit Euler
            vel1_new = jnp.clip(vel1_m + acc1_tot * dt_s, -self.BOX_VEL, self.BOX_VEL)
            vel2_new = jnp.clip(vel2_m + acc2_tot * dt_s, -self.BOX_VEL, self.BOX_VEL)
            pos1_new = jnp.clip(pos1 + vel1_new * dt_s, -self.BOX_POS, self.BOX_POS)
            pos2_new = jnp.clip(pos2 + vel2_new * dt_s, -self.BOX_POS, self.BOX_POS)
            return (pos1_new, vel1_new, pos2_new, vel2_new), w

        pos1, vel1, pos2, vel2 = self._split_state(state.x)
        (pos1_f, vel1_f, pos2_f, vel2_f), w_last = jax.lax.scan(body, (pos1, vel1, pos2, vel2), None, length=self.n_substeps)

        x_new = self._merge_state(pos1_f, vel1_f, pos2_f, vel2_f)
        t_new = state.t + self.dt
        new_state = FootballState(x=x_new, t=t_new, p=state.p, i_star=state.i_star, w_last=w_last)
        p_tackle_now = self._tackle_flag_from_w(w_last, self.BALL_IDX)
        return new_state, p_tackle_now

    # -----------------------------------------------------
    # Debug helpers
    # -----------------------------------------------------
    def assert_ok(self, state: FootballState):
        if not DEBUG_CHECKS:
            return
        assert state.x.shape == (self.B, self.STATE_DIM)
        assert state.p.shape == (self.B, self.I)
        assert state.i_star.shape == (self.B,)
        assert state.w_last.shape == (self.B, self.N, self.N)
        for arr in [state.x, state.t, state.p, state.w_last]:
            if jnp.any(jnp.isnan(arr)):
                raise ValueError("NaN detected in state tensor.")