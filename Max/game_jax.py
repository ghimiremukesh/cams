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
        # -------- free-receiver payoff (smooth "best free runner") -----
        FREE_LAMBDA=0.5,   # weight λ_free for free-runner yardage
        FREE_TAU=0.30,     # τ_free: freeness threshold (lower = stricter)
        FREE_K=20.0,       # k_free: sharpness of freeness sigmoid
        FREE_TEMP=0.15,    # T_free: softmax temperature over runners
        FREE_ALPHA=0.4,    # α_free: how much freeness lifts runner score
    )

    if N == 11:
        off_x = lineup_off_x
        def_x = lineup_def_x

        # Narrower offensive line spread and WRs a bit inside the sideline
        y_ol = np.array([-0.60, -0.30, 0.00, 0.30, 0.60], dtype=np.float32)
        te_y = np.float32(1.10)
        wrL_y = np.float32(-1.30)
        wrR_y = np.float32(+1.30)
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
        y_lb = np.array([-0.80,  0.00, 0.80], dtype=np.float32)
        CB_yL, CB_yR = -1.50, 1.50
        FS_y, SS_y = -0.90, 0.90

        DL_x = def_x
        LB_x = def_x - 0.20     # linebackers a tad deeper
        CB_x = def_x + 0.10     # corners a hair more pressed than DL
        S_x  = def_x + 0.45     # deep safeties (toward defense)

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

        # -------------------------------------------------
        # Per-role kinematic caps (closer to real roles)
        # -------------------------------------------------
        if self.N == 11:
            # Offence indices: 0 LT,1 LG,2 C,3 RG,4 RT,5 TE,6 WR-L,7 WR-R,8 QB,9 FB,10 RB
            vel_off = np.array([3.0,3.0,2.8,3.0,3.0, 3.2,4.5,4.5, 2.8,3.2,4.2], np.float32)
            acc_off = np.array([2.0,2.0,1.8,2.0,2.0, 2.2,2.8,2.8, 2.2,2.4,2.8], np.float32)
            # Defence indices: 0 LDE,1 LDT,2 RDT,3 RDE,4 SLB,5 MLB,6 WLB,7 CB-L,8 CB-R,9 FS,10 SS
            vel_def = np.array([3.5,3.2,3.2,3.5, 3.8,3.6,3.8, 4.6,4.6,4.8,4.6], np.float32)
            acc_def = np.array([2.2,2.0,2.0,2.2, 2.6,2.4,2.6, 3.0,3.0,3.0,3.0], np.float32)
        else:
            vel_off = np.full((self.N,), self.BOX_VEL, np.float32)
            acc_off = np.full((self.N,), self.BOX_ACC, np.float32)
            vel_def = np.full((self.N,), self.BOX_VEL, np.float32)
            acc_def = np.full((self.N,), self.BOX_ACC, np.float32)

        self.vel_off = jnp.asarray(vel_off, dtype=F32)
        self.acc_off = jnp.asarray(acc_off, dtype=F32)
        self.vel_def = jnp.asarray(vel_def, dtype=F32)
        self.acc_def = jnp.asarray(acc_def, dtype=F32)

        # -------------------------------------------------
        # Role-biased contact weights (stickier in the box)
        # -------------------------------------------------
        self.contact_weight = jnp.ones((self.N, self.N), dtype=F32)
        if self.N == 11:
            roles_off = {"OL": [0,1,2,3,4], "TE":[5], "WR":[6,7], "QB":[8], "FB":[9], "RB":[10]}
            roles_def = {"DL": [0,1,2,3], "LB":[4,5,6], "CB":[7,8], "S":[9,10]}

            # Heavier weight for trench battles (OL vs DL/LB)
            for i in roles_off["OL"] + roles_off["TE"]:
                for j in roles_def["DL"] + roles_def["LB"]:
                    self.contact_weight = self.contact_weight.at[i, j].set(1.5)

            # Lighter on the perimeter (WR vs CB)
            for i in roles_off["WR"]:
                for j in roles_def["CB"]:
                    self.contact_weight = self.contact_weight.at[i, j].set(0.5)

        # Opponent mask (2N x 2N) for potential future use; kept for parity
        M = 2 * self.N
        side = jnp.arange(M) < self.N
        self.opp_mask = (side[:, None] ^ side[None, :]).astype(F32)

        # Merge/tackle parameters
        self.k_tackle = 60.0
        self.w_tackle_thr = self.merge_r2

        # Uniform prior
        self.P0 = jnp.full((1, self.I), 1.0 / self.I, dtype=F32)

        # --------------- free-receiver payoff params -------------------
        self.free_lambda = float(self.spec.get("FREE_LAMBDA", 0.5))
        self.free_tau    = float(self.spec.get("FREE_TAU", 0.30))
        self.free_k      = float(self.spec.get("FREE_K", 20.0))
        self.free_temp   = float(self.spec.get("FREE_TEMP", 0.15))
        self.free_alpha  = float(self.spec.get("FREE_ALPHA", 0.4))

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
        # Scale by role-pairing affinity (trench vs perimeter)
        w = w * self.contact_weight[None, :, :]

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
        """
        Terminal zero-sum payoff selected by hidden type i★:
          • i★=0 → RB yardage:         L = - x_ball
          • i★=1 → best free receiver: L = - x_free
        (Offence wants this LOW / more negative.)
        Returns: (B,)
        """
        pos1, _, _, _ = self._split_state(state.x)
        B = state.x.shape[0]
        batch = jnp.arange(B, dtype=jnp.int32)

        # --- RB forward yardage (no lateral preference) ----------------
        x_ball = pos1[batch, self.BALL_IDX, 0]                # (B,)
        L_rb = -x_ball                                        # (B,)

        # --- Smooth "best free receiver" yardage -----------------------
        w = state.w_last                                       # (B,N,N)
        merge_sum = jnp.sum(w, axis=2)                         # (B,N)
        # Freeness s_i in (0,1): higher=fewer/softer merges
        s = jax.nn.sigmoid(self.free_k * (self.free_tau - merge_sum))  # (B,N)

        # Exclude RB from the pool
        idx = jnp.arange(self.N)
        mask_non_rb = (idx != self.BALL_IDX)[None, :]          # (1,N) → (B,N)

        # Forward positions for offence
        x_off = pos1[:, :, 0]                                  # (B,N)

        # Score boosted by freeness (no extra global weight here)
        x_eff = x_off + self.free_alpha * s

        # Mask RB by setting its score to a large negative before softmax-max
        very_neg = jnp.array(-1e9, dtype=F32)
        x_eff_masked = jnp.where(mask_non_rb, x_eff, very_neg)

        # Smooth max over non-RB players
        T = self.free_temp
        x_free = T * jax.nn.logsumexp(x_eff_masked / T, axis=1)  # (B,)
        L_free = -x_free                                         # (B,)

        # Select payoff by type: i★=0 → RB, i★=1 → Free WR
        return jnp.where(state.i_star == 0, L_rb, L_free)

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
            u1: (B, 2N) or (B,2)  — offence controls
            u2: (B, 2N) or (B,2)  — defence controls
        Returns:
            new_state, p_tackle_now
        """
        B = state.x.shape[0]
        dt_s = self.dt / float(self.n_substeps)

        def _clip_act_off(u):
            # Accept (B,2) (one proto) or (B,2N); expand if needed, then clip per-role.
            if u.shape[-1] == 2:
                u = jnp.tile(u, (1, self.N))
            u = u.reshape(B, self.N, 2)
            return jnp.clip(u, -self.acc_off[None, :, None], self.acc_off[None, :, None])

        def _clip_act_def(u):
            if u.shape[-1] == 2:
                u = jnp.tile(u, (1, self.N))
            u = u.reshape(B, self.N, 2)
            return jnp.clip(u, -self.acc_def[None, :, None], self.acc_def[None, :, None])

        u1_mat = _clip_act_off(u1)
        u2_mat = _clip_act_def(u2)

        def body(carry, _):
            pos1, vel1, pos2, vel2 = carry
            zeros = jnp.zeros_like(vel1)

            # Smooth merge (sticky velocities/accels) + weights
            vel1_m, vel2_m, acc1_c, acc2_c, w = self._merge(pos1, pos2, vel1, vel2, zeros, zeros)  # w: (B,N,N)

            # Merge probabilities for attackers/defenders
            p_merge_a = 1.0 - jnp.exp(-jnp.sum(w, axis=2, keepdims=True))      # (B,N,1)
            # BUGFIX: make defenders’ factor (B,N,1), not (B,1,N)
            p_merge_d = 1.0 - jnp.exp(-jnp.sum(w, axis=1))[:, :, None]         # (B,N,1)

            # Blend sticky accel vs. control
            acc1_tot = p_merge_a * acc1_c + (1.0 - p_merge_a) * u1_mat         # (B,N,2)
            acc2_tot = p_merge_d * acc2_c + (1.0 - p_merge_d) * u2_mat         # (B,N,2)

            # Semi-implicit Euler
            vel1_new = jnp.clip(vel1_m + acc1_tot * dt_s, -self.vel_off[None, :, None], self.vel_off[None, :, None])
            vel2_new = jnp.clip(vel2_m + acc2_tot * dt_s, -self.vel_def[None, :, None], self.vel_def[None, :, None])
            pos1_new = jnp.clip(pos1 + vel1_new * dt_s, -self.BOX_POS, self.BOX_POS)
            pos2_new = jnp.clip(pos2 + vel2_new * dt_s, -self.BOX_POS, self.BOX_POS)

            return (pos1_new, vel1_new, pos2_new, vel2_new), w

        pos1, vel1, pos2, vel2 = self._split_state(state.x)
        (pos1_f, vel1_f, pos2_f, vel2_f), w_seq = jax.lax.scan(
            body, (pos1, vel1, pos2, vel2), None, length=self.n_substeps
        )
        # BUGFIX: scan returns the whole sequence; take the final merge weights
        w_last = w_seq[-1]                                              # (B,N,N)

        x_new = self._merge_state(pos1_f, vel1_f, pos2_f, vel2_f)
        t_new = state.t + self.dt
        new_state = FootballState(x=x_new, t=t_new, p=state.p, i_star=state.i_star, w_last=w_last)

        # Use angle- and speed-aware tackle probability
        pos1_fin, vel1_fin, pos2_fin, vel2_fin = self._split_state(x_new)
        p_tackle_now = self._tackle_prob(w_last, pos1_fin, pos2_fin, vel1_fin, vel2_fin)
        return new_state, p_tackle_now

    def _tackle_prob(self, w_merge: jnp.ndarray, pos1: jnp.ndarray, pos2: jnp.ndarray,
                     vel1: jnp.ndarray, vel2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute differentiable tackle probability combining proximity (merge weight)
        and closing speed toward the RB. Returns (B,).
        """
        # RB row vs all defenders
        w_rb = w_merge[:, self.BALL_IDX, :]  # (B,N)

        # Vector from defenders to RB
        diff = pos2 - pos1[:, self.BALL_IDX:self.BALL_IDX+1, :]   # (B,N,2)
        d = jnp.linalg.norm(diff, axis=-1) + 1e-6                 # (B,N)
        dir_to_rb = diff / d[:, :, None]                          # (B,N,2)

        # Relative velocity (defender - RB)
        rel = vel2 - vel1[:, self.BALL_IDX:self.BALL_IDX+1, :]    # (B,N,2)
        closing = jnp.sum(rel * (-dir_to_rb), axis=-1)            # (B,N) >0 if closing

        # Combine proximity & closing; weights chosen to keep gradients stable
        z = 3.0 * w_rb + 0.2 * jnp.clip(closing, 0.0, None)
        p_i = jax.nn.sigmoid(z)                                    # (B,N)
        p = 1.0 - jnp.prod(1.0 - p_i, axis=-1)                     # probabilistic OR
        return p

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