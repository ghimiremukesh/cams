"""
game.py
==============
Define base game and all its extensions:
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from pathlib import Path

# ---------------------------------------------------------------------------
class BaseGame(nn.Module):
    """Abstract interface every concrete game must implement."""
    def __init__(self, *,
                 horizon: float = 1.0,
                 dt: float = 0.25,
                 n_types: int = 2,
                 device: torch.device | str = "cpu"):
        super().__init__()
        self.T  = horizon
        self.dt = dt
        self.K  = int(round(horizon / dt))
        self.I  = n_types
        self.device = torch.device(device)

    # -- methods each concrete game must override -------------------------
    def reset(self, batch_size: int = 1) -> Dict[str, Tensor]:
        raise NotImplementedError

    def rollout(self,
                p1_policy: nn.Module,
                p2_policy: nn.Module,
                ) -> tuple[Tensor, List[Dict[str, Any]]]:
        raise NotImplementedError

    def animate(self, traj: List[Dict[str, Any]]):
        pass


# ---------------------------------------------------------------------------
class HexnerGame(BaseGame):
    """Hexner two‑player differential game."""


    # ------------------------------------------------------------------
    def __init__(self, spec, batch_size):
        self.spec    = spec
        self.dt      = spec["tau"]
        self.T       = spec["T"]
        self.K       = int(round(self.T / self.dt))
        self.I       = spec["n_types"]
        self.device  = torch.device(spec["device"]) if spec.get("device") else torch.device("cpu")

        # static constants
        self.BOX_POS   = spec["BOX_POS"]
        self.BOX_VEL   = spec["BOX_VEL"]
        self.BOX_ACC   = spec["BOX_ACC"]

        self.Z_TARGETS = spec["Z_TARGETS"].to(self.device)
        self.Kmat      = spec["Kmat"].to(self.device)
        self.R1        = spec["R1"].to(self.device)
        self.R2        = spec["R2"].to(self.device)

        self.B         = batch_size

        super().__init__(horizon=self.T, dt=self.dt, n_types=self.I, device=self.device)

        # # random seed for reproducibility
        # self.rng = np.random.default_rng(4321)
        # torch.manual_seed(4321)

        # --- feature‑dimension bookkeeping ---------------------------
        # Δ¹ simplex has I-1 free coords → BELIEF_DIM
        self.BELIEF_DIM: int = self.I - 1
        # observation features = flattened target matrix + belief coord dim
        self.ACTION_DIM = int(self.R1.shape[0])
        self.STATE_DIM = int(self.Z_TARGETS.numel())
        self.FEAT_DIM: int = self.STATE_DIM + self.BELIEF_DIM
        
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        batch_size = self.B
        self.x = torch.zeros(batch_size, self.STATE_DIM, device=self.device)
        self.x[:, 0] = -0.5  # P1 initial x
        self.x[:, 4] = +0.5  # P2 initial x

        self.t = torch.zeros(batch_size, device=self.device)
        self.p = torch.full((batch_size, self.I), 1.0 / self.I, device=self.device)
        self.i_star = torch.randint(0, self.I, (batch_size,), device=self.device)

        self.traj: List[Dict[str, Tensor]] = []

        self.P0 = torch.full((1, self.I), 1.0 / self.I, device=self.device)   # uniform prior

        return {"x": self.x.detach(), "p": self.p.detach(), "t": self.t.detach()}

    # ------------------------------------------------------------------
    def _running_loss(self, u1: Tensor, u2: Tensor) -> Tensor:
        u1_cost = (u1 @ self.R1 @ u1.T).diag()
        u2_cost = (u2 @ self.R2 @ u2.T).diag()
        return 0.5 * (u1_cost - u2_cost) * self.dt

    def _terminal_loss(self) -> Tensor:
        x1 = self.x[:, 0:int(self.STATE_DIM/2)]
        x2 = self.x[:, int(self.STATE_DIM/2):self.STATE_DIM]
        del1 = x1 - self.Z_TARGETS[self.i_star]
        del2 = x2 - self.Z_TARGETS[self.i_star]
        term = 0.5 * ((del1 @ self.Kmat * del1).sum(-1) -
                      (del2 @ self.Kmat * del2).sum(-1))
        return term

    # ------------------------------------------------------------------
    @staticmethod
    def _bayes_update(p: Tensor, A: Tensor, j: Tensor):
        B, I = p.shape
        j = j.to(dtype=torch.long).view(B, 1, 1)          # (B,1,1)

        # Gather the selected column for every batch element --------------
        # index  shape : (B, I, 1) – replicate the batch-specific j along row dim
        index = j.expand(-1, I, 1)                        # (B,I,1)
        Aj = A.gather(dim=2, index=index).squeeze(-1)     # (B,I)

        # Bayes rule ------------------------------------------------------
        numer = Aj * p                                    # (B,I)
        denom = numer.sum(-1, keepdim=True).clamp_min(1e-8)
        return numer / denom                              # (B,I)

    # ------------------------------------------------------------------
    def step(self, u1: Tensor, u2: Tensor):
        u1 = u1.reshape(self.B, self.ACTION_DIM)
        u2 = u2.reshape(self.B, self.ACTION_DIM)

        position_dim = int(self.STATE_DIM/4)
        pos1, vel1 = self.x[:, 0:position_dim], self.x[:, position_dim:2*position_dim]
        pos2, vel2 = self.x[:, 2*position_dim:3*position_dim], self.x[:, 3*position_dim:4*position_dim]

        pos1 = (pos1 + vel1 * self.dt + 0.5 * u1 * self.dt**2).clamp(-self.BOX_POS, self.BOX_POS)
        vel1 = (vel1 + u1 * self.dt).clamp(-self.BOX_VEL, self.BOX_VEL)
        pos2 = (pos2 + vel2 * self.dt + 0.5 * u2 * self.dt**2).clamp(-self.BOX_POS, self.BOX_POS)
        vel2 = (vel2 + u2 * self.dt).clamp(-self.BOX_VEL, self.BOX_VEL)

        self.x = torch.cat([pos1, vel1, pos2, vel2], -1)
        self.t = self.t + self.dt

    # ------------------------------------------------------------------
    def rollout(self, p1_net, p2_net):
        """
        Unroll K steps and return:
            • total_loss  – scalar differentiable
            • traj        – list of diagnostics per step  (for viz)
        """
        running_costs: List[Tensor] = []

        for k in range(self.K):
            # ------------- build observation ----------------------------
            obs = {"t": self.t, "x": self.x, "p": self.p}

            # ------------- query policies -------------------------------
            u1, misc1 = p1_net.action_only(obs, self.i_star, k)   # (B,2)
            u2, misc2 = p2_net.action_only(obs, k)                # (B,2)

            # ------------- physics + cost -------------------------------
            self.step_dynamics(u1, u2)
            running_costs.append(self._running_loss(u1, u2))

            # ------------- Bayes belief update --------------------------
            self.p = self._bayes_update(self.p, misc1["A"], misc1["j"])

            # ------------- trajectory logging (for visualisation) -------
            self.traj.append({
                "t"      : self.t.detach().cpu(),
                "p1_xy"  : self.x[:, 0:2].detach().cpu(),
                "p2_xy"  : self.x[:, 4:6].detach().cpu(),
                "belief" : self.p.detach().cpu(),
                "A"      : misc1["A"].cpu(),        # (B,I,I)
                "μ"      : misc1["μ"].cpu(),        # (B,I,2)
                "j"      : misc1["j"].cpu()
            })

        total_running = torch.stack(running_costs, dim=0).sum(0)      # (B,)
        total_loss    = (total_running + self._terminal_loss()).mean()  # scalar
        return total_loss, self.traj

    # ------------------------------------------------------------------
    def belief_coord(self, p: Tensor) -> Tensor:
        """
        Convert public belief p ∈ Δᴵ to its I-1 free barycentric coordinates.
        For I = 2 this is simply p₀.
        """
        return p[..., :self.BELIEF_DIM]
    


    # ------------------------------------------------------------------
    def summary(self, stats: Dict[str, float]):
        print(f"[Iter {stats['iter']:04d}]  L={stats['loss']:+.3f}  |g₁|={stats['g1']:.3f}  |g₂|={stats['g2']:.3f}")

    # ------------------------------------------------------------------
    def visualize_most_likely(self, p1_policy, p2_policy, fps: int = 6):
        """
        Render one HTML animation per hidden type i★ by following the
        *most-probable public-message sequence* under the current P1 policy.

        Returns
        -------
        list[ IPython.display.HTML ]  – one entry for each hidden type.
        """
        import numpy as np
        import torch
        from IPython.display import HTML

        dev  = self.device
        outs = []

        for i_star in range(self.I):
            # -------- reset environment ----------------------------------
            self.reset()
            obs = {"x": self.x, "p": self.p, "t": self.t}

            # -------- trajectory caches ----------------------------------
            traj_p1 = [self.x[0, 0:2].cpu().numpy()]
            traj_p2 = [self.x[0, 4:6].cpu().numpy()]
            p_belief = [self.p[0, 0].item()]
            times    = [0.0]

            for k in range(self.K):
                with torch.no_grad():
                    out = p1_policy.forward(obs, k)        # NEW – no history arg
                    A     = torch.softmax(out["A_logits"][0], dim=-1)   # (I,I)
                    row   = A[i_star]                                      # row i★
                    j_k   = torch.argmax(row).item()                      # MAP column

                    μ_tbl = out["μ"][0]                                   # (I,d)
                    u1    = μ_tbl[j_k].unsqueeze(0)                       # (1,d)
                    u2    = p2_policy.forward(obs, k)                     # (1,d)

                # ------ dynamics & belief update --------------------------
                self.step(u1, u2)
                self.p = self._bayes_update(self.p,
                                            A.unsqueeze(0),              # (1,I,I)
                                            torch.tensor([j_k],
                                                        device=dev))    # (1,)

                obs = {"x": self.x, "p": self.p, "t": self.t}

                traj_p1.append(self.x[0, 0:2].cpu().detach().numpy())
                traj_p2.append(self.x[0, 4:6].cpu().detach().numpy())
                p_belief.append(self.p[0, 0].item())
                times.append((k + 1) * self.dt)

            # -------- build HTML animation (uses existing helper) ----------
            html, ani = self._make_hexner_animation(
                np.array(traj_p1),
                np.array(traj_p2),
                np.array(times),
                np.array(p_belief),
                i_star, fps
            )
            outs.append((html, ani))

        return outs


    # helper that reuses your earlier Matplotlib setup
    def _make_hexner_animation(self, p1_xy, p2_xy, times, p_belief,
                               i_star, fps):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
        ax0.set_xlim(-self.BOX_POS-.2, self.BOX_POS+.2)
        ax0.set_ylim(-self.BOX_POS-.2, self.BOX_POS+.2)
        ax0.set_aspect("equal")
        ax0.set_title(f"Most-likely path – type {i_star}")

        for idx, tgt in enumerate(self.Z_TARGETS.cpu().numpy()):
            ax0.plot(tgt[0], tgt[1], marker="*", ms=14,
                    color="black" if idx==i_star else "grey")

        p1_sc = ax0.scatter([], [], s=80, c="red")
        p2_sc = ax0.scatter([], [], s=80, c="blue")

        ax1.set_xlim(0, self.T)
        ax1.set_ylim(-.05, 1.05)
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("belief p[type-0]")
        ax1.plot(times, p_belief, color="blue")
        ax1.axhline(y=float(i_star==0), color="black", ls="--")
        ax1.set_title("Belief trajectory")

        def init():
            empty = np.empty((0,2))
            p1_sc.set_offsets(empty)
            p2_sc.set_offsets(empty)
            return p1_sc, p2_sc

        def update(frame):
            p1_sc.set_offsets(p1_xy[frame])
            p2_sc.set_offsets(p2_xy[frame])
            return p1_sc, p2_sc

        ani = animation.FuncAnimation(fig, update, frames=len(times),
                                    init_func=init, blit=True,
                                    interval=1000/fps)
        plt.close(fig)
        return HTML(ani.to_jshtml()), ani











# =========================================================
# A differentiable, two-team American-football running-play
# model built on the same abstractions as HexnerGame.
# ---------------------------------------------------------

# ---------------------------------------------------------
# Helper: default spec generator
# ---------------------------------------------------------
# ---------------------------------------------------------
# Helper: default spec generator
# ---------------------------------------------------------
def default_football_spec(N: int = 11,
                          horizon: float = 2.0,
                          dt: float = 0.05,
                          device: str = "cpu",
                          *,
                          box_pos: float = 1.6,
                          box_vel: float = 5.0,
                          box_acc: float = 3.0,
                          # merge ---------------------------------
                          merge_radius: float = 0.15,
                          # line-up offset ------------------------
                          lineup_off_x: float = -1.2,   # offence x-coord
                          lineup_def_x: float = -0.8,   # defence x-coord
                          n_substeps: int = 4,
                          ) -> Dict[str, Any]:
    """Return a dict with all tunable parameters collected in one place."""
    twoN = 2 * N          # actions per team (ax,ay for every player)
    eye  = torch.eye(twoN)

    spec = {
        "tau"        : dt,
        "T"          : horizon,
        "device"     : device,
        # roster size / information-set size --------------------
        "N_PLAYERS"  : N,
        "n_types"    : 2,               # inside-power vs edge-sweep
        # hard bounds ------------------------------------------
        "BOX_POS"    : box_pos,
        "BOX_VEL"    : box_vel,
        "BOX_ACC"    : box_acc,
        # quadratic running-cost weights -----------------------
        "R1"         : eye,             # offence weight matrix
        "R2"         : eye,             # defence weight matrix
        "MERGE_RADIUS": merge_radius,
        "LINEUP_OFF_X": lineup_off_x,
        "LINEUP_DEF_X": lineup_def_x,
        "TACKLE_PENALTY": 5.0,      # extra −yards if tackled (makes loss big) 
        "RB_DEPTH"   : 0.25,        # depth of RB position (used only if formation not provided)
        "n_substeps" : n_substeps,  # substeps to simulate contact
        # ---------------- payoff table -----------------
        #  columns:  [ α_y  ]  (sign controls bias)
        "P_OFFSETS": torch.tensor([
            [-0.8],   # type-0  inside-power :  −0.8 |y|
            [+0.8],   # type-1  edge-sweep   :  +0.8 |y|
        ]),
    }

    # If N==11, inject a realistic formation (I-formation vs 4-3 base)
    if N == 11:
        # Coordinate system:
        #  • x increases toward the defence (offence starts at more-negative x)
        #  • y is lateral, bounded by ±BOX_POS
        off_x = lineup_off_x
        def_x = lineup_def_x

        # ----- OFFENCE (I-formation, 21 personnel) -------------------------
        # Index map (offence):
        #   0 LT, 1 LG, 2 C, 3 RG, 4 RT, 5 TE (right), 6 WR-L, 7 WR-R,
        #   8 QB, 9 FB, 10 RB
        y_ol = torch.tensor([-0.80, -0.40, 0.00, 0.40, 0.80])  # OL spread
        te_y  = torch.tensor( 1.10)
        wrL_y = torch.tensor(-1.45)
        wrR_y = torch.tensor( 1.45)
        QB_x  = off_x - 0.20
        FB_x  = off_x - 0.30
        RB_x  = off_x - 0.40

        off_pos = torch.stack([
            torch.tensor([off_x, y_ol[0]]),  # LT
            torch.tensor([off_x, y_ol[1]]),  # LG
            torch.tensor([off_x, y_ol[2]]),  # C
            torch.tensor([off_x, y_ol[3]]),  # RG
            torch.tensor([off_x, y_ol[4]]),  # RT
            torch.tensor([off_x, te_y     ]),# TE (right)
            torch.tensor([off_x, wrL_y    ]),# WR-L (X)
            torch.tensor([off_x, wrR_y    ]),# WR-R (Z)
            torch.tensor([QB_x, 0.0       ]),# QB
            torch.tensor([FB_x, 0.20      ]),# FB
            torch.tensor([RB_x, 0.00      ]),# RB (ball-carrier)
        ], dim=0)

        # ----- DEFENCE (4-3 base) ------------------------------------------
        # Index map (defence):
        #   0 LDE, 1 LDT, 2 RDT, 3 RDE, 4 SLB, 5 MLB, 6 WLB,
        #   7 CB-L, 8 CB-R, 9 FS, 10 SS
        y_dl = torch.tensor([-0.60, -0.20, 0.20, 0.60])  # DL alignments
        y_lb = torch.tensor([-0.80,  0.00, 0.80])        # SLB, MLB, WLB
        CB_yL, CB_yR = -1.45, 1.45
        FS_y, SS_y   = -0.90, 0.90

        DL_x = def_x
        LB_x = def_x - 0.15
        CB_x = def_x + 0.05        # slightly pressed
        S_x  = def_x - 0.45        # deep safeties

        def_pos = torch.stack([
            torch.tensor([DL_x, y_dl[0]]),  # LDE
            torch.tensor([DL_x, y_dl[1]]),  # LDT
            torch.tensor([DL_x, y_dl[2]]),  # RDT
            torch.tensor([DL_x, y_dl[3]]),  # RDE
            torch.tensor([LB_x, y_lb[0]]),  # SLB
            torch.tensor([LB_x, y_lb[1]]),  # MLB
            torch.tensor([LB_x, y_lb[2]]),  # WLB
            torch.tensor([CB_x, CB_yL   ]), # CB-L
            torch.tensor([CB_x, CB_yR   ]), # CB-R
            torch.tensor([S_x , FS_y    ]), # FS
            torch.tensor([S_x , SS_y    ]), # SS
        ], dim=0)

        # Enforce bounds softly by clipping within the playable box
        off_pos[:, 0] = off_pos[:, 0].clamp(-box_pos, box_pos)
        off_pos[:, 1] = off_pos[:, 1].clamp(-box_pos, box_pos)
        def_pos[:, 0] = def_pos[:, 0].clamp(-box_pos, box_pos)
        def_pos[:, 1] = def_pos[:, 1].clamp(-box_pos, box_pos)

        spec["OFF_POS"]  = off_pos
        spec["DEF_POS"]  = def_pos
        spec["RB_INDEX"] = 10
        spec["PLAY_ROLES"] = {
            "offence": ["LT","LG","C","RG","RT","TE","WR-L","WR-R","QB","FB","RB"],
            "defence": ["LDE","LDT","RDT","RDE","SLB","MLB","WLB","CB-L","CB-R","FS","SS"],
            "notes": "I-formation (21 personnel) vs 4-3 base"
        }

    return spec


# ---------------------------------------------------------
class FootballGame(BaseGame):
    """
    Two-team (offence P1, defence P2) differential game in which
    every athlete is a point mass with double-integrator dynamics
    and a smooth, differentiable repulsive contact force.
    """
    # -----------------------------------------------------
    def __init__(self, spec: Dict[str, Any], batch_size: int = 1):
        # -- copy spec --------------------------------------------------
        self.spec      = spec
        self.dt        = spec["tau"]
        self.T         = spec["T"]
        self.K         = int(round(self.T / self.dt))
        self.device    = torch.device(spec["device"]) if spec.get("device") else torch.device("cpu")

        # roster size ---------------------------------------------------
        self.N         = spec["N_PLAYERS"]
        self.I         = spec["n_types"]        # = self.N
        self.B         = batch_size

        # caps ----------------------------------------------------------
        self.BOX_POS   = spec["BOX_POS"]
        self.BOX_VEL   = spec["BOX_VEL"]
        self.BOX_ACC   = spec["BOX_ACC"]

        # quadratic cost weights ---------------------------------------
        self.R1        = spec["R1"].to(self.device)
        self.R2        = spec["R2"].to(self.device)

        # cache optional formation tensors on device and align N if provided
        self.OFF_POS = self.spec.get("OFF_POS")
        self.DEF_POS = self.spec.get("DEF_POS")
        if self.OFF_POS is not None:
            self.OFF_POS = self.OFF_POS.to(self.device)
        if self.DEF_POS is not None:
            self.DEF_POS = self.DEF_POS.to(self.device)
        # If explicit positions exist and disagree with N, trust the formation
        if (self.OFF_POS is not None) and (self.OFF_POS.shape[0] != self.N):
            self.N = int(self.OFF_POS.shape[0])

        self.merge_r2   = spec["MERGE_RADIUS"] ** 2
        self.P_OFFSETS  = spec["P_OFFSETS"].to(self.device)  # shape (I,1)
        self.tackle_pen  = spec["TACKLE_PENALTY"]  
        self.RB_DEPTH   = spec["RB_DEPTH"]
        self.BALL_IDX = int(self.spec.get("RB_INDEX", self.N // 2))
        self.w_tackle_thr = self.merge_r2
        self.k_tackle     = 60.0          # steepness; 60 ≈ 4 cm logistic band

        self.n_substeps = spec["n_substeps"]   # physics sub-steps per user-visible dt

        self.eps       = 1e-6

        # bookkeeping dimensions ---------------------------------------
        self.ACTION_DIM = 2 * self.N               # ax,ay for each player on one team
        self.STATE_DIM  = 8 * self.N               # [pos,vel] × 2 teams
        self.BELIEF_DIM = self.I - 1               # Δᴵ → ℝ^{I-1}
        self.FEAT_DIM   = self.STATE_DIM + self.BELIEF_DIM

        self.PLAY_NAMES = spec.get(
            "PLAY_NAMES",
            ["Inside Power", "Edge Sweep"]          # len = self.I
        )

        super().__init__(horizon=self.T, dt=self.dt,
                         n_types=self.I, device=self.device)

        # pre-compute opponent mask M×M (team1 vs team2)
        M = 2 * self.N
        side = torch.arange(M) < self.N
        self.register_buffer(
            "opp_mask", (side.unsqueeze(1) ^ side.unsqueeze(0)).float())  # (M,M)

        self.reset()

    # -----------------------------------------------------
    # utilities
    # -----------------------------------------------------
    def _split_state(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (pos1, vel1, pos2, vel2) each of shape (B,N,2)."""
        B, _    = x.shape
        pos1    = x[:, 0          : 2*self.N         ].reshape(B, self.N, 2)
        vel1    = x[:, 2*self.N   : 4*self.N         ].reshape(B, self.N, 2)
        pos2    = x[:, 4*self.N   : 6*self.N         ].reshape(B, self.N, 2)
        vel2    = x[:, 6*self.N   : 8*self.N         ].reshape(B, self.N, 2)
        return pos1, vel1, pos2, vel2

    def _merge_state(self,
                     pos1: Tensor, vel1: Tensor,
                     pos2: Tensor, vel2: Tensor) -> Tensor:
        """Inverse of _split_state."""
        return torch.cat([pos1.reshape(self.B, -1),
                          vel1.reshape(self.B, -1),
                          pos2.reshape(self.B, -1),
                          vel2.reshape(self.B, -1)], dim=-1)

    # -----------------------------------------------------
    def reset(self, batch_size: int | None = None):
        if batch_size is not None:
            self.B = batch_size

        B, N, dev = self.B, self.N, self.device

        # If explicit positions are provided (e.g., 11v11 formation), use them.
        if (self.OFF_POS is not None) and (self.DEF_POS is not None) \
           and (self.OFF_POS.shape == (self.N, 2)) and (self.DEF_POS.shape == (self.N, 2)):
            pos1_row = self.OFF_POS
            pos2_row = self.DEF_POS
        else:
            # ---------- procedural line-up for arbitrary N (fallback) ----------
            lanes = torch.linspace(-0.9 * self.BOX_POS, 0.9 * self.BOX_POS, steps=self.N, device=dev)

            # Offence (P1): everyone on LOS at x = off_x, RB deeper by RB_DEPTH
            y_off = lanes.clone()
            off_x = self.spec["LINEUP_OFF_X"]
            rb_dx = self.spec["RB_DEPTH"]
            x_off = torch.full((self.N,), off_x, device=dev)
            rb_idx = int(self.BALL_IDX)
            if 0 <= rb_idx < self.N:
                x_off[rb_idx] = off_x - rb_dx

            # Defence (P2): place DL on LOS, LBs slightly deeper, DBs deepest.
            y_def = lanes.clone()
            def_x = self.spec["LINEUP_DEF_X"]
            x_def = torch.full((self.N,), def_x, device=dev)

            # Assign depth by role groups using center-first ordering
            perm = torch.argsort(torch.abs(y_def))  # center-first ordering
            n_dl = min(4, self.N)
            n_lb = min(3, max(self.N - n_dl - 4, 0))
            n_db = self.N - n_dl - n_lb
            x_def[perm[:n_dl]] = def_x
            x_def[perm[n_dl:n_dl + n_lb]] = def_x - 0.20
            if n_db > 0:
                x_def[perm[n_dl + n_lb:]] = def_x - 0.50

            pos1_row = torch.stack([x_off, y_off], dim=-1)  # (N,2)
            pos2_row = torch.stack([x_def, y_def], dim=-1)  # (N,2)

        pos1       = pos1_row.unsqueeze(0).repeat(B, 1, 1)   # (B,N,2)
        pos2       = pos2_row.unsqueeze(0).repeat(B, 1, 1)
        vel1       = torch.zeros_like(pos1)
        vel2       = torch.zeros_like(pos2)

        self.x     = self._merge_state(pos1, vel1, pos2, vel2).to(dev)  # (B,STATE_DIM)
        self.t     = torch.zeros(B, device=dev)

        self.P0    = torch.full((1, self.I), 1.0 / self.I, device=dev)
        self.p     = self.P0.repeat(B, 1)                                # public belief
        self.i_star= torch.randint(0, self.I, (B,), device=dev)          # hidden ball-carrier
        self.traj  : List[Dict[str, Any]] = []

        return {"x": self.x.detach(), "p": self.p.detach(), "t": self.t.detach()}

    # ------------------------------------------------------------------
    def _merge(self,
            pos1: torch.Tensor, pos2: torch.Tensor,
            vel1: torch.Tensor, vel2: torch.Tensor,
            acc1: torch.Tensor, acc2: torch.Tensor):
        """
        Smoothly glue attacker–defender pairs within MERGE_RADIUS.
        All ops are out-of-place and autograd-safe.
        Returns updated (vel1, vel2, acc1, acc2, w) where
        w ∈ [0,1] measures pairwise “stickiness”.
        """
        diff   = pos1.unsqueeze(2) - pos2.unsqueeze(1)            # (B,N,N,2)
        dist2  = (diff.square()).sum(-1)                          # (B,N,N)

        w = torch.sigmoid(self.k_tackle * (self.w_tackle_thr - dist2))      # soft bandwidth

        # attackers ------------------------------------------------------
        w_sum_a  = w.sum(2, keepdim=True)                         # (B,N,1)
        vel1_new = (vel1 + (w.unsqueeze(-1) * vel2.unsqueeze(1)).sum(2)) / (1 + w_sum_a)
        acc1_new = (acc1 + (w.unsqueeze(-1) * acc2.unsqueeze(1)).sum(2)) / (1 + w_sum_a)

        # defenders ------------------------------------------------------
        w_t      = w.transpose(1, 2)                              # (B,N,N)
        w_sum_d  = w_t.sum(2, keepdim=True)
        vel2_new = (vel2 + (w_t.unsqueeze(-1) * vel1.unsqueeze(1)).sum(2)) / (1 + w_sum_d)
        acc2_new = (acc2 + (w_t.unsqueeze(-1) * acc1.unsqueeze(1)).sum(2)) / (1 + w_sum_d)

        return vel1_new, vel2_new, acc1_new, acc2_new, w           # w reused later

    # -----------------------------------------------------
    def _running_loss(self,
                      u1: Tensor, u2: Tensor,
                      p_tackle_now: Tensor | None = None) -> Tensor:
        """
        Quadratic control effort + tackle penalty per step.
        If p_tackle_now is None, compute it from the last merge weights.
        """
        cost_u1 = (u1 @ self.R1 @ u1.T).diag()
        cost_u2 = (u2 @ self.R2 @ u2.T).diag()
        ctrl    = 0.1 * 0.5 * (cost_u1 - cost_u2) * self.dt
        if p_tackle_now is None:
            p_tackle_now = self._tackle_flag(self.w_last)
        tack    = self.tackle_pen * p_tackle_now
        return ctrl + tack

    # -----------------------------------------------------
    def _terminal_loss(self):
        pos1, pos2 = self._split_state(self.x)[0], self._split_state(self.x)[2]
        batch      = torch.arange(self.B, device=self.device)

        eta = 0.6          # LOS-cross importance
        beta = 5.0         # softplus steepness

        x_ball = pos1[batch, self.BALL_IDX, 0]     # ← use fixed ball index
        y_ball = pos1[batch, self.BALL_IDX, 1]

        alpha = self.P_OFFSETS[self.i_star, 0]     # ±0.8
        base  = -(x_ball + alpha*torch.abs(y_ball))

        # add forward-drive term ONLY for sweep (type-1)
        bonus = eta * F.softplus(-x_ball, beta=beta)
        base  = torch.where(self.i_star==1, base - bonus, base)
        return base

    # ------------------------------------------------------------------
    def _tackle_flag(self, w_merge: torch.Tensor) -> torch.Tensor:
        """
        Differentiable tackle probability  p ∈ (0,1)  for each batch element.
        A defender contributes with logistic weight; OR is implemented as
        1 - Π(1 - p_i) so gradients pass through every term.
        """
        # merge weights with the RB  →  (B, N)
        w_rb = w_merge[:, self.BALL_IDX]

        # probabilistic OR  (smooth)
        p_tackle = 1.0 - torch.prod(1.0 - w_rb, dim=-1)                    # (B,)

        return p_tackle                        # differentiable scalar
    
    # -----------------------------------------------------
    def step(self, u1: Tensor, u2: Tensor):
        # --- normalise shapes ----------------------------------------
        def _shp(u):
            if u.shape[-1] == 2:
                u = u.repeat_interleave(self.N, -1)
            assert u.shape[-1] == self.ACTION_DIM
            return u.view(self.B, self.N, 2).clamp(-self.BOX_ACC, self.BOX_ACC)

        u1 = _shp(u1)
        u2 = _shp(u2)
        dt_s = self.dt / self.n_substeps

        for _ in range(self.n_substeps):
            # unpack ---------------------------------------------------
            pos1, vel1, pos2, vel2 = self._split_state(self.x)

            # 1) smooth merge first -----------------------------------
            vel1, vel2, acc1_c, acc2_c, w = self._merge(
                    pos1, pos2, vel1, vel2,
                    torch.zeros_like(vel1), torch.zeros_like(vel2))
            self.w_last = w      # save for terminal-loss check

            # 2) total accel = controls unless merged -----------------
            # ---- smooth merge probabilities ---------------------------------
            p_merge_a = 1.0 - torch.exp(-w.sum(2, keepdim=True))   # (B,N,1) attackers
            p_merge_d = 1.0 - torch.exp(-w.sum(1).unsqueeze(-1))   # (B,N,1)            
            # ---- convex blend of accelerations ------------------------------
            acc1_tot = p_merge_a * acc1_c + (1.0 - p_merge_a) * u1  # offence
            acc2_tot = p_merge_d * acc2_c + (1.0 - p_merge_d) * u2  # defence

            # 3) semi-implicit Euler ---------------------------------
            vel1 = (vel1 + acc1_tot * dt_s).clamp(-self.BOX_VEL, self.BOX_VEL)
            vel2 = (vel2 + acc2_tot * dt_s).clamp(-self.BOX_VEL, self.BOX_VEL)
            pos1 = (pos1 + vel1 * dt_s).clamp(-self.BOX_POS, self.BOX_POS)
            pos2 = (pos2 + vel2 * dt_s).clamp(-self.BOX_POS, self.BOX_POS)

            # # 4) inelastic impulse (no bounce) ------------------------
            # vel_all = torch.cat([vel1, vel2], 1)
            # vel_all = self._inelastic_impulse(torch.cat([pos1, pos2], 1), vel_all)
            # vel1, vel2 = vel_all[:, :self.N], vel_all[:, self.N:]

            # 5) commit state ----------------------------------------
            self.x = self._merge_state(pos1, vel1, pos2, vel2)

        self.t += self.dt

    # -----------------------------------------------------
    def rollout(self,
                p1_policy: nn.Module,
                p2_policy: nn.Module
                ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        """
        Unroll K steps and return average zero-sum loss plus diagnostics.
        """
        running_costs: List[Tensor] = []

        for k in range(self.K):
            obs = {"t": self.t, "x": self.x, "p": self.p}

            # ---- query policies -------------------------------------
            u1, misc1 = p1_policy.action_only(obs, self.i_star, k)  # offence sees i★
            u2, misc2 = p2_policy.action_only(obs, k)               # defence does not

            # ---- physics & cost ------------------------------------
            self.step(u1, u2)
            p_tackle_now = self._tackle_flag(self.w_last)           # after _merge
            running_costs.append(
                self._running_loss(u1, u2, p_tackle_now)
            )

            # ---- belief update (same mechanism as Hexner) ----------
            self.p = self._bayes_update(self.p, misc1["A"], misc1["j"])

            # ---- trajectory cache ----------------------------------
            pos1, _, pos2, _ = self._split_state(self.x)
            self.traj.append({
                "t"      : self.t.detach().cpu(),
                "pos1"   : pos1.detach().cpu(),   # (B,N,2)
                "pos2"   : pos2.detach().cpu(),   # (B,N,2)
                "belief" : self.p.detach().cpu(),
                "A"      : misc1["A"].cpu(),
                "μ"      : misc1["μ"].cpu(),
                "j"      : misc1["j"].cpu()
            })

        total_running = torch.stack(running_costs, dim=0).sum(0)      # (B,)
        total_loss    = (total_running + self._terminal_loss()).mean()
        return total_loss, self.traj

    # ------------------------------------------------------------------
    @staticmethod
    def _bayes_update(p: Tensor, A: Tensor, j: Tensor):
        B, I = p.shape
        j = j.to(dtype=torch.long).view(B, 1, 1)          # (B,1,1)

        # Gather the selected column for every batch element --------------
        # index  shape : (B, I, 1) – replicate the batch-specific j along row dim
        index = j.expand(-1, I, 1)                        # (B,I,1)
        Aj = A.gather(dim=2, index=index).squeeze(-1)     # (B,I)

        # Bayes rule ------------------------------------------------------
        numer = Aj * p                                    # (B,I)
        denom = numer.sum(-1, keepdim=True).clamp_min(1e-8)
        return numer / denom                              # (B,I)

    # ------------------------------------------------------------------
    def belief_coord(self, p: Tensor) -> Tensor:
        """
        Convert public belief p ∈ Δᴵ to its I-1 free barycentric coordinates.
        For I = 2 this is simply p₀.
        """
        return p[..., :self.BELIEF_DIM]
    
    # ------------------------------------------------------------------
    def summary(self, stats: Dict[str, float]):
        print(f"[Iter {stats['iter']:04d}]  L={stats['loss']:+.3f}  |g₁|={stats['g1']:.3f}  |g₂|={stats['g2']:.3f}")

    # ------------------------------------------------------------------
    def _make_football_animation(self,
                                 traj_off, traj_def,
                                 times, p_traj,
                                 i_star: int, fps: int):
        """
        Shared routine: build an HTML <video> showing
            • top  : N-player trajectories (offence red / defence blue)
            • bottom: belief p(t)[i★]
        The two numpy arrays traj_* have shape (T+1, N, 2).
        """
        import matplotlib.pyplot as plt, matplotlib.animation as anim
        from IPython.display import HTML
        import numpy as np

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(6, 8),
            gridspec_kw={"height_ratios": [4, 1]}
        )

        # ── axis limits ------------------------------------------------
        ax_top.set_xlim(-self.BOX_POS - .2, self.BOX_POS + .2)
        ax_top.set_ylim(-self.BOX_POS - .2, self.BOX_POS + .2)
        ax_top.set_aspect("equal")
        ax_top.set_title(f"Most-likely path – type {i_star}")

        scat_off = ax_top.scatter([], [], s=70, c="red")
        scat_def = ax_top.scatter([], [], s=70, c="blue")
        rb_star  = ax_top.scatter([], [], s=140, marker="*",
                                  c="gold", edgecolors="black", lw=.6)

        # ── belief plot ------------------------------------------------
        ax_bot.set_xlim(0, self.T)
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.set_xlabel("time  (s)")
        ax_bot.set_ylabel(f"belief  p[{i_star}]")
        ax_bot.plot(times, p_traj, color="black")

        # ── blit helpers ----------------------------------------------
        def init():
            empty = np.empty((0, 2))
            scat_off.set_offsets(empty)
            scat_def.set_offsets(empty)
            rb_star.set_offsets(empty)
            return scat_off, scat_def, rb_star

        def update(frame):
            scat_off.set_offsets(traj_off[frame])
            scat_def.set_offsets(traj_def[frame])
            rb_star.set_offsets(traj_off[frame, self.BALL_IDX])
            return scat_off, scat_def, rb_star

        ani = anim.FuncAnimation(
            fig, update, frames=len(times),
            init_func=init, blit=True, interval=1000 / fps
        )
        plt.close(fig)
        return HTML(ani.to_jshtml()), ani

    # ────────────────────────────────────────────────────────────────────
    # 2) NEW :  visualize_most_likely   (add below visualize_episode)
    # ────────────────────────────────────────────────────────────────────
    def visualize_most_likely(self,
                              p1_policy: nn.Module,
                              p2_policy: nn.Module,
                              fps: int = 6,
                              save_dir: str | None = None):
        """
        One HTML animation per hidden type i★ following the *most-probable*
        public message sequence under the current Player-1 policy.
        """
        import numpy as np, torch
        outs = []
        dev  = self.device

        for i_star in range(self.I):
            # --- reset environment -----------------------------------
            self.reset(batch_size=1)
            self.i_star.fill_(i_star)
            obs = {"x": self.x, "p": self.p, "t": self.t}

            # --- logging containers ---------------------------------
            pos1, _, pos2, _ = self._split_state(self.x)
            traj_off = [pos1[0].cpu().numpy()]
            traj_def = [pos2[0].cpu().numpy()]
            p_traj   = [self.p[0, 0].item()]
            times    = [0.0]

            for k in range(self.K):
                with torch.no_grad():
                    out = p1_policy.forward(obs, k)
                    A   = torch.softmax(out["A_logits"][0], dim=-1)  # (I,I)
                    row = A[i_star]
                    j_k = torch.argmax(row).item()                   # argmax_j

                    μ_tbl = out["μ"][0]                              # (I,d)
                    u1 = μ_tbl[j_k].unsqueeze(0)                     # (1,d)
                    u2 = p2_policy.forward(obs, k)                   # (1,d)

                # dynamics + belief update ---------------------------
                self.step(u1, u2)
                self.p = self._bayes_update(
                    self.p, A.unsqueeze(0), torch.tensor([j_k], device=dev)
                )

                # cache for plot ------------------------------------
                obs = {"x": self.x, "p": self.p, "t": self.t}
                pos1, _, pos2, _ = self._split_state(self.x)
                traj_off.append(pos1[0].cpu().detach().numpy())
                traj_def.append(pos2[0].cpu().detach().numpy())
                p_traj.append(self.p[0, 0].item())
                times.append((k + 1) * self.dt)

            # --- build animation -----------------------------------
            html, ani = self._make_football_animation(
                np.array(traj_off),
                np.array(traj_def),
                np.array(times),
                np.array(p_traj),
                i_star, fps
            )
            outs.append((html, ani))

        return outs

    
    # ---------------------------------------------------------------
    def save_type_animations(self,
                            p1_pol: nn.Module,
                            p2_pol: nn.Module,
                            iteration: int,
                            fps: int = 6,
                            root_dir: str = "animations") -> list[str]:
        """
        Render one rollout for every hidden type and save each as a gif.
        Returns a list of file paths written.
        """
        import os, datetime
        from pathlib import Path

        stamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(root_dir) / f"solve_{stamp}"
        outdir.mkdir(parents=True, exist_ok=True)

        gif_paths = []
        for i in range(self.I):
            html, ani = self.visualize_episode(
                p1_pol, p2_pol, fps=fps,
                force_type=i,
                return_animation=True)              # needs the raw ani

            fname = (f"iter{iteration:04d}_type{i}_"
                    f"{self.PLAY_NAMES[i].replace(' ', '')}.gif")
            gif_path = outdir / fname
            ani.save(gif_path, writer="pillow", fps=fps)
            gif_paths.append(str(gif_path))

        return gif_paths