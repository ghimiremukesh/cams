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
    #  Visualisation helper (no diff_env dependency)
    # ------------------------------------------------------------------
    def visualize_episode(self, p1_policy, p2_policy, fps: int = 5):
        """
        Return an HTML animation of a single episode using the game's own
        dynamics and the supplied time-indexed policies.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from IPython.display import HTML
        import torch

        # ---------- initialise -----------------------------------------
        self.reset()
        obs = {"x": self.x, "p": self.p, "t": self.t}
        i_star = self.i_star

        p_traj, t_traj, p1_xy, p2_xy = [], [], [], []

        print("\n========== DEBUG ROLL-OUT ==========")
        for step in range(self.K):
            # ---------- policy queries ------------------------------------------
            with torch.no_grad():
                u1, misc1 = p1_policy.action_only(obs, i_star, step)
                u2, _     = p2_policy.action_only(obs, step)

            # ---------- diagnostics ---------------------------------------------
            print(f"\n[t = {step*self.dt: .2f} s]")
            # full probability matrix A  (I×I)
            A_mat = misc1["A"][0].cpu().detach().numpy()           # (I,I)
            print("P1 probability matrix  A :")
            for r in range(self.I):
                print(f"  row {r}:", np.round(A_mat[r], 3))

            # prototype action table μ  (shared across rows)
            mu_tbl = misc1["μ"][0].cpu().numpy()              # (I,2)
            print("Global prototype actions μ :")
            for idx, vec in enumerate(mu_tbl):
                print(f"  idx {idx}: {np.round(vec, 3)}")

            j = misc1["j"].item()
            print("Chosen prototype idx :", j)
            print("u₁ action       :", u1.squeeze(0).cpu().detach().numpy())
            print("u₂ action       :", u2.squeeze(0).cpu().detach().numpy())

            # advance dynamics & belief ---------------------------------
            self.step(u1, u2)
            self.p = self._bayes_update(self.p, misc1["A"], misc1["j"])

            # cache for plotting ----------------------------------------
            obs = {"x": self.x, "p": self.p, "t": self.t}
            p1_xy.append(self.x[0, 0:2].cpu().detach().numpy())
            p2_xy.append(self.x[0, 4:6].cpu().detach().numpy())
            p_traj.append(self.p[0, 0].item())
            t_traj.append((step + 1) * self.dt)

        print("====================================\n")

        # ------------------------------------------------------------------------
        #  Build the Matplotlib animation
        # ------------------------------------------------------------------------
        p_belief = np.array([0.5] + p_traj)               # prepend t=0 value
        times    = np.array([0.0] + t_traj)
        p1_xy    = np.vstack([self.x[0, 0:2].cpu().detach().numpy()] + p1_xy)
        p2_xy    = np.vstack([self.x[0, 4:6].cpu().detach().numpy()] + p2_xy)

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
        # ----- top: trajectories ----------------------------------------------
        ax0.set_xlim(-self.BOX_POS - .2, self.BOX_POS + .2)
        ax0.set_ylim(-self.BOX_POS - .2, self.BOX_POS + .2)
        ax0.set_aspect("equal")
        ax0.set_title("Hexner – trajectories")

        # target markers
        for idx, tgt in enumerate(self.Z_TARGETS.cpu().numpy()):
            star_kw = dict(marker="*", ms=14,
                        color="black" if idx == i_star.item() else "grey")
            ax0.plot(tgt[0], tgt[1], **star_kw)

        p1_sc = ax0.scatter([], [], s=80, c="red")
        p2_sc = ax0.scatter([], [], s=80, c="blue")

        # ----- bottom: belief ---------------------------------------------------
        ax1.set_xlim(0, self.T)
        ax1.set_ylim(-.05, 1.05)
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("public belief p[type-0]")
        ax1.plot(times, p_belief, color="blue")
        ax1.axhline(y=float(i_star.item() == 0), color="black", ls="--")
        ax1.set_title("Belief trajectory")

        def init():
            p1_sc.set_offsets(np.empty((0, 2)))
            p2_sc.set_offsets(np.empty((0, 2)))
            return p1_sc, p2_sc

        def update(frame):
            p1_sc.set_offsets(p1_xy[frame])
            p2_sc.set_offsets(p2_xy[frame])
            return p1_sc, p2_sc

        ani = animation.FuncAnimation(fig, update, frames=len(times),
                                    init_func=init, blit=True,
                                    interval=1000 / fps)
        plt.close(fig)
        return HTML(ani.to_jshtml())


# =========================================================
# A differentiable, two-team American-football running-play
# model built on the same abstractions as HexnerGame.
# ---------------------------------------------------------

# ---------------------------------------------------------
# Helper: default spec generator
# ---------------------------------------------------------
def default_football_spec(N: int = 11,
                          horizon: float = 2.0,
                          dt: float = 0.05,
                          device: str = "cpu",
                          *,
                          box_pos: float = 1.2,
                          box_vel: float = 5.0,
                          box_acc: float = 3.0,
                          # merge ---------------------------------
                          merge_radius: float = 0.25,
                          merge_sigma:  float = 0.05,
                          # line-up offset ------------------------
                          lineup_off_x: float = -1.2,   # offence x-coord
                          lineup_def_x: float = -0.4,   # defence x-coord
                          n_substeps: int = 4,
                          ) -> Dict[str, Any]:
    """Return a dict with all tunable parameters collected in one place."""
    twoN = 2 * N          # actions per team (ax,ay for every player)
    eye  = torch.eye(twoN)

    return {
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
        "MERGE_SIGMA" : merge_sigma,
        "LINEUP_OFF_X": lineup_off_x,
        "LINEUP_DEF_X": lineup_def_x,
        "n_substeps" : n_substeps,    # substeps to simulate contact
        # ---------------- payoff table -----------------
        #  columns:  [ α_y  ]  (sign controls bias)
        "P_OFFSETS": torch.tensor([
            [-0.8],   # type-0  inside-power :  −0.8 |y|
            [+0.8],   # type-1  edge-sweep   :  +0.8 |y|
        ]),
    }


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

        self.merge_r2   = spec["MERGE_RADIUS"] ** 2
        self.merge_sig2 = spec["MERGE_SIGMA"]  ** 2
        self.P_OFFSETS  = spec["P_OFFSETS"].to(self.device)  # shape (I,1)

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

        # line-up: vertical string centred at y=0, x = −1 / +1
        idx        = torch.arange(N, device=dev)          # 0 … N−1
        y_coords   = (idx - (N - 1) / 2) / (N - 1)        # ≈ −½ … +½
        off_x = self.spec["LINEUP_OFF_X"]          # e.g., −1.2
        def_x = self.spec["LINEUP_DEF_X"]          # e.g., −0.4
        pos1_row = torch.stack([torch.full_like(y_coords, off_x),  y_coords], dim=-1)
        pos2_row = torch.stack([torch.full_like(y_coords, def_x),  y_coords], dim=-1)

        pos1       = pos1_row.unsqueeze(0).repeat(B, 1, 1)   # (B,N,2)
        pos2       = pos2_row.unsqueeze(0).repeat(B, 1, 1)
        vel1       = torch.zeros_like(pos1)
        vel2       = torch.zeros_like(pos2)

        self.x     = self._merge_state(pos1, vel1, pos2, vel2).to(dev)  # (B,STATE_DIM)
        self.t     = torch.zeros(B, device=dev)

        self.P0    = torch.full((1, self.I), 1.0 / self.I, device=dev)   # <-- add this
        self.p     = self.P0.repeat(B, 1)                                # public belief
        self.i_star= torch.randint(0, self.I, (B,), device=dev)         # hidden ball-carrier
        self.traj  : List[Dict[str, Any]] = []

        return {"x": self.x.detach(), "p": self.p.detach(), "t": self.t.detach()}

    # -----------------------------------------------------
    #   Physics helpers
    # -----------------------------------------------------
    # def _compute_repulsion(self, pos_all: Tensor) -> Tensor:
    #     """
    #     Smooth pair-wise repulsive acceleration for every player
    #     (both teams together).  
    #     pos_all: (B, 2N, 2) – concatenated positions.
    #     Returns a tensor of the same leading shape with Δ̈ contributions.
    #     """
    #     B, M, _ = pos_all.shape                            # M = 2N
    #     diff    = pos_all.unsqueeze(2) - pos_all.unsqueeze(1)     # (B,M,M,2)
    #     dist2   = (diff**2).sum(-1) + self.eps                      # (B,M,M)

    #     mask    = (dist2 < self.r_cut2) & (dist2 > 0)              # ignore self-pairs
    #     # magnitude: θ p / (‖Δx‖²)^{p/2+1}
    #     mag     = self.theta * self.rep_power * mask / (dist2 ** (self.rep_power/2 + 1))  # (B,M,M)
    #     force   = (mag.unsqueeze(-1) * diff).sum(2)                # (B,M,2) signed

    #     return force                                               # acceleration contribution

    # def _contact_acc(self, pos_all: Tensor, vel_all: Tensor) -> Tensor:
    #     """
    #     Kelvin–Voigt spring + dashpot between OPPOSING players only.
    #     Return accelerations (B,2N,2).  Masses are 1.
    #     """
    #     diff  = pos_all.unsqueeze(2) - pos_all.unsqueeze(1)       # (B,M,M,2)
    #     dist2 = (diff**2).sum(-1) + 1e-6
    #     mask  = (dist2 < self.r_cut2).float() * self.opp_mask     # include only opponents

    #     r     = dist2.sqrt()
    #     n     = diff / r.unsqueeze(-1)

    #     # spring -------------------------
    #     delta = (self.r_cut - r).clamp(min=0.0)
    #     F_s   = self.rep_k * delta.unsqueeze(-1) * n

    #     # dash-pot -----------------------
    #     v_rel = vel_all.unsqueeze(2) - vel_all.unsqueeze(1)
    #     vn    = (v_rel * n).sum(-1, keepdim=True).clamp(max=0.0)  # only approaching
    #     F_d   = self.rep_c * vn * n

    #     F = (F_s + F_d) * mask.unsqueeze(-1)
    #     return F.sum(2)               # (B,M,2)

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

        gaussian = torch.exp(-dist2 / (2 * self.merge_sig2))      # soft bandwidth
        inside   = (dist2 < self.merge_r2).float()                # hard cut-off
        w        = gaussian * inside                              # out-of-place ✔️

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

    # ------------------------------------------------------------------
    # def _inelastic_impulse(self, pos_all, vel_all):
    #     diff  = pos_all.unsqueeze(2) - pos_all.unsqueeze(1)
    #     dist2 = (diff**2).sum(-1) + 1e-6
    #     mask  = (dist2 < self.merge_r2).float() * self.opp_mask   # only within merge radius
    #     r     = dist2.sqrt()
    #     n     = diff / r.unsqueeze(-1)

    #     v_rel = vel_all.unsqueeze(2) - vel_all.unsqueeze(1)
    #     vn    = (v_rel * n).sum(-1, keepdim=True)
    #     approaching = (vn < 0).float() * mask.unsqueeze(-1)

    #     j = -vn * approaching * 0.5                               # masses=1, e=0
    #     vel_all = vel_all + (j * n).sum(2) - (j * n).sum(1)
    #     return vel_all

    # -----------------------------------------------------
    def _running_loss(self, u1: Tensor, u2: Tensor) -> Tensor:
        """
        Quadratic control effort difference, scaled by dt.
        """
        u1_cost = (u1 @ self.R1 @ u1.T).diag()
        u2_cost = (u2 @ self.R2 @ u2.T).diag()
        return 0.5 * (u1_cost - u2_cost) * self.dt         # (B,)

    def _terminal_loss(self) -> Tensor:
        # """
        # Offence wants its hidden ball-carrier to reach high +x.
        # P1 cost   = −x_{i★}(T)  
        # P2 cost   = +x_{i★}(T)  (implicit in minimax difference)
        # """
        # pos1, _, _, _ = self._split_state(self.x)          # (B,N,2)
        # batch_idx      = torch.arange(self.B, device=self.device)
        # xi_star        = pos1[batch_idx, self.i_star, 0]   # grab x-coord
        # return -xi_star                                    # (B,)
        """
        Two hidden pay-off cases:
          type-0 : inside-power  ->  -( x − 0.8 |y| )
          type-1 : edge-sweep    ->  -( x + 0.8 |y| )
        """
        pos1, _, _, _ = self._split_state(self.x)          # (B,N,2)
        batch_idx     = torch.arange(self.B, device=self.device)

        x_ball = pos1[batch_idx, self.i_star, 0]           # (B,)
        y_ball = pos1[batch_idx, self.i_star, 1]           # (B,)

        alpha  = self.P_OFFSETS[self.i_star, 0]            # +0.8 or –0.8

        term   = -(x_ball + alpha * torch.abs(y_ball))     # offence maximises
        return term  

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

            # 2) total accel = controls unless merged -----------------
            mask_a = (w.sum(2, keepdim=True) > 0)          # (B,N,1)
            mask_d = (w.sum(1, keepdim=False) > 0).unsqueeze(-1)  # (B,N,1)
            acc1_tot = torch.where(mask_a, acc1_c, u1)
            acc2_tot = torch.where(mask_d, acc2_c, u2)

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
            running_costs.append(self._running_loss(u1, u2))

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

    # -----------------------------------------------------
    #  Visualisation helper  (no diff_env dependency)
    # -----------------------------------------------------
    def visualize_episode(self,
                        p1_policy: nn.Module,
                        p2_policy: nn.Module,
                        fps: int = 6):
        """
        Render a single play.
        • Red  = offence players
        • Blue = defence players
        • Text label (upper‐left) shows the hidden play type.
        """
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from IPython.display import HTML
        import numpy as np
        import torch

        # ---------- run one episode -------------------------------
        self.reset(batch_size=1)
        obs     = {"x": self.x, "p": self.p, "t": self.t}
        i_star  = int(self.i_star.item())               # 0 or 1
        play_nm = self.PLAY_NAMES[i_star]

        traj_off, traj_def = [], []
        for k in range(self.K):
            with torch.no_grad():
                u1, _ = p1_policy.action_only(obs, self.i_star, k)
                u2, _ = p2_policy.action_only(obs, k)
            self.step(u1, u2)
            pos1, _, pos2, _ = self._split_state(self.x)
            traj_off.append(pos1[0].cpu().detach().numpy())      # (N,2)
            traj_def.append(pos2[0].cpu().detach().numpy())
            obs = {"x": self.x, "p": self.p, "t": self.t}

        traj_off.insert(0, traj_off[0])   # duplicate first for frame 0
        traj_def.insert(0, traj_def[0])

        # ---------- build animation --------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.BOX_POS - .2, self.BOX_POS + .2)
        ax.set_ylim(-self.BOX_POS - .2, self.BOX_POS + .2)
        ax.set_aspect("equal")
        ax.set_title("Differentiable Football – play demo")

        # text label for the hidden type
        txt = ax.text(0.02, 0.95,
                    f"Play: {play_nm}",
                    transform=ax.transAxes,
                    fontsize=12, fontweight="bold",
                    verticalalignment="top")

        scat_off = ax.scatter([], [], s=80, c="red", label="Offence")
        scat_def = ax.scatter([], [], s=80, c="blue", label="Defence")
        ax.legend(loc="upper right")

        def init():
            scat_off.set_offsets(np.empty((0, 2)))   # ← was []  (❌)
            scat_def.set_offsets(np.empty((0, 2)))   # ← was []
            return scat_off, scat_def, txt

        def update(frame):
            scat_off.set_offsets(traj_off[frame])
            scat_def.set_offsets(traj_def[frame])
            return scat_off, scat_def, txt

        ani = animation.FuncAnimation(fig, update,
                                    frames=len(traj_off),
                                    interval=1000 / fps,
                                    init_func=init, blit=True)
        plt.close(fig)
        return HTML(ani.to_jshtml())