"""
player.py
====================================
Define player
"""

from __future__ import annotations
import torch, torch.nn as nn
from torch import Tensor
from typing import List

# -------------------------------------------------------------
class _CAMSNet(nn.Module):
    """One step-specific CAMS policy π₁ᵏ."""
    def __init__(self, game, hidden: int):
        super().__init__()
        self.game = game
        self.FEAT_DIM = game.FEAT_DIM
        self.I = game.I
        self.ACTION_DIM  = game.ACTION_DIM
        self.net = nn.Sequential(
            nn.Linear(self.FEAT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU()
        )
        self.logit_head = nn.Linear(hidden, self.I * self.I)   # (I×I) logits
        self.mu_head    = nn.Linear(hidden, self.I * self.ACTION_DIM)   # I prototypes

    def forward(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        h = self.net(torch.cat([obs["x"],
                                self.game.belief_coord(obs["p"])], dim=-1))
        A_logits = self.logit_head(h).view(-1, self.I, self.I)      # (B,I,I)
        μ        = self.game.BOX_ACC * torch.tanh(
                       self.mu_head(h).view(-1, self.I, self.ACTION_DIM))    # (B,I,ACTION_DIM)
        return {"A_logits": A_logits, "μ": μ}

# -------------------------------------------------------------
class CAMS_INFORMED(nn.Module):
    """
    Wrapper holding K independent sub-nets π₁ᵏ.
    API:
        u , misc = policy.action_only(obs, i_star, k)
    """
    def __init__(self, game, spec):
        super().__init__()
        self.ACTION_DIM  = game.ACTION_DIM
        hidden = spec["hidden"]
        self.TEMPERATURE  = spec["temperature"]                     # row-softmax temperature for P1
        self.subnets = nn.ModuleList([_CAMSNet(game, hidden) for _ in range(game.K)])

    def forward(self, obs: dict[str, Tensor], k: int):
        return self.subnets[k](obs)

    @torch.no_grad()
    def action_only(self, obs: dict[str, Tensor], i_star: Tensor, k: int):
        """
        Select row i_star and argmax column for step k.
        """
        out        = self.forward(obs, k)
        A_logits   = out["A_logits"]                    # (B,I,I)
        μ_proto    = out["μ"]                           # (B,I,ACTION_DIM)

        B_idx      = torch.arange(A_logits.size(0), device=A_logits.device)
        logits_row = A_logits[B_idx, i_star]
        j          = torch.argmax(logits_row, dim=-1)   # (B,)

        u          = μ_proto[B_idx, j]                  # (B,ACTION_DIM)

        misc = {
            "A":   torch.softmax(A_logits / self.TEMPERATURE, dim=-1),
            "row": torch.softmax(logits_row, dim=-1),
            "j":   j,
            "μ":   μ_proto
        }
        return u, misc
    
# -------------------------------------------------------------
class _BRNet(nn.Module):
    """One deterministic best-response sub-net for step k."""
    def __init__(self, game, spec):
        super().__init__()
        self.game = game
        hidden = spec["hidden"]
        self.FEAT_DIM = game.FEAT_DIM
        self.I = game.I
        self.ACTION_DIM  = game.ACTION_DIM
        self.net = nn.Sequential(
            nn.Linear(self.FEAT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, self.ACTION_DIM), nn.Tanh()
        )
    def forward(self, obs):
        # from setup import BOX_ACC
        z = torch.cat([obs["x"], self.game.belief_coord(obs["p"])], dim=-1)  # no t
        return self.net(z) * self.game.BOX_ACC

# -------------------------------------------------------------
class BR(nn.Module):
    """Wrapper with K independent best-response sub-nets."""
    def __init__(self, game, spec):
        super().__init__()
        self.subnets = nn.ModuleList([_BRNet(game, spec) for _ in range(game.K)])
    @torch.enable_grad()
    def forward(self, obs, k: int):
        return self.subnets[k](obs)                  # (B,ACTION_DIM)
    @torch.no_grad()
    def action_only(self, obs, k: int):
        u = self.forward(obs, k)
        return u, {"μ": u.detach()}
    

# ======================================================================
#  Explicit parametric strategies (no neural nets)
#  ─────────────────────────────────────────────────────────────────────
#  • One flat Parameter tensor per time-step
#  • Slices are looked-up by an information-state index (default 0 for now)
#  • Public API identical to the NN wrappers above:
#        u , misc = policy.action_only(obs, i_star, k)          # P1
#        u , misc = policy.action_only(obs, k)                  # P2
# ======================================================================

from typing import List
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------
#  Utility – encode public-message history to an integer index
# ---------------------------------------------------------------------

def info_index(history: List[int], I: int) -> int:
    idx = 0
    for j in history:
        idx = idx * I + j
    return idx

# ---------------------------------------------------------------------
#  Player‑1 explicit mixed strategy with collapsing capability
# ---------------------------------------------------------------------

class P1ExplicitStrategy(nn.Module):
    def __init__(self, game, *, init_scale=1e-2,
                 temperature=1.0, feat_eps=0.05):
        super().__init__()
        self.game = game                    # keep reference
        self.I, self.d, self.K = game.I, game.ACTION_DIM, game.K
        self.T, self.dev = temperature, game.device
        self.feat_eps = feat_eps

        block = self.I * (self.I + self.d)
        self.params, self._pure, self._centroid = nn.ParameterList(), [], []

        for t in range(1, self.K + 1):
            n_states = self.I ** (t - 1)
            dim = n_states * block
            self.params.append(
                nn.Parameter(init_scale * torch.randn(dim, device=self.dev))
            )
            default_pure = (t == self.K) or (t == self.K - 1)
            self._pure.append(
                torch.full((n_states,), default_pure,
                           dtype=torch.bool, device=self.dev)
            )
            # centroid: (n_states, FEAT_DIM)  – NaN means “unset”
            self._centroid.append(
                torch.full((n_states, game.FEAT_DIM),
                           float('nan'), device=self.dev)
            )
    # ------------- low‑level slice ------------------------------------
    def _slice(self, t: int, idx: int):
        block = self.I * (self.I + self.d)
        start = idx * block
        θ     = self.params[t]                       # flat tensor
        Λ_flat = θ[start : start + self.I * self.I]
        Λ      = Λ_flat.view(self.I, self.I)
        μ_flat = θ[start + self.I * self.I : start + block]
        μ_tbl  = μ_flat.view(self.I, self.d)
        return Λ, μ_tbl

    # --------------------------------------------------------------------
    def _feat(self, obs):
        """Return (B, FEAT_DIM) feature tensor for distance check."""
        return torch.cat([obs["x"],
                          self.game.belief_coord(obs["p"])], dim=-1)

    # ---------- main forward --------------------------------------------
    def forward(self, obs: dict[str, Tensor], k: int,
                history: list[int] | None = None):

        if history is None:
            history = []
        idx = 0
        for j in history:
            idx = idx * self.I + j

        # -------- determine parent purity ---------------------------------
        parent_pure = False
        if k > 0:
            parent_idx = idx // self.I          # divide by I → drop last digit
            parent_pure = self._pure[k-1][parent_idx].item()

        # -------- reversible purity check ---------------------------------
        if self._pure[k][idx]:
            # Only test distance if *parent is not pure*
            if not parent_pure:
                feat_now = self._feat(obs).mean(0)          # (FEAT_DIM,)
                cent     = self._centroid[k][idx]

                if torch.isnan(cent).any():
                    self._centroid[k][idx].copy_(feat_now)  # first visit
                elif torch.dist(feat_now, cent) > self.feat_eps:
                    # UN-COLLAPSE this node
                    self._pure[k][idx] = False

        # ---------- choose behaviour ------------------------------------
        if self._pure[k][idx]:
            # deterministic identity logits; broadcast first μ-row
            Λσ = torch.full((self.I, self.I), -50.0, device=self.dev)
            diag = torch.arange(self.I, device=self.dev)
            Λσ[diag, diag] = 50.0
            _, μ_tbl0 = self._slice(k, idx)
            μ_tbl = μ_tbl0[0].expand(self.I, -1)
        else:
            Λσ, μ_tbl = self._slice(k, idx)

        B = obs["x"].size(0)
        return {
            "A_logits": Λσ.expand(B, -1, -1).contiguous(),
            "μ":        μ_tbl.expand(B, -1, -1).contiguous()
        }

    # ---------- auto_collapse unchanged, but call after solver step -----
    # (use the bottom-up version you already integrated)

    # ------------------------------------------------------------------
    def action_only(self,
                    obs   : dict[str, Tensor],
                    i_star: Tensor,
                    k     : int,
                    history: List[int] | None = None):
        out        = self.forward(obs, k, history)
        A_logits   = out["A_logits"]
        μ_proto    = out["μ"]
        B          = A_logits.size(0)
        logits_row = A_logits[torch.arange(B, device=self.dev), i_star]
        j          = torch.argmax(logits_row, dim=-1)
        u          = μ_proto[torch.arange(B, device=self.dev), j]
        misc = {
            "A":   torch.softmax(A_logits / self.T, dim=-1),
            "row": torch.softmax(logits_row / self.T, dim=-1),
            "j":   j,
            "μ":   μ_proto
        }
        return u, misc

    # ------------------------------------------------------------------
    def auto_collapse(self, ent_thr: float = 1e-3) -> int:
        """Scan all nodes; mark as pure if each row ≈ one‑hot on diag."""
        new_cnt = 0
        for k in range(self.K):
            if self._pure[k].all():
                continue  # whole layer already collapsed
            n_states = self.I ** (k)
            block    = self.I * (self.I + self.d)
            θ        = self.params[k]
            for idx in range(n_states):
                if self._pure[k][idx]:
                    continue
                start   = idx * block
                Λ_flat  = θ[start : start + self.I * self.I]
                Λ       = Λ_flat.view(self.I, self.I)
                row_p   = torch.softmax(Λ / self.T, dim=-1)
                max_val, arg = row_p.max(dim=-1)
                unique = arg.unique().numel() == self.I
                diag_like = (max_val > 1 - ent_thr) & unique
                if diag_like.all():
                    self._pure[k][idx] = True
                    new_cnt += 1
        return new_cnt

    # ------------------------------------------------------------------
    def step_is_pure(self, k: int) -> bool:
        return self._pure[k].all().item()
