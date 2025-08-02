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

"""explicit_strategy.py
────────────────────────────────────────
Flat-parameter representations for both players that now match the
problem specification exactly:

• **Each information-state σ stores**
    – Λσ ∈ ℝ^{I×I} (row-softmax logits)
    – μσ ∈ ℝ^{I×d} (one prototype action per announced column)
  so the parameter block size is  **I·(I + d)**.

• At step *t (1-based)* there are I^{t-1} information-states, giving a
  flat tensor length  *I^{t-1}·I·(I + d) = I^{t}·(I + d)*.

Public APIs unchanged:
    ▸ forward(obs, k[, history])  – returns dict with "A_logits", "μ"
    ▸ action_only(...)           – used by Game.rollout()

The helper `info_index(history, I)` encodes the public-message history
(j₁,…,j_{t−1}) into an integer 0 ≤ idx < I^{t−1}.
"""

from typing import List
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------
#  Utility – encode public-message history to an integer index
# ---------------------------------------------------------------------

def info_index(history: List[int], I: int) -> int:
    """Base-*I* encoding of the sequence (empty ⇒ 0)."""
    idx = 0
    for j in history:
        idx = idx * I + j
    return idx

# ---------------------------------------------------------------------
#  Player-1 explicit mixed strategy
# ---------------------------------------------------------------------

class P1ExplicitStrategy(nn.Module):
    """Explicit (Λσ , μσ) parameterisation for Player 1."""

    def __init__(self, game, *, init_scale: float = 1e-2, temperature: float = 1.0):
        super().__init__()
        self.I   = game.I
        self.d   = game.ACTION_DIM
        self.K   = game.K
        self.T   = temperature
        self.dev = game.device

        # one flat Parameter per time-step --------------------------------
        self.params = nn.ParameterList()
        for t in range(1, self.K + 1):           # t = 1…K  (1-based)
            n_states = self.I ** (t - 1)
            block    = self.I * (self.I + self.d)          # I² + I·d
            dim      = n_states * block
            self.params.append(nn.Parameter(init_scale * torch.randn(dim, device=self.dev)))

    # ------------------------------------------------------------------
    def _slice(self, t: int, idx: int):
        """Return (Λσ , μσ) views for step t (0-based) and info-state idx."""
        block = self.I * (self.I + self.d)
        start = idx * block
        θ     = self.params[t]                       # flat tensor

        Λ_flat = θ[start : start + self.I * self.I]
        Λ      = Λ_flat.view(self.I, self.I)         # (I,I)

        μ_flat = θ[start + self.I * self.I : start + block]   # (I·d,)
        μ_tbl  = μ_flat.view(self.I, self.d)                  # (I,d)
        return Λ, μ_tbl

    # ------------------------------------------------------------------
    def forward(self,
                obs: dict[str, Tensor],
                k: int,
                history: List[int] | None = None):
        """Match the signature of the old NN forward(obs, k)."""
        if history is None:
            history = []
        idx = info_index(history, self.I)
        Λσ, μ_tbl = self._slice(k, idx)                # (I,I), (I,d)

        B = obs["x"].shape[0]
        A_logits = Λσ.expand(B, -1, -1).contiguous()   # (B,I,I)
        μ        = μ_tbl.expand(B, -1, -1).contiguous()# (B,I,d)

        return {"A_logits": A_logits, "μ": μ}

    # ------------------------------------------------------------------
    def action_only(self,
                    obs   : dict[str, Tensor],
                    i_star: Tensor,
                    k     : int,
                    history: List[int] | None = None):
        """Autograd-compatible – **no `@torch.no_grad()`**."""
        out        = self.forward(obs, k, history)
        A_logits   = out["A_logits"]      # (B,I,I)
        μ_proto    = out["μ"]             # (B,I,d)
        B          = A_logits.size(0)

        logits_row = A_logits[torch.arange(B, device=self.dev), i_star]  # (B,I)
        j          = torch.argmax(logits_row, dim=-1)                    # (B,)
        u          = μ_proto[torch.arange(B, device=self.dev), j]        # (B,d)

        misc = {
            "A"  : torch.softmax(A_logits / self.T, dim=-1),
            "row": torch.softmax(logits_row / self.T, dim=-1),
            "j"  : j,
            "μ"  : μ_proto
        }
        return u, misc

# ---------------------------------------------------------------------
#  Player-2 explicit best response
# ---------------------------------------------------------------------

class P2ExplicitStrategy(nn.Module):
    """Deterministic BR: one μσ per information-state."""

    def __init__(self, game, *, init_scale: float = 1e-2):
        super().__init__()
        self.I   = game.I
        self.d   = game.ACTION_DIM
        self.K   = game.K
        self.dev = game.device

        self.params = nn.ParameterList()
        for t in range(1, self.K + 1):
            n_states = self.I ** (t - 1)
            dim      = n_states * self.d
            self.params.append(nn.Parameter(init_scale * torch.randn(dim, device=self.dev)))

    # ------------------------------------------------------------------
    def _slice(self, t: int, idx: int):
        start = idx * self.d
        return self.params[t][start : start + self.d]    # view (d,)

    # ------------------------------------------------------------------
    def forward(self,
                obs    : dict[str, Tensor],
                k      : int,
                history: List[int] | None = None):
        if history is None:
            history = []
        idx = info_index(history, self.I)
        μσ  = self._slice(k, idx)              # (d,)
        B   = obs["x"].shape[0]
        return μσ.expand(B, -1).contiguous()   # (B,d)

    # ------------------------------------------------------------------
    def action_only(self,
                    obs    : dict[str, Tensor],
                    k      : int,
                    history: List[int] | None = None):
        u = self.forward(obs, k, history)
        return u, {"μ": u}
