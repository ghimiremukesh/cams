"""
player.py
====================================
Define player
"""

import torch, torch.nn as nn
from torch import Tensor

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