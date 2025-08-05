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
    """
    One step-specific CAMS policy π₁ᵏ.
    If `last_layer=True` the subnet outputs *only* the I·d prototype
    actions; the logits are replaced by a fixed identity matrix.
    """
    def __init__(self, game, hidden: int, *, last_layer: bool = False):
        super().__init__()
        self.game        = game
        self.I           = game.I
        self.d           = game.ACTION_DIM
        self.last_layer  = last_layer             # NEW FLAG

        self.net = nn.Sequential(
            nn.Linear(game.FEAT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),        nn.ReLU()
        )

        if not last_layer:                                  # I×I logits
            self.logit_head = nn.Linear(hidden, self.I * self.I)

        # always keep the full μ-table (I prototypes)
        self.mu_head = nn.Linear(hidden, self.I * self.d)

        # pre-build identity-logit tensor on device for speed
        id_logits = torch.full((self.I, self.I), -50.0)
        id_logits[torch.arange(self.I), torch.arange(self.I)] = 50.0
        self.register_buffer("_ID", id_logits)

    # ---------------------------------------------------------
    def forward(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        h = self.net(torch.cat([obs["x"],
                                self.game.belief_coord(obs["p"])], dim=-1))

        if self.last_layer:
            A_logits = self._ID.expand(h.size(0), -1, -1)      # (B,I,I)
        else:
            A_logits = self.logit_head(h).view(-1, self.I, self.I)

        μ = self.game.BOX_ACC * torch.tanh(
                self.mu_head(h).view(-1, self.I, self.d))

        return {"A_logits": A_logits, "μ": μ}

# -------------------------------------------------------------
class CAMS_INFORMED(nn.Module):
    """Wrapper holding K independent sub-nets π₁ᵏ."""
    def __init__(self, game, spec):
        super().__init__()
        
        self.I, self.d = game.I, game.ACTION_DIM
        h = spec["hidden"]
        self.ent_thr   = spec["ent_thr_belief"]

        # --- new root parameters ------------------------------------
        init = spec.get("init_scale", 1e-2)
        self.root_logits = nn.Parameter(init * torch.randn(self.I, self.I))
        self.root_mu     = nn.Parameter(init * torch.randn(self.I, self.d))

        # --- k = 1 … K-1 still neural nets ---------------------------
        self.subnets = nn.ModuleList([
            _CAMSNet(game, h, last_layer=(k == game.K - 1))   # flag only on last
            for k in range(1, game.K)                         # exclude k=0 root
        ])

    # unchanged – keeps existing training / viz calls -----------
    def forward(self, obs, k):
        if k == 0:                                             # root step
            B = obs["x"].shape[0]
            A = self.root_logits.expand(B, -1, -1).contiguous()
            μ = self.root_mu.expand(B, -1, -1).contiguous()
            return {"A_logits": A, "μ": μ}
        else:                                                  # k ≥ 1
            return self.subnets[k-1](obs)

    @torch.no_grad()
    def action_only(self, obs, i_star, k):
        out  = self.forward(obs, k)
        Alog = out["A_logits"]                   # (B,I,I)
        μtbl = out["μ"]                          # (B,I,d)
        B    = Alog.size(0)

        # last layer ⇒ deterministic j = i_star
        if k == len(self.subnets) - 1:
            j = i_star
        else:
            logits_row = Alog[torch.arange(B), i_star]
            j = torch.argmax(logits_row, -1)

        u = μtbl[torch.arange(B), j]

        return u, {
            "A":   torch.softmax(Alog / self.TEMP, -1),
            "row": torch.softmax(Alog[torch.arange(B), i_star] / self.TEMP, -1),
            "j":   j,
            "μ":   μtbl
        }
    
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
        self.d = game.ACTION_DIM
        init = spec.get("init_scale", 1e-2)

        # explicit μ for root
        self.root_mu = nn.Parameter(init * torch.randn(game.ACTION_DIM))

        # k = 1 … K-1 small nets
        self.subnets = nn.ModuleList([_BRNet(game, spec) 
                                      for _ in range(game.K - 1)])

    @torch.enable_grad()
    def forward(self, obs, k: int):
        if k == 0:
            B = obs["x"].shape[0]
            return self.root_mu.expand(B, -1).contiguous()
        return self.subnets[k-1](obs)
    
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
                 temperature=1.0, ent_thr_belief=1e-3):
        super().__init__()
        self.game, self.I = game, game.I
        self.d, self.K    = game.ACTION_DIM, game.K
        self.T, self.dev  = temperature, game.device
        self.ent_thr      = ent_thr_belief            # entropy threshold

        block = self.I * (self.I + self.d)
        self.params = nn.ParameterList()
        for t in range(1, self.K + 1):
            dim = (self.I ** (t - 1)) * block
            self.params.append(
                nn.Parameter(init_scale * torch.randn(dim, device=self.dev))
            )

    # ---------- fast block-slicer -----------------------------------
    def _slice_blocks(self, k, idx_tensor):
        """Gather Λ and μ for a batch of history indices."""
        I, d, dev = self.I, self.d, self.dev
        block = I * (I + d)
        θ = self.params[k].view(-1)
        starts = idx_tensor * block                    # (S,)

        offs_Λ = torch.arange(I*I, device=dev)
        Λ_flat = θ[(starts[:, None] + offs_Λ).reshape(-1)]
        Λ_out  = Λ_flat.view(-1, I, I)                 # (S,I,I)

        offs_μ = torch.arange(I*I, I*(I+d), device=dev)
        μ_flat = θ[(starts[:, None] + offs_μ).reshape(-1)]
        μ_out  = μ_flat.view(-1, I, d)                 # (S,I,d)
        return Λ_out, μ_out

    # ---------- vectorised forward for solver ------------------------
    def forward_batch(self, obs, k, idx_tensor, belief, prune: bool = True):
        """
        Returns dict with A_logits (S,I,I) and μ (S,I,d).
        Purity rule:
            • if k == K-1           → treat as first-pure (I×d table)
            • elif entropy(belief)<ent_thr  → deeper-pure (one μ vector)
            • else                 → non-pure (full block)
        """

        # ---------- raw slices -------------------------------------------
        Λ_out, μ_out = self._slice_blocks(k, idx_tensor)

        I, d, dev = self.I, self.d, self.dev
        S         = idx_tensor.size(0)

        # ---------- 1) enforce identity on the *last* layer --------------
        if k == self.K - 1:
            id_logits = torch.full((I, I), -50.0, device=dev)
            id_logits[torch.arange(I), torch.arange(I)] = 50.0
            Λ_out[:] = id_logits        # all paths at layer K-1

            # NOTE: we **do not** collapse μ here – keep I×d prototypes

        # ---------- 2) optional deeper-pure collapse ---------------------
        if prune and k < self.K - 1:
            # belief entropy for each path
            ent = -(belief * (belief + 1e-12).log()).sum(-1)    # (S,)
            deeper_mask = ent < self.ent_thr                   # (S,) bool

            if deeper_mask.any():
                # (a) broadcast single μ
                μ_single = μ_out[deeper_mask, 0].unsqueeze(1)  # (S_p,1,d)
                μ_out[deeper_mask] = μ_single.expand(-1, I, -1)

                # (b) identity logits for those rows
                id_logits = torch.full((I, I), -50.0, device=dev)
                id_logits[torch.arange(I), torch.arange(I)] = 50.0
                Λ_out[deeper_mask] = id_logits

        return {"A_logits": Λ_out, "μ": μ_out}


    # ---------- single-path wrapper for visualisation ----------------
    def forward(self, obs, k, history=None):
        if history is None:
            history = []
        idx = 0
        for j in history:
            idx = idx * self.I + j
        idx_t = torch.tensor([idx], device=self.dev)
        belief = obs["p"][:1]
        out = self.forward_batch(obs, k, idx_t, belief, prune=False)
        return out

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
