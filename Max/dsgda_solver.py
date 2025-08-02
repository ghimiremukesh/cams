"""
dsgda_solver.py
====================================
Define minimax solver for the game
"""

from __future__ import annotations
import math, itertools, time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor
from game import HexnerGame

# ── momentum buffer ───────────────────────────────────────────────────
class MomentumBuffer:
    def __init__(self, params, beta, device):
        self.params = list(params)
        self.m      = [torch.zeros_like(p, device=device) for p in self.params]
        self.beta   = beta

    def update(self):
        """Exponential moving average of the current gradients."""
        for m, p in zip(self.m, self.params):
            if p.grad is not None:
                m.mul_(self.beta).add_(p.grad, alpha=1.0 - self.beta)

    def clip_(self, C: float):
        """ℓ₂ clip each momentum vector to length ≤ C."""
        for m in self.m:
            n = m.norm()
            if n > C:
                m.mul_(C / n)

    def apply_step(self, ascent: bool, lr: float):
        """Gradient *ascent* if ascent=True, else descent."""
        sign = +1.0 if ascent else -1.0
        for p, m in zip(self.params, self.m):
            p.data.add_(m, alpha=sign * lr)

# ── main solver class ────────────────────────────────────────────────
class DSGDASolver:
    def __init__(self, game, p1, p2, spec):
        """
        Parameters
        ----------
        lr_p1    : learning rate for Player-1 parameters
        lr_p2    : learning rate for Player-2 parameters
        momentum : momentum coefficient (0.6 ≈ recommended by paper)
        C2_p1    : square of clipping radius for P1 momentum
        C2_p2    : square of clipping radius for P2 momentum
        """
        self.lr_p1    = spec["lr_p1"]
        self.lr_p2    = spec["lr_p2"]
        self.C1       = math.sqrt(spec["C2_p1"])
        self.C2       = math.sqrt(spec["C2_p2"])
        self.momentum = spec["momentum"]

        self.game = game
        self.spec = spec
        K = game.K; I = game.I; feat = game.FEAT_DIM

        self.p1 = p1.to(game.device)
        self.p2 = p2.to(game.device)

        # parameters & momentum
        self.p1_vars = [p for p in self.p1.parameters() if p.requires_grad]
        self.p2_vars = [p for p in self.p2.parameters() if p.requires_grad]
        self.buf_p1 = MomentumBuffer(self.p1_vars, self.momentum, self.game.device)
        self.buf_p2 = MomentumBuffer(self.p2_vars, self.momentum, self.game.device)

        self.device = game.device; self.I = I; self.K = K

    # ----------------------------------------------------------------─
    def _build_feat(self, obs):
        # concat state (x) and belief coordinate (first I-1 entries)
        return torch.cat([obs["x"], obs["p"][..., : self.game.BELIEF_DIM]], dim=-1)

    # ---------------------------------------------------------------------------
    #  Helper – generate and memoise all sequences  (S,K)  long tensor
    # ---------------------------------------------------------------------------
    def _sequences(self, K: int, I: int) -> Tensor:
        _SEQ_CACHE = {}
        if K not in _SEQ_CACHE:
            _SEQ_CACHE[K] = torch.tensor(
                list(itertools.product(range(I), repeat=K)),
                device=self.game.device, dtype=torch.long
            )                             # (S,K)
        return _SEQ_CACHE[K]

    # ---------------------------------------------------------------------------
    def exact_loss(self) -> Tensor:
        """
        Vectorised exact loss:
            • batch over all paths  S = I**K   (1024 for I=2,K=10)
            • two forward passes (one per type) instead of S.
        """
        seq = self._sequences(self.game.K, self.game.I)            # (S,K)
        S   = seq.size(0)
        total = torch.tensor(0., device=self.device)

        GameClass = self.game.__class__

        for i_star in range(self.I):
            prior_i = self.game.P0[0, i_star]

            # ----- create one big batched environment ------------------------
            env = GameClass(self.game.spec, batch_size=S)
            env.i_star.fill_(i_star)
            env.p.copy_(self.game.P0.repeat(S, 1))

            path_prob   = torch.ones(S, device=self.device)     # (S,)
            running_acc = torch.zeros(S, device=self.device)    # (S,)

            for k in range(self.K):
                j_k = seq[:, k]                           # (S,)

                obs = {"t": env.t, "x": env.x, "p": env.p}

                # P1 network for step k  (batched)
                out       = self.p1.forward(obs, k)
                A_logits  = out["A_logits"]               # (S,I,I)
                μ_proto   = out["μ"]                      # (S,I,2)

                # soft row probs and multiply into path_prob
                row_probs = torch.softmax(A_logits[:, i_star], dim=-1)  # (S,I)
                prob_j    = row_probs.gather(1, j_k.unsqueeze(1)).squeeze(1)
                path_prob = path_prob * prob_j

                # P1 & P2 continuous controls
                u1 = μ_proto[torch.arange(S, device=self.device), j_k]       # (S,2)
                u2 = self.p2.forward(obs, k)                      # (S,2)

                # dynamics + running cost
                env.step(u1, u2)
                running_acc = running_acc + env._running_loss(u1, u2)

                # Bayes update (vectorised)
                A_soft = torch.softmax(A_logits, dim=-1)
                env.p  = env._bayes_update(env.p, A_soft, j_k)

            # terminal cost
            L_paths = running_acc + env._terminal_loss()    # (S,)

            total += prior_i * (path_prob * L_paths).sum()

        return total    

    # ---------------------------------------------------------------------
    def step(self) -> dict[str, float]:
        """
        Perform one GDA iteration and return diagnostics.
        """
        # ---------- zero old grads ---------------------------------------
        for p in itertools.chain(self.p1_vars, self.p2_vars):
            if p.grad is not None:
                p.grad.zero_()

        # ---------- exact objective & back-prop --------------------------
        loss = self.exact_loss()      # scalar (P1 maximises)
        loss.backward()

        # ---------- EMA momentum update & ℓ₂ clip ------------------------
        self.buf_p1.update();  self.buf_p2.update()
        self.buf_p1.clip_(self.C1); self.buf_p2.clip_(self.C2)

        # ---------- gradient (a/des)cent step ----------------------------
        self.buf_p1.apply_step(ascent=False,  lr=self.lr_p1)   # maximise
        self.buf_p2.apply_step(ascent=True, lr=self.lr_p2)   # minimise

        # ---------- diagnostics -----------------------------------------
        g_p1 = torch.stack([m.norm() for m in self.buf_p1.m]).mean().item()
        g_p2 = torch.stack([m.norm() for m in self.buf_p2.m]).mean().item()

        return {"L": loss.item(),
                "g_p1": g_p1,
                "g_p2": g_p2}
