from __future__ import annotations
"""dsgda_solver.py  – timing + logging version
────────────────────────────────────────────────────────
Adds fine‑grained wall‑clock timing for each phase of an iteration and
persists per‑epoch metadata to a JSONL file so your run is fully
reproducible.

New public attributes
---------------------
    • `self.log_path` : file path where JSON lines are appended
    • `self.meta`     : in‑memory list of dicts (optional)

Each call to `step()` returns a dict
    {
      'L'          : float,   # loss value
      'g_p1'       : float,
      'g_p2'       : float,
      'collapsed'  : int,
      't_loss'     : ms,
      't_backward' : ms,
      't_momentum' : ms,
      't_step'     : ms,
      't_collapse' : ms
    }

and writes the same record as a JSON line.
"""

import math, itertools, json, time, os, datetime
from typing import Dict, List
import torch
from torch import Tensor

# ---------------------------------------------------------------------
class MomentumBuffer:
    def __init__(self, params, beta, device):
        self.params = list(params)
        self.m      = [torch.zeros_like(p, device=device) for p in self.params]
        self.beta   = beta
    def update(self):
        for m, p in zip(self.m, self.params):
            if p.grad is not None:
                m.mul_(self.beta).add_(p.grad, alpha=1.0 - self.beta)
    def clip_(self, C: float):
        for m in self.m:
            n = m.norm()
            if n > C:
                m.mul_(C / n)
    def apply_step(self, ascent: bool, lr: float):
        s = +1.0 if ascent else -1.0
        for p, m in zip(self.params, self.m):
            p.data.add_(m, alpha=s * lr)

# ---------------------------------------------------------------------
class DSGDASolver:
    """Gradient-descent/ascent solver with optional tree pruning,
    timing instrumentation, JSONL logging, and lightweight checkpoints
    (model + optimiser state) that you can trigger from your training
    loop whenever you visualise.
    """
    def __init__(self, game, p1, p2, spec, *, log_dir: str = "runs", prune: bool = True):
        """Create solver and open JSONL log file."""
        self.lr_p1, self.lr_p2 = spec["lr_p1"], spec["lr_p2"]
        self.C1, self.C2       = math.sqrt(spec["C2_p1"]), math.sqrt(spec["C2_p2"])
        self.momentum          = spec["momentum"]

        self.game, self.p1, self.p2 = game, p1.to(game.device), p2.to(game.device)
        self.device, self.I, self.K = game.device, game.I, game.K

        self.p1_vars = [p for p in self.p1.parameters() if p.requires_grad]
        self.p2_vars = [p for p in self.p2.parameters() if p.requires_grad]
        self.buf_p1  = MomentumBuffer(self.p1_vars, self.momentum, self.device)
        self.buf_p2  = MomentumBuffer(self.p2_vars, self.momentum, self.device)

        # logging -------------------------------------------------------
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"run_{ts}.jsonl")
        self.meta: List[Dict] = []
        # file naming helpers ------------------------------------------
        self.run_stamp = os.path.basename(self.log_path)[4:-6]   # strip 'run_' + '.jsonl'
        self.ckpt_dir  = os.path.join(log_dir, f"ckpt_{self.run_stamp}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # toggle tree-pruning
        self.prune = prune
        # toggle tree‑pruning
        self.prune = prune

    # ------------- helper for pruned Cartesian product ----------------
    def _enumerate_paths(self, i_star: int) -> Tensor:
        """Return (S,K) LongTensor of allowed message sequences.
        If self.prune=False the full I^K grid is returned."""
        if not self.prune:
            full = torch.cartesian_prod(*[torch.arange(self.I, device=self.device)
                                          for _ in range(self.K)])  # (I^K, K)
            return full + 0 * i_star   # broadcast, keep grad free

        # pruning mode --------------------------------------------------
        active = [k for k in range(self.K) if not self.p1.step_is_pure(k)]
        if not active:
            return torch.full((1, self.K), i_star, dtype=torch.long, device=self.device)
        ranges = [torch.arange(self.I, device=self.device) for _ in active]
        grid = ranges[0].unsqueeze(1) if len(ranges) == 1 else torch.cartesian_prod(*ranges)
        paths = torch.full((grid.size(0), self.K), i_star, dtype=torch.long, device=self.device)
        paths[:, active] = grid
        return paths

    # ---------------------------------------------------------------
    def _reachable_masks(self):
        """
        Compute, for every layer k, a bool mask of length I**k indicating
        which nodes are reachable under the current collapse mask *and*
        pruning flag.  Works on CPU or CUDA without triggering advanced-
        indexing assertions.
        """
        device = self.device
        reachable = [torch.zeros_like(m, device=device) for m in self.p1._pure]

        for i_star in range(self.I):
            seq = self._enumerate_paths(i_star)           # (S, K) LongTensor on device
            S   = seq.size(0)

            # layer 0 (root) always reachable
            reachable[0][0] = True

            # running base-I history index
            idx = torch.zeros(S, dtype=torch.long, device=device)

            for k in range(1, self.K):                    # layers 1 … K-1
                idx = idx * self.I + seq[:, k-1]          # update in place
                # use unique() + index_fill_ to avoid duplicate-index bugs
                idx_u = idx.unique()
                reachable[k].index_fill_(0, idx_u, True)

        return reachable

    # ---------------------------------------------------------------
    def _count_active_p1(self):
        """
        Count *reachable* parameters of Player-1, distinguishing
            • non-pure nodes,
            • first-pure nodes,
            • deeper pure nodes.
        """
        I, d, K = self.I, self.game.ACTION_DIM, self.K
        block_nonpure = I * (I + d)
        block_first   = I * d
        block_deep    = d

        # ---------- reachable indices on CPU to avoid CUDA asserts ---
        reachable = [set() for _ in range(K)]
        reachable[0].add(0)
        for i_star in range(I):
            seq = self._enumerate_paths(i_star).cpu()      # (S,K)
            idx = 0
            for k in range(1, K):
                idx = idx * I + seq[:, k-1]
                reachable[k].update(idx.tolist())

        # ---------- parameter tally -----------------------------------
        active = 0
        for k in range(K):
            mask_pure = self.p1._pure[k].cpu()             # BoolTensor
            for idx in reachable[k]:
                if not mask_pure[idx]:                     # non-pure
                    active += block_nonpure
                else:
                    if k == 0:                             # root can’t be here
                        continue
                    parent_idx = idx // I
                    parent_pure = self.p1._pure[k-1][parent_idx].item()
                    active += block_deep if parent_pure else block_first
        return active

    # ------------------------------------------------------------------
    def exact_loss(self) -> torch.Tensor:
        """
        Compute the zero-sum objective exactly, summing over all message
        sequences that are still feasible under the current purity masks.
        Behaviour is controlled by `self.prune`:

            prune = False   → enumerate the full I^K Cartesian grid.
            prune = True    → call self._enumerate_paths(i★) which
                            removes branches below pure nodes.

        Returns
        -------
        torch.Tensor   scalar loss (P1 maximises, P2 minimises)
        """
        total   = torch.tensor(0.0, device=self.device)
        GameCls = self.game.__class__

        for i_star in range(self.I):                       # loop over types
            seq = self._enumerate_paths(i_star)            # (S, K) LongTensor
            S   = seq.size(0)
            prior_i = self.game.P0[0, i_star]

            # ----- batched environment ------------------------------------
            env = GameCls(self.game.spec, batch_size=S)
            env.i_star.fill_(i_star)
            env.p.copy_(self.game.P0.repeat(S, 1))         # public belief

            path_prob   = torch.ones(S, device=self.device)
            running_acc = torch.zeros(S, device=self.device)

            for k in range(self.K):
                j_k = seq[:, k]                            # (S,)

                # ---------- observation dict ------------------------------
                obs = {"t": env.t, "x": env.x, "p": env.p}

                # ---------- Player-1 --------------------------------------
                out       = self.p1.forward(obs, k)        # dict with "A_logits", "μ"
                A_logits  = out["A_logits"]                # (S, I, I)
                μ_proto   = out["μ"]                       # (S, I, d)

                rows      = torch.softmax(A_logits[:, i_star], dim=-1)  # (S, I)
                path_prob = path_prob * rows.gather(1, j_k.unsqueeze(1)).squeeze(1)

                u1 = μ_proto[torch.arange(S, device=self.device), j_k]  # (S, d)

                # ---------- Player-2 --------------------------------------
                u2 = self.p2.forward(obs, k)                              # (S, d)

                # ---------- dynamics & running loss -----------------------
                env.step(u1, u2)
                running_acc += env._running_loss(u1, u2)                  # new unified API

                # ---------- Bayesian update -------------------------------
                A_soft = torch.softmax(A_logits, dim=-1)
                env.p  = env._bayes_update(env.p, A_soft, j_k)

            # ---------- terminal cost & accumulate ------------------------
            L_paths = running_acc + env._terminal_loss()                   # (S,)
            total  += prior_i * (path_prob * L_paths).sum()

        return total

    # ------------------------------------------------------------------
    def step(self) -> Dict[str, float]:
        t0 = time.perf_counter()
        # zero grads ----------------------------------------------------
        for p in itertools.chain(self.p1_vars, self.p2_vars):
            if p.grad is not None:
                p.grad.zero_()

        # compute loss --------------------------------------------------
        t1 = time.perf_counter()
        loss = self.exact_loss()
        t2 = time.perf_counter()

        # backward ------------------------------------------------------
        loss.backward()
        t3 = time.perf_counter()

        # momentum update & clip ---------------------------------------
        self.buf_p1.update(); self.buf_p2.update()
        self.buf_p1.clip_(self.C1); self.buf_p2.clip_(self.C2)
        t4 = time.perf_counter()

        # param step ----------------------------------------------------
        self.buf_p1.apply_step(ascent=False, lr=self.lr_p1)
        self.buf_p2.apply_step(ascent=True,  lr=self.lr_p2)
        t5 = time.perf_counter()

        # auto collapse -------------------------------------------------
        n_collapse = 0
        if self.prune:
            n_collapse = self.p1.auto_collapse(ent_thr=1e-1)
        t6 = time.perf_counter()

        # grads norms ---------------------------------------------------
        g_p1 = torch.stack([m.norm() for m in self.buf_p1.m]).mean().item()
        g_p2 = torch.stack([m.norm() for m in self.buf_p2.m]).mean().item()

        # count active P1 parameters
        active = self._count_active_p1()
        
        rec = {
            "iter"      : len(self.meta),
            "L"         : loss.item(),
            "g_p1"      : g_p1,
            "g_p2"      : g_p2,
            "collapsed" : int(n_collapse),
            "n_p1_active" : active,         
            "t_loss"     : (t2 - t1)*1e3,
            "t_backward" : (t3 - t2)*1e3,
            "t_momentum" : (t4 - t3)*1e3,
            "t_step"     : (t5 - t4)*1e3,
            "t_collapse" : (t6 - t5)*1e3,
            "wall_ms"    : (t6 - t0)*1e3,
        }
        # -- keep in RAM & file ---------------------------------------
        self.meta.append(rec)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

                # ---------------------------------------------------------------
        return rec

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str | int):
        """Persist current parameters & momentum buffers.
        The file is written to  ckpt_RUNSTAMP/ckpt_{tag}.pt  ."""
        fname = os.path.join(self.ckpt_dir, f"ckpt_{tag}.pt")
        torch.save({
            "iter"     : tag,
            "p1_state" : self.p1.state_dict(),
            "p2_state" : self.p2.state_dict(),
            "buf_p1"   : [m.clone().cpu() for m in self.buf_p1.m],
            "buf_p2"   : [m.clone().cpu() for m in self.buf_p2.m],
            "spec"     : self.game.spec,
        }, fname)
        return fname
