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
    def __init__(self, game, p1, p2, spec, *, log_dir: str = "runs", 
                 prune: bool = True,
                 prune_every=10,      # run pruning once every N steps
                 prune_warmup=0,      # skip the first W iterations
                 ):
        """Create solver and open JSONL log file."""
        self.lr_p1, self.lr_p2 = spec["lr_p1"], spec["lr_p2"]
        self.C1, self.C2       = math.sqrt(spec["C2_p1"]), math.sqrt(spec["C2_p2"])
        self.momentum          = spec["momentum"]

        self.game, self.p1, self.p2 = game, p1.to(game.device), p2.to(game.device)
        self.device, self.I, self.K, self.d = game.device, game.I, game.K, game.ACTION_DIM

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

        # pruning parameters
        self.prune      = prune
        a_min = (1 / self.I) ** self.K          # note: exponent K
        self.eps_prob  = 1e-3 * a_min
        self.delta_row = 1e-2 * a_min
        self.prune_every   = prune_every
        self.prune_warmup  = prune_warmup

        # ---------- full grid -----------------------------------------
        grid = torch.cartesian_prod(
            *[torch.arange(self.I) for _ in range(self.K)]
        )                                                    # (I^K, K) CPU
        self.paths      = grid.to(self.device).long()        # start with full
        self.prob_prev  = torch.ones(grid.size(0),
                                     device=self.device)     # π_old
        # row cache: list[layer] -> dict{idx(int): row_prob (I,)}
        self.row_prev   = [dict() for _ in range(self.K)]

        self.belief_curr = [dict() for _ in range(self.K)]
        self.row_curr    = [dict() for _ in range(self.K)]

    # -----------------------------------------------------------------
    def _expand_subgrid(self, prefix_idx: int, depth: int):
        """Return a (I**depth , K) tensor of sequences that share
           the given prefix index (encoded base-I) up to layer `layer`."""
        I, K = self.I, self.K
        # decode prefix to list of digits (length = layer)
        digits = []
        tmp = prefix_idx
        while tmp:
            digits.append(tmp % I)
            tmp //= I
        digits = digits[::-1]
        layer = len(digits)
        # cartesian for suffix
        suffix = torch.cartesian_prod(
            *[torch.arange(I, device=self.device) for _ in range(depth)]
        )
        S = suffix.size(0)
        seq = torch.empty((S, K), dtype=torch.long, device=self.device)
        if layer:
            seq[:, :layer] = torch.tensor(digits,
                                          device=self.device).expand(S, -1)
        seq[:, layer:] = suffix
        return seq

    # -----------------------------------------------------------------
    def _build_paths_next(self):
        """
        Build the pruned path tensor for the next iteration.

        Side-effects
        ------------
        • self.n_p1_active  ←  exact # of P1 parameters still in graph
        Returns
        -------
        Tensor  (S_next, K)  pruned path set on solver.device
        """
        I, d, K = self.I, self.game.ACTION_DIM, self.K
        eps, delta = self.eps_prob, self.delta_row
        device = self.device

        # --------- 1. keep paths whose previous prob > eps ----------------
        keep_prob_mask = self.prob_prev > eps
        paths_keep = self.paths[keep_prob_mask]            # (S_keep,K)

        # --------- 2. restore paths under rows that jumped by > delta -----
        restore = []
        for k in range(K - 1):                              # no leaves
            for idx, row_old in self.row_prev[k].items():
                row_new = self.row_curr[k].get(idx)
                if row_new is None:
                    continue
                if torch.sum(torch.abs(row_new - row_old)) > delta:
                    depth = K - k - 1
                    restore.append(self._expand_subgrid(idx, depth))
        if restore:
            paths_restore = torch.unique(torch.cat(restore, 0), dim=0)
            paths_next = torch.unique(torch.cat([paths_keep,
                                                paths_restore], 0), dim=0)
        else:
            paths_next = paths_keep

        # -----------------------------------------------
        # 3.  parameter count for P1   (belief-entropy rule)
        # -----------------------------------------------
        params = 0
        ent_thr = self.p1.ent_thr

        # build map layer→{idx→belief} from belief_curr cache
        belief_layer = self.belief_curr      # list[K] of dicts{idx: Tensor(I,)}

        # gather nodes reachable via paths_next
        nodes = [set() for _ in range(K)]
        for seq in paths_next.cpu():
            idx = 0
            for k in range(K):
                if k > 0:
                    idx = idx * I + seq[k-1].item()
                nodes[k].add(idx)

        for k in range(K):
            for idx in nodes[k]:
                if k == K - 1:
                    params += I * d                          # last layer
                    continue
                belief = belief_layer[k].get(idx)
                if belief is None:
                    # safety: treat as non-pure
                    params += I * (I + d)
                    continue
                ent = -(belief * (belief + 1e-12).log()).sum().item()
                if ent < ent_thr:                            # deeper-pure
                    params += d
                else:                                        # non-pure OR first-pure
                    params += I * (I + d)

        self.n_p1_active = params
        return paths_next.to(device)

    # -----------------------------------------------------------------
    def exact_loss(self, apply_prune) -> torch.Tensor:
        """
        Exact objective over the currently kept path set `self.paths`.

        *  self.paths   : (S,K) LongTensor of message sequences
        *  self.row_curr     caches outgoing row distributions per node
        *  self.belief_curr  caches belief vectors per node
        *  self.prob_curr    stores  π(seq)  = Σ_i  p(i) π(seq|i)   (for pruning)
        """
        paths = self.paths                       # (S,K)
        S, K  = paths.size()
        I, d  = self.I, self.game.ACTION_DIM
        dev   = self.device

        # fresh caches for this iteration
        self.row_curr     = [dict() for _ in range(K)]
        self.belief_curr  = [dict() for _ in range(K)]
        self.prob_curr    = torch.zeros(S, device=dev)

        total = torch.tensor(0.0, device=dev)
        
        for i_star in range(I):                                    # loop hidden type
            seq = paths                                            # (S,K)
            env = self.game.__class__(self.game.spec, batch_size=S)
            env.i_star.fill_(i_star)
            env.p.copy_(self.game.P0.repeat(S, 1))

            pi = torch.ones(S, device=dev)                        # path prob p(seq|i★)
            running = torch.zeros(S, device=dev)

            for k in range(K):
                # ------------------------------------------------------------------
                # 1) observation and P1 forward (one subnet per layer)
                # ------------------------------------------------------------------
                obs = {"t": env.t, "x": env.x, "p": env.p}
                out = self.p1.forward(obs, k)                          # NEW
                A_logits, μ_proto = out["A_logits"], out["μ"]          # (S,I,I), (S,I,d)

                # ------------------------------------------------------------------
                # 2) optional deep-pure collapse (only when pruning step is active)
                # ------------------------------------------------------------------
                if self.prune and k < K - 1:                          # never for last layer
                    ent = -(env.p * (env.p + 1e-12).log()).sum(-1)     # (S,)
                    deep_mask = ent < self.p1.ent_thr                  # (S,) bool
                    if deep_mask.any():
                        id_logits = self.p1.subnets[k]._ID             # (I,I) buffer
                        A_logits[deep_mask] = id_logits                # detach – no grad
                        μ_single = μ_proto[deep_mask, 0].unsqueeze(1)  # (Sd,1,d)
                        μ_proto[deep_mask] = μ_single.expand(-1, self.I, -1)

                # ------------------------------------------------------------------
                # 3) row probabilities & path probability update
                # ------------------------------------------------------------------
                rows = torch.softmax(A_logits[:, i_star], dim=-1)      # (S,I)
                j_k  = seq[:, k]                                       # (S,)
                pi   = pi * rows.gather(1, j_k.unsqueeze(1)).squeeze(1)

                # ------------------------------------------------------------------
                # 4) cache row / belief for pruning statistics (once per node)
                # ------------------------------------------------------------------
                if apply_prune:
                    # node key = base-I integer of history up to layer k
                    coef = self.I ** torch.arange(k, -1, -1, device=dev)
                    node_key = (seq[:, :k+1] * coef).sum(-1)           # (S,)
                    uniq, inv = torch.unique(node_key, return_inverse=True)
                    self.row_curr[k].update({
                        int(u.item()): rows[inv == i][0].detach().cpu()
                        for i, u in enumerate(uniq)
                    })
                    self.belief_curr[k].update({
                        int(u.item()): env.p[inv == i][0].detach().cpu()
                        for i, u in enumerate(uniq)
                    })

                # ------------------------------------------------------------------
                # 5) dynamics and running cost
                # ------------------------------------------------------------------
                u1 = μ_proto[torch.arange(S, device=dev), j_k]         # (S,d)
                u2 = self.p2.forward(obs, k)                           # (S,d)
                env.step(u1, u2)
                env.p = env._bayes_update(env.p,
                                        torch.softmax(A_logits, -1), j_k)
                running += env._running_loss(u1, u2)                   # (S,)

            # ---------- terminal cost & accumulation ------------------------
            L_paths = running + env._terminal_loss()                          # (S,)
            prior   = self.game.P0[0, i_star]
            total  += prior * (pi * L_paths).sum()

            # accumulate unconditional path probability for pruning
            self.prob_curr += prior * pi

        return total

    # ------------------------------------------------------------------
    def step(self) -> Dict[str, float]:
        t0 = time.perf_counter()
        
        # ------- build new path set before computing loss --------------
        apply_prune = (
            self.prune and
            (len(self.meta) >= self.prune_warmup) and
            ((len(self.meta) - self.prune_warmup) % self.prune_every == 0)
        )

        if apply_prune:
            self.paths = self._build_paths_next()

        S_paths = self.paths.size(0) 
        # self.n_p1_active = self.I * (self.I + self.d) * ((self.I**(self.K-1)) - 1) + self.I * self.d * (self.I**(self.K-1))  # full grid

        # zero grads ----------------------------------------------------
        for p in itertools.chain(self.p1_vars, self.p2_vars):
            if p.grad is not None:
                p.grad.zero_()

        # compute loss --------------------------------------------------
        t1 = time.perf_counter()
        loss = self.exact_loss(apply_prune)
        # ---------- rotate caches --------------------------------------
        self.prob_prev = self.prob_curr.detach()
        self.row_prev  = self.row_curr

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

        # grads norms ---------------------------------------------------
        g_p1 = torch.stack([m.norm() for m in self.buf_p1.m]).mean().item()
        g_p2 = torch.stack([m.norm() for m in self.buf_p2.m]).mean().item()

        rec = {
            "iter"      : len(self.meta),
            "L"         : loss.item(),
            "g_p1"      : g_p1,
            "g_p2"      : g_p2,
            "n_seq"     : S_paths,
            "t_prune"    : (t1 - t0)*1e3,         
            "t_loss"     : (t2 - t1)*1e3,
            "t_backward" : (t3 - t2)*1e3,
            "t_momentum" : (t4 - t3)*1e3,
            "t_step"     : (t5 - t4)*1e3,
            "wall_ms"    : (t5 - t0)*1e3,
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
