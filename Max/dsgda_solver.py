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
from pathlib import Path
from util import _stamp

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
    def __init__(self, game, p1, p2, spec, *, log_root: str = "Max/runs", 
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

        # ============================================================
        #  ──  unified directory layout  ─────────────────────────────
        #  Max/runs/
        #        └── 2025-08-08_15-45-12/          ← self.run_dir
        #              ├── log.jsonl              ← self.log_path
        #              ├── ckpt/                  ← self.ckpt_dir
        #              └── anim/                  ← self.anim_dir
        # ============================================================
        stamp         = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.stamp    = stamp                     # expose in case user wants it
        self.run_dir  = Path(log_root).expanduser() / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = str((self.run_dir / "log.jsonl").resolve())
        self.ckpt_dir = str((self.run_dir / "ckpt").resolve())
        self.anim_dir = str((self.run_dir / "anim").resolve())
        Path(self.ckpt_dir).mkdir(exist_ok=True)
        Path(self.anim_dir).mkdir(exist_ok=True)

        self.meta: List[Dict] = []

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

        # --------- 2) restore paths under rows that jumped by > delta ----------
        restore = []
        for k in range(K - 1):                    # no restore at leaves
            ids_old  = self._row_prev_ids[k]
            rows_old = self._row_prev_rows[k]
            ids_new  = self._row_curr_ids[k]
            rows_new = self._row_curr_rows[k]
            if (ids_old is None) or (rows_old is None) or (ids_new is None) or (rows_new is None):
                continue
            # align old→new by node id using searchsorted (ids_new is sorted)
            pos_in_new = torch.searchsorted(ids_new, ids_old)
            valid = (pos_in_new >= 0) & (pos_in_new < ids_new.numel()) & (ids_new[pos_in_new] == ids_old)
            if valid.any():
                diff = (rows_new[pos_in_new[valid]] - rows_old[valid]).abs().sum(dim=1)
                jumped = valid.clone()
                jumped[valid] = diff > delta
                # expand all subtrees of the jumped nodes
                idxs_to_restore = ids_old[jumped]                        # (M,)
                if idxs_to_restore.numel() > 0:
                    depth = K - k - 1
                    # vectorised subgrid expansion for many prefixes
                    # (just loop; M is small; keeps code simple)
                    for prefix_idx in idxs_to_restore.tolist():
                        restore.append(self._expand_subgrid(prefix_idx, depth))

        paths_next = paths_keep
        if restore:
            paths_restore = torch.unique(torch.cat(restore, 0), dim=0)
            paths_next = torch.unique(torch.cat([paths_keep, paths_restore], 0), dim=0)

        # # -----------------------------------------------
        # # 3.  parameter count for P1   (belief-entropy rule)
        # # -----------------------------------------------
        # params = 0
        # ent_thr = self.p1.ent_thr

        # # build map layer→{idx→belief} from belief_curr cache
        # belief_layer = self.belief_curr      # list[K] of dicts{idx: Tensor(I,)}

        # # gather nodes reachable via paths_next
        # nodes = [set() for _ in range(K)]
        # for seq in paths_next.cpu():
        #     idx = 0
        #     for k in range(K):
        #         if k > 0:
        #             idx = idx * I + seq[k-1].item()
        #         nodes[k].add(idx)

        # for k in range(K):
        #     for idx in nodes[k]:
        #         if k == K - 1:
        #             params += I * d                          # last layer
        #             continue
        #         belief = belief_layer[k].get(idx)
        #         if belief is None:
        #             # safety: treat as non-pure
        #             params += I * (I + d)
        #             continue
        #         ent = -(belief * (belief + 1e-12).log()).sum().item()
        #         if ent < ent_thr:                            # deeper-pure
        #             params += d
        #         else:                                        # non-pure OR first-pure
        #             params += I * (I + d)

        # self.n_p1_active = params
        return paths_next.to(device)

    # -----------------------------------------------------------------
    def exact_loss(self, apply_prune) -> torch.Tensor:
        """
        Vectorised over types:
        • Evolve the env once over the current path set (batch = S).
        • Maintain per-seq, per-type path probabilities pi[:, i].
        • Compute terminal loss per type (cheap) on the final state.
        Also fills GPU caches used by pruning: _row_curr_ids/_row_curr_rows.
        """
        paths = self.paths                                  # (S,K)
        S, K  = paths.size()
        I, d  = self.I, self.game.ACTION_DIM
        dev   = self.device

        # fresh GPU caches for this iteration (pruning)
        self._row_curr_ids  = [None for _ in range(self.K)]
        self._row_curr_rows = [None for _ in range(self.K)]
        self._bel_curr_ids  = [None for _ in range(self.K)]
        self._bel_curr_p    = [None for _ in range(self.K)]
        self.prob_curr      = torch.zeros(S, device=dev)   # π(seq) = Σ_i p0[i] π(seq|i)

        # ── env over sequences only (dynamics do not depend on i★) ─────────
        env = self.game.__class__(self.game.spec, batch_size=S)
        # i_star set later only when computing terminal loss
        env.p.copy_(self.game.P0.repeat(S, 1))

        # per-sequence, per-type path prob
        pi = torch.ones(S, I, device=dev)                  # π(seq|i) for all i at once
        running = torch.zeros(S, device=dev)

        for k in range(K):
            # base-I history index idx(seq, k)
            if k == 0:
                idx = torch.zeros(S, dtype=torch.long, device=dev)
            else:
                coef = I ** torch.arange(k-1, -1, -1, device=dev)     # (k,)
                idx  = (paths[:, :k] * coef).sum(-1)                  # (S,)

            # group by node so P1 forward happens once per unique node
            sorted_idx, order = torch.sort(idx)                       # (S,), (S,)
            is_new = torch.ones_like(sorted_idx, dtype=torch.bool)
            is_new[1:] = sorted_idx[1:] != sorted_idx[:-1]
            node_starts = torch.nonzero(is_new, as_tuple=False).squeeze(1)  # (N_k,)
            uniq = sorted_idx[node_starts]                                   # (N_k,)
            rep_seq = order[node_starts]                                     # representative seq per node
            pos = torch.searchsorted(uniq, idx)                              # map each seq → its node

            # representative observations per node
            node_obs = {
                "t": env.t[rep_seq] if env.t.dim() else env.t,      # scalar-safe
                "x": env.x[rep_seq],                                # (N_k, …)
                "p": env.p[rep_seq],                                # (N_k, I)
            }

            # P1 forward once per node
            out_node = self.p1.forward(node_obs, k)                 # (N_k,I,I), (N_k,I,d)
            A_logits_node, μ_node = out_node["A_logits"], out_node["μ"]

            # force identity logits at last layer (but keep full μ table)
            if k == K - 1:
                id_logits = torch.full((I, I), -50.0, device=dev)
                id_logits[torch.arange(I, device=dev), torch.arange(I, device=dev)] = 50.0
                A_logits_node = id_logits.expand_as(A_logits_node)

            # optional deep-pure collapse (belief-entropy) per node (not last layer)
            if self.prune and k < K - 1:
                ent_node = -(node_obs["p"] * (node_obs["p"] + 1e-12).log()).sum(-1)  # (N_k,)
                deep_mask_node = ent_node < self.p1.ent_thr
                if deep_mask_node.any():
                    id_logits = torch.full((I, I), -50.0, device=dev)
                    id_logits[torch.arange(I, device=dev), torch.arange(I, device=dev)] = 50.0
                    A_logits_node[deep_mask_node] = id_logits
                    # broadcast a single μ across prototypes for deeper-pure nodes
                    μ_single = μ_node[deep_mask_node, 0].unsqueeze(1)             # (Ndeep,1,d)
                    μ_node[deep_mask_node] = μ_single.expand(-1, I, -1)

            # expand node outputs back to sequence order
            A_logits_seq = A_logits_node[pos]                        # (S,I,I)
            μ_proto      = μ_node[pos]                               # (S,I,d)

            # per-type row probs for chosen column j_k
            A_soft_seq = torch.softmax(A_logits_seq, dim=-1)         # (S,I,I)
            j_k = paths[:, k].view(S, 1, 1)                          # (S,1,1)
            probs = A_soft_seq.gather(2, j_k.expand(-1, I, 1)).squeeze(-1)  # (S,I)
            pi *= probs                                               # update π(seq|i) for all i

            # cache type-agnostic q-rows per node for pruning: q = p ⊤ A
            if apply_prune:
                q_node = torch.einsum('ni,nij->nj', node_obs["p"], torch.softmax(A_logits_node, dim=-1))  # (N_k,I)
                self._row_curr_ids[k]  = uniq
                self._row_curr_rows[k] = q_node
                self._bel_curr_ids[k]  = uniq
                self._bel_curr_p[k]    = node_obs["p"]

            # dynamics & running cost use the chosen prototype action (type-agnostic)
            j_seq = paths[:, k]                                      # (S,)
            u1    = μ_proto[torch.arange(S, device=dev), j_seq]      # (S,d)
            u2    = self.p2.forward({"t": env.t, "x": env.x, "p": env.p}, k)  # (S,d)

            env.step(u1, u2)
            env.p = env._bayes_update(env.p, A_soft_seq, j_seq)
            running += env._running_loss(u1, u2)

        # terminal loss does depend on i★; compute it cheaply for each type on final state
        prior = self.game.P0[0]                                      # (I,)
        total = torch.tensor(0.0, device=dev)
        for i_star in range(I):
            env.i_star.fill_(i_star)
            L_term = env._terminal_loss()                            # (S,)
            L_tot  = running + L_term                                # (S,)
            total += prior[i_star] * (pi[:, i_star] * L_tot).sum()
            # unconditional path probability for pruning
            self.prob_curr += prior[i_star] * pi[:, i_star]

        return total

    # ------------------------------------------------------------------
    def step(self) -> Dict[str, float]:
        t0 = _stamp(self.device)
        
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
        t1 = _stamp(self.device)
        loss = self.exact_loss(apply_prune)
        # rotate row caches (GPU tensors)
        self.prob_prev = self.prob_curr.detach()
        self._row_prev_ids  = [x if x is None else x.clone()  for x in self._row_curr_ids]
        self._row_prev_rows = [x if x is None else x.clone()  for x in self._row_curr_rows]

        t2 = _stamp(self.device)

        # backward ------------------------------------------------------
        loss.backward()
        t3 = _stamp(self.device)

        # momentum update & clip ---------------------------------------
        self.buf_p1.update(); self.buf_p2.update()
        self.buf_p1.clip_(self.C1); self.buf_p2.clip_(self.C2)
        t4 = _stamp(self.device)

        # param step ----------------------------------------------------
        self.buf_p1.apply_step(ascent=False, lr=self.lr_p1)
        self.buf_p2.apply_step(ascent=True,  lr=self.lr_p2)
        t5 = _stamp(self.device)

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
        fname = Path(self.ckpt_dir) / f"ckpt_{tag}.pt"
        torch.save({
            "iter"     : tag,
            "p1_state" : self.p1.state_dict(),
            "p2_state" : self.p2.state_dict(),
            "buf_p1"   : [m.clone().cpu() for m in self.buf_p1.m],
            "buf_p2"   : [m.clone().cpu() for m in self.buf_p2.m],
            "spec"     : self.game.spec,
            "meta_log"    : self.meta,
            "paths"    : self.paths.cpu(),
        }, fname)
        return fname
