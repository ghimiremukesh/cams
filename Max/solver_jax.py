# dsgda_solver_jax.py
# =========================================================
# Clean DSGDA solver (JAX) for the football game.
# - No pruning, no tree visualisation.
# - Exact expectation over all public-message sequences (I^K).
# - Separate optimisers for P1 (descent) and P2 (ascent).
# - JSONL logging + periodic checkpoints compatible with your run layout.
# =========================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List
import os, json, time, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import serialization as flax_serial

from game_jax import FootballGame, FootballState
from player_jax import CAMSInformed, BR


# =========================================================
# Global switches
# =========================================================
F32 = jnp.float32
PROFILE_TIMES = True     # set False to skip timing fields (kept for parity)
DEBUG_INFO    = False    # print shapes and sanity stats

# Pruning controls
PRUNE = True
PRUNE_EVERY = 10
PRUNE_WARMUP = 100
EPS_FRACTION = 1e-3      # epsilon = EPS_FRACTION * (1/I)^K
DELTA_ROW_FRACTION = 1e-2  # delta = DELTA_ROW_FRACTION * (1/I)^K
SIZE_RATIO_GATE = 0.6      # accept new path set only if S_next <= SIZE_RATIO_GATE * S_curr


# =========================================================
# Helpers
# =========================================================
def _all_paths(I: int, K: int) -> jnp.ndarray:
    """Return tensor of shape (I**K, K) with base-I digit rows."""
    S = I ** K
    idx = jnp.arange(S, dtype=jnp.int32)
    cols = []
    for power in range(K - 1, -1, -1):
        cols.append(((idx // (I ** power)) % I))
    return jnp.stack(cols, axis=1).astype(jnp.int32)   # (S, K)


def _global_grad_norm(grads) -> float:
    sq = 0.0
    for g in jax.tree_util.tree_leaves(grads):
        sq = sq + jnp.sum(jnp.asarray(g, dtype=F32) ** 2)
    return float(jnp.sqrt(sq))

# ---- add near top of file ----
def _to_jsonable(obj):
    """Convert nested structures with jnp/np arrays to plain JSON types."""
    import numpy as _np
    import jax.numpy as _jnp
    from dataclasses import is_dataclass, asdict

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    # JAX arrays (DeviceArray)
    if isinstance(obj, (_jnp.ndarray,)):
        return _np.asarray(obj).tolist()
    # Fallback: string repr
    return str(obj)

# =========================================================
# Loss factory
# =========================================================
def make_loss_fn(
    game: FootballGame,
    p1_model: CAMSInformed,
    p2_model: BR,
    paths: jnp.ndarray,               # (S, K) int32
):
    I, K = game.I, game.K
    S = paths.shape[0]
    prior = jnp.asarray(game.P0[0], dtype=F32)  # (I,)

    def loss_fn(p1_params, p2_params) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        # ----- reset env batch = S sequences --------------------------
        state0, _ = game.reset(batch_size=S)      # PRNG not needed; deterministic lineup
        state = FootballState(
            x=state0.x, t=state0.t, p=state0.p,
            i_star=jnp.zeros((S,), dtype=jnp.int32),
            w_last=state0.w_last
        )

        row_ids_list = []   # length-K, each (S,)
        row_q_list   = []   # length-K, each (S,I)

        # path-type probabilities π(seq|i) for all i
        pi = jnp.ones((S, I), dtype=F32)
        running = jnp.zeros((S,), dtype=F32)

        # roll over K steps
        for k in range(K):
            obs = {"x": state.x, "p": state.p, "t": state.t}

            out = p1_model.apply({"params": p1_params}, obs, k)           # {"A_logits": (S,I,I), "μ": (S,I,d)}
            A_logits = out["A_logits"]
            mu_tbl   = out["μ"]                               # (S,I,d)

            # soft policy over columns
            A_soft = jax.nn.softmax(A_logits, axis=-1)        # (S,I,I)

            # per-sequence node id (base-I prefix index up to k)
            if k == 0:
                node_idx = jnp.zeros((S,), dtype=jnp.int32)
            else:
                coef = (I ** jnp.arange(k-1, -1, -1, dtype=jnp.int32)).astype(jnp.int32)
                node_idx = jnp.sum((paths[:, :k] * coef[None, :]), axis=1).astype(jnp.int32)
            row_ids_list.append(node_idx)

            # type-agnostic next-column distribution q = p^T softmax(A)
            q_row = jnp.einsum('si,sij->sj', state.p, A_soft)  # (S,I)
            row_q_list.append(q_row)

            # chosen column per sequence
            j_k = paths[:, k]                                 # (S,)
            j_idx = j_k.reshape(S, 1, 1)
            Aj = jnp.take_along_axis(A_soft, jnp.tile(j_idx, (1, I, 1)), axis=2).squeeze(-1)  # (S,I)
            pi = pi * Aj

            # type-agnostic prototype action for offence
            u1 = mu_tbl[jnp.arange(S), j_k]                   # (S,d)

            # defence action
            u2 = p2_model.apply({"params": p2_params}, obs, k)            # (S,d)

            # physics step
            state, p_tackle_now = game.step(state, u1, u2)
            running = running + game._running_loss(u1, u2, p_tackle_now)  # (S,)

            # public belief Bayes update
            state = FootballState(
                x=state.x, t=state.t,
                p=game._bayes_update(state.p, A_soft, j_k),
                i_star=state.i_star, w_last=state.w_last
            )

        # terminal part depends on i★ only through the terminal loss
        total = 0.0
        for i_star in range(I):
            state_i = FootballState(x=state.x, t=state.t, p=state.p,
                                    i_star=jnp.full((S,), i_star, dtype=jnp.int32),
                                    w_last=state.w_last)
            L_term = game._terminal_loss(state_i)             # (S,)
            L_tot  = running + L_term                         # (S,)
            total += prior[i_star] * jnp.sum(pi[:, i_star] * L_tot)

        prob_seq = jnp.sum(pi * prior[None, :], axis=1)  # (S,)
        aux = {
            "prob_seq": prob_seq,
            "row_ids_seq": row_ids_list,
            "row_q_seq": row_q_list,
        }
        return total, aux

    # JIT for speed; models captured as static Python objects in closure
    return jax.jit(loss_fn)


# =========================================================
# Solver
# =========================================================
@dataclass
class DSGDASpec:
    lr_p1: float = 3e-3
    lr_p2: float = 1e-2
    momentum: float = 0.6
    C2_p1: float = 10.0   # used as global-norm clip (via sqrt like PyTorch buffer clip)
    C2_p2: float = 10.0


class DSGDASolver:
    """
    JAX DSGDA with separate optax pipelines for P1 (descent) and P2 (ascent).
    Training-only; viz is handled externally on CPU.
    """
    def __init__(
        self,
        game: FootballGame,
        p1_model: CAMSInformed, p1_params,
        p2_model: BR,           p2_params,
        spec: Dict[str, Any],
        *,
        log_root: str = "Max/runs",
    ):
        self.game = game
        self.p1_model, self.p2_model = p1_model, p2_model
        self.p1_params, self.p2_params = p1_params, p2_params

        self.spec = DSGDASpec(**spec)
        self.I, self.K = game.I, game.K

        # Full message grid (I^K × K)
        self.paths = _all_paths(self.I, self.K)   # (S,K)
        self.S_paths = int(self.paths.shape[0])

        # Directories
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.stamp    = stamp
        self.run_dir  = Path(log_root).expanduser() / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "ckpt"
        self.anim_dir = self.run_dir / "anim"   # kept for parity; viz code may write here
        self.ckpt_dir.mkdir(exist_ok=True)
        self.anim_dir.mkdir(exist_ok=True)
        self.log_path = str((self.run_dir / "log.jsonl").resolve())

        self.meta: List[Dict[str, float]] = []

        # Optax transforms
        c1 = float(self.spec.C2_p1) ** 0.5
        c2 = float(self.spec.C2_p2) ** 0.5
        self.opt_p1 = optax.chain(
            optax.clip_by_global_norm(c1),
            optax.sgd(learning_rate=self.spec.lr_p1, momentum=self.spec.momentum, nesterov=False),
        )
        self.opt_p2 = optax.chain(
            optax.clip_by_global_norm(c2),
            optax.sgd(learning_rate=self.spec.lr_p2, momentum=self.spec.momentum, nesterov=False),
        )
        self.opt_state_p1 = self.opt_p1.init(self.p1_params)
        self.opt_state_p2 = self.opt_p2.init(self.p2_params)

        # JIT loss
        self._loss = make_loss_fn(self.game, self.p1_model, self.p2_model, self.paths)

        # Value+grads function
        self._vg = jax.jit(jax.value_and_grad(self._loss, argnums=(0, 1), has_aux=True))

        # ---- pruning state ----
        self.prune = PRUNE
        self.prune_every = PRUNE_EVERY
        self.prune_warmup = PRUNE_WARMUP
        a_min = (1.0 / self.I) ** self.K
        self.eps_prob = EPS_FRACTION * a_min
        self.delta_row = DELTA_ROW_FRACTION * a_min
        self.size_ratio_gate = SIZE_RATIO_GATE

        self.prob_prev = None   # np.ndarray (S,)
        self.row_prev_ids: List = [None for _ in range(self.K)]   # each np.ndarray (S,) of node ids
        self.row_prev_rows: List = [None for _ in range(self.K)]  # each np.ndarray (S,I) q-rows

    # -----------------------------------------------------
    def _expand_subgrid(self, prefix_idx: int, depth: int) -> jnp.ndarray:
        """Return (I**depth, K) sequences sharing the given base-I prefix.
        The prefix covers the first L = (K - depth) columns; the suffix covers the
        remaining `depth` columns. `prefix_idx` is the base-I encoding of that prefix.
        """
        I, K = self.I, self.K
        L = K - int(depth)
        # decode prefix digits from integer in base-I (without leading zeros)
        digs = []
        tmp = int(prefix_idx)
        while tmp > 0:
            digs.append(tmp % I)
            tmp //= I
        digs = digs[::-1]
        # pad with leading zeros to length L
        if L > 0:
            if len(digs) < L:
                digs = [0] * (L - len(digs)) + digs
            prefix_vec = jnp.asarray(digs[:L], dtype=jnp.int32)
        else:
            prefix_vec = jnp.zeros((0,), dtype=jnp.int32)

        # build suffix cartesian grid of shape (I**depth, depth)
        if depth > 0:
            grids = [jnp.arange(I, dtype=jnp.int32) for _ in range(depth)]
            mesh = jnp.stack(jnp.meshgrid(*grids, indexing='ij'), axis=-1).reshape(-1, depth)
        else:
            mesh = jnp.zeros((1, 0), dtype=jnp.int32)  # one row, zero columns

        S = mesh.shape[0]
        seq = jnp.zeros((S, K), dtype=jnp.int32)
        if L > 0:
            seq = seq.at[:, :L].set(jnp.broadcast_to(prefix_vec, (S, L)))
        if depth > 0:
            seq = seq.at[:, L:].set(mesh)  # fill the last `depth` columns
        return seq

    # -----------------------------------------------------
    def step(self) -> Dict[str, float]:
        if PROFILE_TIMES:
            t0 = time.perf_counter()

        # loss and grads
        (loss, aux), (gr_p1, gr_p2) = self._vg(self.p1_params, self.p2_params)

        if PROFILE_TIMES:
            t1 = time.perf_counter()

        # gradient norms (for logging)
        g1 = _global_grad_norm(gr_p1)
        g2 = _global_grad_norm(gr_p2)

        # Apply updates (P1 descent, P2 ascent → minimise -loss)
        updates_p1, self.opt_state_p1 = self.opt_p1.update(gr_p1, self.opt_state_p1, self.p1_params)
        self.p1_params = optax.apply_updates(self.p1_params, updates_p1)

        updates_p2, self.opt_state_p2 = self.opt_p2.update(
            jax.tree_map(lambda g: -g, gr_p2), self.opt_state_p2, self.p2_params
        )
        self.p2_params = optax.apply_updates(self.p2_params, updates_p2)

        if PROFILE_TIMES:
            t2 = time.perf_counter()

        # ---------------- pruning (host-side; applies to next iter) ----------------
        t_prune_ms = 0.0
        if self.prune:
            iter_idx = len(self.meta)
            should_check = (iter_idx >= self.prune_warmup) and ((iter_idx - self.prune_warmup) % self.prune_every == 0)
            if should_check:
                tpr0 = time.perf_counter()
                # aux caches are jax arrays; convert to numpy for grouping
                prob_curr = jnp.asarray(aux["prob_seq"]).astype(float)
                row_ids_seq = [jnp.asarray(a).astype(int) for a in aux["row_ids_seq"]]
                row_q_seq   = [jnp.asarray(a).astype(float) for a in aux["row_q_seq"]]

                # 1) keep paths with prob > eps (prefer current probs; fallback to prev only if shapes match)
                if (self.prob_prev is not None) and (
                    int(jnp.asarray(self.prob_prev).shape[0]) == int(self.paths.shape[0])
                ):
                    keep_mask = (jnp.asarray(self.prob_prev) > self.eps_prob)
                else:
                    keep_mask = (prob_curr > self.eps_prob)
                paths_keep = jnp.asarray(self.paths)[keep_mask]

                # 2) restore subtrees where node q-row changed a lot
                restores = []
                for k in range(self.K - 1):  # no restore at leaves
                    ids_new  = jnp.asarray(row_ids_seq[k])  # (S,)
                    rows_new = jnp.asarray(row_q_seq[k])    # (S,I)
                    ids_old  = self.row_prev_ids[k]
                    rows_old = self.row_prev_rows[k]
                    if ids_old is None or rows_old is None:
                        continue
                    # group by node using first occurrence as representative
                    # new
                    order_new = jnp.argsort(ids_new)
                    ids_new_sorted = ids_new[order_new]
                    is_new = jnp.concatenate([jnp.array([True]), ids_new_sorted[1:] != ids_new_sorted[:-1]])
                    rep_pos_new = jnp.where(is_new, size=is_new.shape[0])[0]
                    uniq_new = ids_new_sorted[is_new]
                    rep_rows_new = rows_new[order_new][is_new]
                    # old
                    ids_old_np = jnp.asarray(ids_old)
                    rows_old_np = jnp.asarray(rows_old)
                    order_old = jnp.argsort(ids_old_np)
                    ids_old_sorted = ids_old_np[order_old]
                    is_old = jnp.concatenate([jnp.array([True]), ids_old_sorted[1:] != ids_old_sorted[:-1]])
                    uniq_old = ids_old_sorted[is_old]
                    rep_rows_old = rows_old_np[order_old][is_old]
                    # align old→new by searchsorted
                    pos_in_new = jnp.searchsorted(uniq_new, uniq_old)
                    valid = (pos_in_new >= 0) & (pos_in_new < uniq_new.shape[0]) & (uniq_new[pos_in_new] == uniq_old)
                    if bool(jnp.any(valid)):
                        diff = jnp.sum(jnp.abs(rep_rows_new[pos_in_new[valid]] - rep_rows_old[valid]), axis=1)
                        jumped_ids = uniq_old[valid][diff > self.delta_row]
                        if jumped_ids.size > 0:
                            depth = self.K - (k + 1)
                            for pid in list(map(int, jnp.asarray(jumped_ids).tolist())):
                                restores.append(self._expand_subgrid(pid, depth))
                if len(restores) > 0:
                    paths_restore = jnp.unique(jnp.concatenate(restores, axis=0), axis=0)
                    paths_next = jnp.unique(jnp.concatenate([paths_keep, paths_restore], axis=0), axis=0)
                else:
                    paths_next = jnp.unique(paths_keep, axis=0)

                S_curr = int(self.paths.shape[0])
                S_next = int(paths_next.shape[0])
                accept = (S_next <= max(1, int(self.size_ratio_gate * S_curr))) and (S_next < S_curr)

                if accept:
                    # swap in new paths and rebuild jitted fns
                    self.paths = jnp.asarray(paths_next, dtype=jnp.int32)
                    self.S_paths = int(self.paths.shape[0])
                    self._loss = make_loss_fn(self.game, self.p1_model, self.p2_model, self.paths)
                    self._vg = jax.jit(jax.value_and_grad(self._loss, argnums=(0, 1), has_aux=True))
                    # reset caches; recompute on next iteration to avoid shape mismatch
                    self.prob_prev = None
                    for k2 in range(self.K):
                        self.row_prev_ids[k2]  = None
                        self.row_prev_rows[k2] = None
                if not accept:
                    # rotate caches for next decision (shapes unchanged)
                    self.prob_prev = jnp.asarray(prob_curr)
                    for k in range(self.K):
                        self.row_prev_ids[k]  = jnp.asarray(row_ids_seq[k])
                        self.row_prev_rows[k] = jnp.asarray(row_q_seq[k])

                t_prune_ms = (time.perf_counter() - tpr0) * 1e3

        rec = {
            "iter"      : len(self.meta),
            "L"         : float(loss),
            "g_p1"      : float(g1),
            "g_p2"      : float(g2),
            "n_seq"     : int(self.S_paths),
            # Timing (kept keys for plot_run compatibility)
            "t_prune"    : float(t_prune_ms),
            "t_loss"     : (t1 - t0) * 1e3 if PROFILE_TIMES else 0.0,
            "t_backward" : 0.0,                                 # fused in JAX
            "t_momentum" : (t2 - t1) * 1e3 if PROFILE_TIMES else 0.0,  # includes param updates
            "t_step"     : 0.0,
            "wall_ms"    : (t2 - t0) * 1e3 if PROFILE_TIMES else 0.0,
        }

        self.meta.append(rec)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

        if DEBUG_INFO and (rec["iter"] % 50 == 0):
            print(f"[{rec['iter']:04d}] L={rec['L']:+.4f}  g1={rec['g_p1']:.3f}  g2={rec['g_p2']:.3f}  S={rec['n_seq']}")

        return rec

    # -----------------------------------------------------
    def save_checkpoint(self, tag: str | int):
        """
        Write two files:
        • ckpt_{tag}.msgpack : binary Flax-serialized pytree
        • ckpt_{tag}.meta.json : small JSON with specs/paths count/etc.
        """
        # 1) Binary state (safe & fast for params/opt states / jax arrays)
        state_tree = {
            "iter": int(tag) if isinstance(tag, int) else tag,
            "p1_params": self.p1_params,
            "p2_params": self.p2_params,
            "opt_state_p1": self.opt_state_p1,
            "opt_state_p2": self.opt_state_p2,
            "paths": self.paths,            # (S,K) jnp.int32
            # Keep full training log in the JSONL file; no need to duplicate here.
        }
        ckpt_path = (self.ckpt_dir / f"ckpt_{tag}.msgpack").resolve()
        with open(ckpt_path, "wb") as fh:
            fh.write(flax_serial.to_bytes(state_tree))

        # 2) Human-friendly metadata (JSON-serializable only)
        meta = {
            "iter": state_tree["iter"],
            "stamp": self.stamp,
            "spec": _to_jsonable(self.spec),          # DSGDASpec dataclass
            "game_spec": _to_jsonable(self.game.spec),
            "S_paths": int(self.S_paths),
            "log_path": self.log_path,
        }
        meta_path = (self.ckpt_dir / f"ckpt_{tag}.meta.json").resolve()
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)

        return str(ckpt_path)