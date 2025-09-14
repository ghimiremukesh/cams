# dsgda_solver_jax_pruning.py
# =========================================================
# JAX DSGDA solver with multi-GPU support and tree pruning.
# - Implements a "Re-JIT" strategy: trains with a static set of paths,
#   periodically prunes paths on the host, and re-compiles the
#   pmapped function for the new, smaller set of paths.
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
from flax.training import common_utils
import flax.jax_utils as flax_utils

from game_jax import FootballGame, FootballState
from player_jax import CAMSInformed, BR


# =========================================================
# Global switches
# =========================================================
F32 = jnp.float32
PROFILE_TIMES = True


# =========================================================
# Helpers (Unchanged)
# =========================================================
def _all_paths(I: int, K: int) -> jnp.ndarray:
    S = I ** K
    idx = jnp.arange(S, dtype=jnp.int32)
    cols = []
    for power in range(K - 1, -1, -1):
        cols.append(((idx // (I ** power)) % I))
    return jnp.stack(cols, axis=1).astype(jnp.int32)

def _global_grad_norm(grads) -> float:
    sq_norms = [jnp.sum(jnp.asarray(g, dtype=F32) ** 2) for g in jax.tree.leaves(grads)]
    return float(jnp.sqrt(jnp.sum(jnp.array(sq_norms))))

def _to_jsonable(obj):
    import numpy as _np
    from dataclasses import is_dataclass, asdict
    if obj is None or isinstance(obj, (bool, int, float, str)): return obj
    if isinstance(obj, (list, tuple)): return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict): return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj): return _to_jsonable(asdict(obj))
    if isinstance(obj, (_np.ndarray, jnp.ndarray)): return _np.asarray(obj).tolist()
    return str(obj)


# =========================================================
# Loss function modified to return path probabilities
# =========================================================
def make_loss_fn(
    game: FootballGame,
    p1_model: CAMSInformed,
    p2_model: BR,
    paths: jnp.ndarray,
    per_device_micro_batch_size: int,
):
    I, K = game.I, game.K
    S = paths.shape[0]
    B_micro = per_device_micro_batch_size
    prior = jnp.asarray(game.P0[0], dtype=F32)

    def loss_fn(p1_params, p2_params, key: jax.random.KeyArray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        state0_batch, _ = game.reset(key=key, batch_size=B_micro)
        x_mega_batch = jnp.repeat(state0_batch.x, S, axis=0)
        p_mega_batch = jnp.repeat(state0_batch.p, S, axis=0)
        state = FootballState(
            x=x_mega_batch, t=jnp.zeros((B_micro * S,), dtype=F32),
            p=p_mega_batch, i_star=jnp.zeros((B_micro * S,), dtype=jnp.int32),
            w_last=jnp.zeros((B_micro * S, game.N, game.N), dtype=F32)
        )
        paths_mega_batch = jnp.tile(paths, (B_micro, 1))
        pi = jnp.ones((B_micro * S, I), dtype=F32)
        running = jnp.zeros((B_micro * S,), dtype=F32)

        for k in range(K):
            obs = {"x": state.x, "p": state.p, "t": state.t}
            out = p1_model.apply({"params": p1_params}, obs, k)
            A_logits, mu_tbl = out["A_logits"], out["μ"]
            A_soft = jax.nn.softmax(A_logits, axis=-1)
            j_k = paths_mega_batch[:, k]
            u1 = mu_tbl[jnp.arange(B_micro * S), j_k]
            u2 = p2_model.apply({"params": p2_params}, obs, k)
            j_idx = j_k.reshape(B_micro * S, 1, 1)
            Aj = jnp.take_along_axis(A_soft, jnp.tile(j_idx, (1, I, 1)), axis=2).squeeze(-1)
            pi *= Aj
            state, p_tackle_now = game.step(state, u1, u2)
            running += game._running_loss(u1, u2, p_tackle_now)
            state = state.tree_replace(p=game._bayes_update(state.p, A_soft, j_k))

        expected_L_per_tree = jnp.zeros((B_micro,), dtype=F32)
        for i_star_val in range(I):
            state_i = state.tree_replace(i_star=jnp.full((B_micro*S,), i_star_val, dtype=jnp.int32))
            L_term = game._terminal_loss(state_i)
            L_tot = running + L_term
            pi_i = pi[:, i_star_val]
            contrib_i = (pi_i * L_tot).reshape(B_micro, S)
            expected_L_per_tree += prior[i_star_val] * jnp.sum(contrib_i, axis=1)

        total_loss = jnp.mean(expected_L_per_tree)
        
        # Calculate path probabilities for pruning
        prob_seq_per_tree = jnp.sum(pi * prior[None, :], axis=1).reshape(B_micro, S)
        avg_prob_seq = jnp.mean(prob_seq_per_tree, axis=0) # Average across the micro-batch
        
        return total_loss, {"loss": total_loss, "avg_prob_seq": avg_prob_seq}

    return loss_fn


# =========================================================
# Solver with Pruning Logic
# =========================================================
@dataclass
class PruningSpec:
    prune_every: int = 100
    prune_warmup: int = 1000
    eps_fraction: float = 1e-4
    size_ratio_gate: float = 0.95

class DSGDASolver:
    def __init__(
        self,
        game: FootballGame,
        p1_model: CAMSInformed, p1_params,
        p2_model: BR,           p2_params,
        solver_spec: Dict[str, Any],
        pruning_spec: Dict[str, Any],
        *,
        log_root: str = "Max/runs",
        seed: int = 0,
        global_batch_size: int = 256,
        per_device_micro_batch_size: int = 32,
    ):
        self.game = game
        self.p1_model, self.p2_model = p1_model, p2_model
        self.key = jax.random.PRNGKey(seed)

        self.solver_spec = DSGDASpec(**solver_spec)
        self.pruning_spec = PruningSpec(**pruning_spec)
        self.I, self.K = game.I, game.K
        
        self.num_devices = jax.device_count()
        self.global_batch_size = global_batch_size
        self.per_device_micro_batch_size = per_device_micro_batch_size
        
        self.per_device_batch_size = self.global_batch_size // self.num_devices
        self.n_accum_steps = self.per_device_batch_size // self.per_device_micro_batch_size

        # Initial full set of paths
        self.paths = _all_paths(self.I, self.K)
        
        # Pruning state
        a_min = (1.0 / self.I) ** self.K
        self.eps_prob = self.pruning_spec.eps_fraction * a_min
        self.prob_prev = None # Cache for path probabilities

        # --- Logging and Checkpointing Setup ---
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.stamp, self.run_dir = stamp, Path(log_root).expanduser() / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir, self.anim_dir = self.run_dir / "ckpt", self.run_dir / "anim"
        self.ckpt_dir.mkdir(exist_ok=True); self.anim_dir.mkdir(exist_ok=True)
        self.log_path = str((self.run_dir / "log.jsonl").resolve())
        self.meta: List[Dict[str, float]] = []

        # --- Initial pmapped function creation ---
        self._rebuild_pmapped_step()

        # --- Replicate model parameters and optimizer state ---
        self.p1_params = flax_utils.replicate(p1_params)
        self.p2_params = flax_utils.replicate(p2_params)

        spec = self.solver_spec
        c1, c2 = float(spec.C2_p1) ** 0.5, float(spec.C2_p2) ** 0.5
        self.opt_p1 = optax.chain(optax.clip_by_global_norm(c1), optax.sgd(spec.lr_p1, spec.momentum))
        self.opt_p2 = optax.chain(optax.clip_by_global_norm(c2), optax.sgd(spec.lr_p2, spec.momentum))
        
        self.opt_state_p1 = flax_utils.replicate(self.opt_p1.init(p1_params))
        self.opt_state_p2 = flax_utils.replicate(self.opt_p2.init(p2_params))

    def _rebuild_pmapped_step(self):
        """Rebuilds and re-JITs the core training step function.
        This is called initially and after every pruning event."""
        loss_fn_micro = make_loss_fn(
            self.game, self.p1_model, self.p2_model, self.paths,
            per_device_micro_batch_size=self.per_device_micro_batch_size
        )
        
        def train_step(p1_params, p2_params, opt_state_p1, opt_state_p2, key):
            if key.ndim > 1: key = jnp.squeeze(key, axis=0)
            
            init_grads_p1 = jax.tree.map(jnp.zeros_like, p1_params)
            init_grads_p2 = jax.tree.map(jnp.zeros_like, p2_params)
            # Accumulator now also holds avg_prob_seq
            init_val = (init_grads_p1, init_grads_p2, 0.0, jnp.zeros_like(self.paths[:,0], dtype=F32), key)

            def accum_step_fn(i, val):
                grads_acc_p1, grads_acc_p2, total_loss, total_probs, loop_key = val
                loop_key, micro_batch_key = jax.random.split(loop_key)
                
                (loss_micro, aux), (gr_p1_micro, gr_p2_micro) = jax.value_and_grad(
                    loss_fn_micro, argnums=(0, 1), has_aux=True
                )(p1_params, p2_params, micro_batch_key)
                
                total_loss += loss_micro
                total_probs += aux["avg_prob_seq"]
                grads_acc_p1 = jax.tree.map(lambda acc, new: acc + new, grads_acc_p1, gr_p1_micro)
                grads_acc_p2 = jax.tree.map(lambda acc, new: acc + new, grads_acc_p2, gr_p2_micro)
                
                return (grads_acc_p1, grads_acc_p2, total_loss, total_probs, loop_key)

            final_grads_p1, final_grads_p2, total_loss, total_probs, _ = jax.lax.fori_loop(
                0, self.n_accum_steps, accum_step_fn, init_val
            )
            
            final_loss = total_loss / self.n_accum_steps
            avg_probs = total_probs / self.n_accum_steps
            final_grads_p1 = jax.tree.map(lambda g: g / self.n_accum_steps, final_grads_p1)
            final_grads_p2 = jax.tree.map(lambda g: g / self.n_accum_steps, final_grads_p2)
            
            final_loss = jax.lax.pmean(final_loss, axis_name='batch')
            avg_probs = jax.lax.pmean(avg_probs, axis_name='batch')
            final_grads_p1 = jax.lax.pmean(final_grads_p1, axis_name='batch')
            final_grads_p2 = jax.lax.pmean(final_grads_p2, axis_name='batch')
            
            updates_p1, new_opt_state_p1 = self.opt_p1.update(final_grads_p1, opt_state_p1, p1_params)
            new_p1_params = optax.apply_updates(p1_params, updates_p1)
            updates_p2, new_opt_state_p2 = self.opt_p2.update(jax.tree.map(lambda g: -g, final_grads_p2), opt_state_p2, p2_params)
            new_p2_params = optax.apply_updates(p2_params, updates_p2)

            g1_norm = _global_grad_norm(final_grads_p1)
            g2_norm = _global_grad_norm(final_grads_p2)

            # Return avg_probs to the host for the pruning decision
            return new_p1_params, new_p2_params, new_opt_state_p1, new_opt_state_p2, final_loss, g1_norm, g2_norm, avg_probs

        self._pmapped_step = jax.pmap(train_step, axis_name='batch')

    def step(self) -> Dict[str, float]:
        t0 = time.perf_counter()
        
        self.key, *step_keys = jax.random.split(self.key, self.num_devices + 1)
        sharded_keys = common_utils.shard(jnp.array(step_keys))
        
        (self.p1_params, self.p2_params, 
         self.opt_state_p1, self.opt_state_p2, 
         loss, g1, g2, avg_probs) = self._pmapped_step(
            self.p1_params, self.p2_params, self.opt_state_p1, self.opt_state_p2, sharded_keys
        )

        t1 = time.perf_counter()
        
        # --- Host-side Pruning Logic ---
        t_prune_ms = 0.0
        iter_idx = len(self.meta)
        spec = self.pruning_spec
        should_check = (iter_idx >= spec.prune_warmup) and ((iter_idx - spec.prune_warmup) % spec.prune_every == 0)

        if should_check:
            tpr0 = time.perf_counter()
            # Use probabilities from the previous step to make the pruning decision for the next step
            if self.prob_prev is not None:
                keep_mask = (self.prob_prev > self.eps_prob)
                paths_next = self.paths[keep_mask]
                
                S_curr = self.paths.shape[0]
                S_next = paths_next.shape[0]

                # Accept the prune if it removes paths and doesn't shrink too aggressively
                accept = (S_next < S_curr) and (S_next > 1) and \
                         (S_next >= int(spec.size_ratio_gate * S_curr))
                
                if accept:
                    print(f"\n[Pruning] Iter {iter_idx}: Activated. Path count {S_curr} -> {S_next}. Re-compiling...")
                    self.paths = paths_next
                    self._rebuild_pmapped_step() # Trigger re-compilation
                    self.prob_prev = None # Reset cache after re-JIT
                else:
                    # If not accepted, just update the cache for the next decision
                    self.prob_prev = flax_utils.unreplicate(avg_probs)
            else:
                 self.prob_prev = flax_utils.unreplicate(avg_probs)
            
            t_prune_ms = (time.perf_counter() - tpr0) * 1e3
        else:
            # Always update the probability cache if we are in the pruning phase
            if iter_idx >= spec.prune_warmup:
                self.prob_prev = flax_utils.unreplicate(avg_probs)


        # --- Logging ---
        loss_host = float(flax_utils.unreplicate(loss))
        g1_host = float(flax_utils.unreplicate(g1))
        g2_host = float(flax_utils.unreplicate(g2))
        
        rec = {"iter": iter_idx, "L": loss_host, "g_p1": g1_host, "g_p2": g2_host,
               "wall_ms": (t1 - t0) * 1e3 if PROFILE_TIMES else 0.0,
               "t_prune_ms": t_prune_ms,
               "S_paths": self.paths.shape[0]}
        self.meta.append(rec)
        with open(self.log_path, "a") as fh: fh.write(json.dumps(rec) + "\n")
        return rec

    def save_checkpoint(self, tag: str | int):
        state_tree = {"iter": int(tag) if isinstance(tag, int) else tag,
                      "p1_params": flax_utils.unreplicate(self.p1_params), 
                      "p2_params": flax_utils.unreplicate(self.p2_params),
                      "opt_state_p1": flax_utils.unreplicate(self.opt_state_p1), 
                      "opt_state_p2": flax_utils.unreplicate(self.opt_state_p2),
                      "paths": self.paths}
        ckpt_path = (self.ckpt_dir / f"ckpt_{tag}.msgpack").resolve()
        with open(ckpt_path, "wb") as fh:
            fh.write(flax_serial.to_bytes(state_tree))

        meta = {"iter": state_tree["iter"], "stamp": self.stamp, 
                "solver_spec": _to_jsonable(self.solver_spec),
                "pruning_spec": _to_jsonable(self.pruning_spec),
                "game_spec": _to_jsonable(self.game.spec), "S_paths": self.paths.shape[0],
                "log_path": self.log_path}
        meta_path = (self.ckpt_dir / f"ckpt_{tag}.meta.json").resolve()
        with open(meta_path, "w") as fh: json.dump(meta, fh)
        return str(ckpt_path)
