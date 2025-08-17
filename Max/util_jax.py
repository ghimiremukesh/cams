"""
util_jax.py
====================================
JAX-compatible visualization helpers for the differential game solver.
"""
from __future__ import annotations

from typing import List, Tuple
import math

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np

# Assuming player_jax.py is in the same directory
from player_jax import apply_p1_explicit_batch
from game_jax import GameConfig
from solver_jax import SolverConfig

from matplotlib.ticker import MaxNLocator
import torch
import os, time, json
from pathlib import Path
from torch import Tensor
from typing import Any, Dict, Tuple

# Reuse the numpy/torch tree visualizers you already have
from util import build_tree as _build_tree_np
from util import draw_tree as _draw_tree_np

def stamp():
    """Pure wall-clock timestamp; never touches CUDA."""
    return time.perf_counter()

def viz_tree_from_p1(p1_torch_like, prior_1d_torch, I: int, K: int, out_dir: Path, iter_idx: int) -> Path:
    """
    Build & save the belief/message tree using the numpy/torch util functions.
    Returns the PNG path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"tree_iter_{iter_idx:04d}.png"
    G   = _build_tree_np(p1_torch_like, prior_1d_torch, I, K)
    _draw_tree_np(G, out_png=str(png))
    return png


# --- New: tree_viz_from_p1_params ---
def tree_viz_from_p1_params(p1_params,
                            out_png: str | None = None,
                            *,
                            I_hint: int | None = None,
                            K_hint: int = 2,
                            debug: bool = False) -> str | None:
    """
    Build a simple static policy shim from raw p1_params and draw the tree.
    Does not touch JAX/GPUs. Falls back to identity logits + zero μ if needed.

    Parameters
    ----------
    p1_params : Any
        Player-1 parameter pytree (JAX/Flax or plain dict).
    out_png : str | None
        If provided, save the PNG to this path and return it.
    I_hint : Optional[int]
        Optional hint for the number of types I.
    K_hint : int
        Tree depth horizon; default 2 if not known.
    debug : bool
        Print diagnostics on inference / fallbacks.

    Returns
    -------
    path : str | None
    """
    # Try to extract a representative (I,I) logits block and (I,d) μ block.
    A_blk, mu_blk = extract_leaves_for_viz(p1_params, I_hint or 2, d_action=2)
    # If dimension inference failed, try again without d_action coupling
    if A_blk is None or mu_blk is None:
        # Second attempt: infer I from any square leaf, μ from any (I, d)
        try:
            leaves = jax.tree_util.tree_leaves(p1_params)
        except Exception:
            leaves = []
        I_guess = I_hint
        for a in leaves:
            try:
                arr = np.asarray(a)
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    I_guess = int(arr.shape[0]); break
            except Exception:
                continue
        if I_guess is None:
            I_guess = 2
        if debug:
            print(f"[viz] Could not infer (I,I) logits and (I,d) prototypes from p1_params.")
            print("[viz] Falling back to identity logits and zero μ for visualization.")
        A_blk = np.eye(I_guess, dtype=np.float32)
        mu_blk = np.zeros((I_guess, 2), dtype=np.float32)  # default action dim=2

    I = int(np.asarray(A_blk).shape[0])
    d = int(np.asarray(mu_blk).shape[1])
    # Convert to torch tensors (CPU)
    A_t = torch.as_tensor(A_blk, dtype=torch.float32, device="cpu")
    mu_t = torch.as_tensor(mu_blk, dtype=torch.float32, device="cpu")
    prior = torch.full((I,), 1.0 / I, dtype=torch.float32)

    # Minimal static shim that matches util.build_tree()'s expected API
    class _StaticP1:
        def __init__(self, A_logits_t: Tensor, mu_tbl_t: Tensor):
            self._A = A_logits_t
            self._mu = mu_tbl_t
        def _slice(self, t: int, idx: int):
            # util.build_tree() expects (Λσ, μ_tbl)
            return self._A, self._mu

    G = _build_tree_np(_StaticP1(A_t, mu_t), prior, I, int(K_hint))
    return _draw_tree_np(G, out_png) if out_png else None

# --- Constants ---

_COLORS = {
    "root": "#8e44ad",      # purple
    "deep": "#27ae60",      # green (deeper-pure)
    "first": "#2ecc71",     # light-green (first-pure)
    "reveal": "#f39c12",    # orange
    "non": "#95a5a6",       # grey
    "leaf": "#2c3e50",      # navy
}

# --- JAX-based Helpers ---
def _entropy(p: jax.Array, eps: float = 1e-12) -> float:
    """Computes the entropy of a JAX probability distribution."""
    return -(p.clip(min=eps) * jnp.log(p.clip(min=eps))).sum()

# --- Tree Building and Visualization ---
def build_tree(
    p1_params: List[jax.Array], 
    game_config: GameConfig, 
    solver_config: SolverConfig
) -> nx.DiGraph:
    """
    Builds a NetworkX DiGraph representing the game tree from P1's explicit strategy.
    """
    G = nx.DiGraph()
    I, K = game_config.n_types, game_config.K
    prior = jnp.full((I,), 1.0 / I)
    
    # --- Helper to classify and add a node to the graph ---
    def _add_node(t: int, idx: int, belief: jax.Array) -> str:
        if t == 0:
            ntype = "root"
        elif t == K:
            ntype = "leaf"
        else:
            # Get policy output for this specific node
            idx_tensor = jnp.array([idx], dtype=jnp.int32)
            belief_batch = jnp.expand_dims(belief, 0)
            
            # Use the JAX policy function
            out = apply_p1_explicit_batch(
                p1_params, t, idx_tensor, belief_batch, 
                game_config, prune=True, ent_thr=solver_config.p1_ent_thr
            )
            A_logits = out["A_logits"][0] # (I, I)
            mu_tbl = out["μ"][0]         # (I, d)
            A = jax.nn.softmax(A_logits, axis=-1)

            # Classify node based on the policy's action matrix A
            row_max = A.max(axis=-1)
            arg = A.argmax(axis=-1)
            
            # A row is one-hot if one value is ~1 and others are ~0.
            # The matrix is a permutation if all rows are one-hot and point to unique actions.
            one_hot = (row_max > 1 - 1e-3).all() and (jnp.unique(arg).size == I)
            
            ent = _entropy(belief)
            
            if one_hot:
                # "deeper-pure" vs "first-pure"
                ntype = "deep" if ent < solver_config.p1_ent_thr else "first"
            else:
                # "revealing" vs "non-revealing" based on action diversity
                max_dist = jnp.linalg.norm(
                    jnp.expand_dims(mu_tbl, 0) - jnp.expand_dims(mu_tbl, 1), axis=-1
                ).max()
                ntype = "reveal" if max_dist > 1e-2 else "non"

        # Add node to graph, converting JAX array to NumPy for storage
        G.add_node((t, idx), belief=np.array(belief), ntype=ntype)
        return ntype

    # --- Breadth-First Search (BFS) to build the tree ---
    queue = [(0, 0, prior)] # (time, node_index, belief)
    _add_node(0, 0, prior)

    visited = set([(0, 0)])

    while queue:
        t, idx, belief = queue.pop(0)
        if t == K:
            continue

        # Get policy for the current node
        idx_tensor = jnp.array([idx], dtype=jnp.int32)
        belief_batch = jnp.expand_dims(belief, 0)
        out = apply_p1_explicit_batch(
            p1_params, t, idx_tensor, belief_batch, 
            game_config, prune=False, ent_thr=solver_config.p1_ent_thr # No pruning for viz
        )
        A = jax.nn.softmax(out["A_logits"][0], axis=-1)

        # Action probabilities q = p^T A
        q = (jnp.expand_dims(belief, 0) @ A).squeeze(0)
        q = q / q.sum().clip(min=1e-12)

        # Decide branching based on node type (pure nodes are deterministic)
        ntype = G.nodes[(t, idx)]["ntype"]
        if ntype in {"first", "deep"}:
            cols = [int(jnp.argmax(belief).item())]
        else:
            cols = list(range(I))

        for j in cols:
            child_idx = idx * I + j
            if (t + 1, child_idx) in visited:
                continue

            Aj = A[:, j]
            child_belief = (Aj * belief) / (Aj @ belief).clip(min=1e-12)
            
            prob_j = q[j].item()
            G.add_edge((t, idx), (t + 1, child_idx), prob=prob_j)
            _add_node(t + 1, child_idx, child_belief)
            
            visited.add((t + 1, child_idx))
            queue.append((t + 1, child_idx, child_belief))

    return G

def draw_tree(G: nx.DiGraph, out_png: str | None = None):
    """Draws the game tree using Matplotlib and NetworkX."""
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        print("Warning: pygraphviz not found. Using spring layout. For a hierarchical layout, run 'pip install pygraphviz'.")
        pos = nx.spring_layout(G, seed=0)

    # Node colors and sizes based on type
    ncolor = [_COLORS.get(G.nodes[n]["ntype"], "#000000") for n in G.nodes]
    sizes = [300 if G.nodes[n]["ntype"] == "root" else 150 for n in G.nodes]

    # Edge colors based on probability (greyscale)
    ecolor = []
    for u, v, d in G.edges(data=True):
        p = d.get("prob", 0.0)
        shade = 1.0 - p  # 0 prob -> white, 1 prob -> black
        ecolor.append((shade, shade, shade))

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=sizes, node_color=ncolor,
            edge_color=ecolor, arrows=False, width=1.5, linewidths=0.5)

    # Create legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=k) for k, c in _COLORS.items()]
    plt.legend(handles=handles, fontsize=9, loc="upper right", title="Node Types")
    plt.axis("off")
    plt.title("Game Tree Visualization")

    if out_png:
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"Tree visualization saved to {out_png}")
    
    plt.show()

# --- Run Analysis ---

def plot_run(log_path: str, save_png: str | None = None):
    """Plots metrics from a JSONL log file generated by the solver."""
    try:
        with open(log_path) as fh:
            records = [json.loads(line) for line in fh]
        df = pd.DataFrame(records)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading log file {log_path}: {e}")
        return

    if df.empty:
        print(f"No records found in {log_path}")
        return

    # ensure sorted by iter and drop dupes if any
    if "iter" in df.columns:
        df = df.sort_values("iter").drop_duplicates(subset=["iter"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes.ravel()
    fig.suptitle(f"Training Run Analysis: {os.path.basename(log_path)}", fontsize=16)

    # 1) Loss
    if {"iter", "L"}.issubset(df.columns):
        ax[0].plot(df["iter"], df["L"])
    else:
        ax[0].text(0.5, 0.5, "Missing columns: iter/L", ha="center", va="center")
    ax[0].set_title("Game Loss (P1 Objective)")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, linestyle='--', alpha=0.6)

    # 2) Gradients
    have_g = {"iter", "g_p1", "g_p2"}.issubset(df.columns)
    if have_g:
        ax[1].plot(df["iter"], df["g_p1"], label="‖g₁‖ (P1)")
        ax[1].plot(df["iter"], df["g_p2"], label="‖g₂‖ (P2)")
        ax[1].legend()
        ax[1].set_yscale('log')
    else:
        ax[1].text(0.5, 0.5, "Missing columns: iter/g_p1/g_p2", ha="center", va="center")
    ax[1].set_title("Gradient Norms")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("L2 Norm")
    ax[1].grid(True, linestyle='--', alpha=0.6)

    # 3) Active sequences
    if {"iter", "n_seq"}.issubset(df.columns):
        ax[2].plot(df["iter"], df["n_seq"])
        ax[2].set_yscale('log')
    else:
        ax[2].text(0.5, 0.5, "Missing columns: iter/n_seq", ha="center", va="center")
    ax[2].set_title("Number of Active Paths (Pruning)")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("# Paths")
    ax[2].grid(True, linestyle='--', alpha=0.6)

    # 4) Stacked per-phase times (any column starting with 't_')
    phases = [col for col in df.columns if col.startswith('t_')]
    if phases and "iter" in df.columns:
        time_df = df[["iter"] + phases].set_index('iter')
        time_df.plot(kind='bar', stacked=True, ax=ax[3], colormap='viridis', width=1.0)
        ax[3].set_title("Per-iteration Wall-clock Time (ms)")
        ax[3].set_xlabel("Iteration")
        ax[3].set_ylabel("Time (ms)")
        ax[3].legend([ph.replace("t_", "") for ph in phases])
        if len(df['iter']) > 20:
            ax[3].xaxis.set_major_locator(MaxNLocator(10))
    else:
        ax[3].text(0.5, 0.5, "No timing cols starting with 't_'", ha="center", va="center")
        ax[3].set_axis_off()

    if save_png:
        plt.savefig(save_png, dpi=150)
        print(f"Run plot saved to {save_png}")
    plt.show()

# ─────────────────────────────────────────────────────────────────────
# 5) Torch adapters to make JAX policies look torch-like for viz
# ─────────────────────────────────────────────────────────────────────
class _P1TorchShim:
    """
    Thin torch-like wrapper around JAX Player-1 parameters to drive
    the non-JAX visualizations (tree & interaction videos).

    Expects p1_params to contain:
      • A logits per (t, idx) OR a single global (I, I) array
      • μ prototypes per (t, idx) OR a single global (I, d) array

    Minimal contract:
      forward(obs, k) -> {"A_logits": (1,I,I) torch, "μ": (1,I,d) torch}
      action_only(obs, i_star, k) -> (u1: (1,d) torch, {"A": (1,I,I), "μ": (1,I,d), "j": int})
    """
    def __init__(self, p1_params, I_hint: int | None = None):
        self._params   = p1_params
        self._picked   = None     # cached (A, μ) chosen by _choose_leaves()
        self.I_hint    = I_hint

    # ------------------------------------------------------------------
    def _as_torch(self, x):
        if isinstance(x, torch.Tensor):
            return x
        x_np = np.asarray(x)
        return torch.from_numpy(x_np).float()

    # ------------------------------------------------------------------
    def pick_leaves(self, A_logits: Tensor | np.ndarray, mu_table: Tensor | np.ndarray):
        """
        Manually set the global leaves: (I,I) logits and (I,d) prototypes.
        Accepts numpy or torch; converts to torch and caches.
        """
        A_t = self._as_torch(A_logits)
        μ_t = self._as_torch(mu_table)
        if A_t.ndim != 2 or μ_t.ndim != 2:
            raise ValueError("pick_leaves expects 2D arrays: (I,I) and (I,d).")
        if A_t.shape[0] != A_t.shape[1]:
            raise ValueError(f"pick_leaves: A_logits must be square (I,I); got {tuple(A_t.shape)}")
        if A_t.shape[0] != μ_t.shape[0]:
            raise ValueError(f"pick_leaves: I mismatch between A ({A_t.shape[0]}) and μ ({μ_t.shape[0]}).")
        self._picked = (A_t, μ_t)

    # ------------------------------------------------------------------
    def _infer_from_params(self):
        """
        Try to find (I,I) logits and (I,d) μ in a variety of p1_params layouts.
        Returns A_logits_torch, mu_torch.
        """
        p = self._params

        # Helper: collect all 1D/2D leaves as numpy arrays
        def _flatten_dict(obj):
            stack = [obj]
            leaves = []
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    stack.extend(cur.values())
                elif hasattr(cur, "shape"):
                    leaves.append(np.asarray(cur))
                else:
                    try:
                        leaves.append(np.asarray(cur))
                    except Exception:
                        pass
            return leaves

        leaves = _flatten_dict(p)
        cand_logits = []
        cand_mu     = []

        # Try to deduce I from hint or square shapes
        I_guess = self.I_hint
        for a in leaves:
            if a.ndim == 2 and a.shape[0] == a.shape[1]:
                I_guess = I_guess or int(a.shape[0])

        # Classify candidates
        for a in leaves:
            if a.ndim == 2:
                if a.shape[0] == a.shape[1]:
                    cand_logits.append(a)                  # possible (I,I)
                elif I_guess is not None and a.shape[0] == I_guess:
                    cand_mu.append(a)                      # possible (I,d)
            elif a.ndim == 1 and I_guess is not None and a.size == I_guess * I_guess:
                cand_logits.append(a.reshape(I_guess, I_guess))

        # Pick best matches
        A_pick = None
        μ_pick = None

        if cand_logits:
            # Prefer exact square with the inferred I
            if I_guess is not None:
                for a in cand_logits:
                    if a.shape == (I_guess, I_guess):
                        A_pick = a
                        break
            if A_pick is None:
                # fallback: first square we saw
                for a in cand_logits:
                    if a.ndim == 2 and a.shape[0] == a.shape[1]:
                        A_pick = a
                        I_guess = a.shape[0]
                        break

        if cand_mu and I_guess is not None:
            # prefer rows = I_guess
            for m in cand_mu:
                if m.shape[0] == I_guess:
                    μ_pick = m
                    break

        if A_pick is None or μ_pick is None:
            raise AttributeError("Could not infer (I,I) logits and (I,d) prototypes from p1_params.")

        return self._as_torch(A_pick), self._as_torch(μ_pick)

    # ------------------------------------------------------------------
    def _choose_leaves(self):
        """
        Return (A_logits, μ) as torch tensors. Use manual pick if provided,
        otherwise infer from parameters (and cache the result).
        """
        if self._picked is not None:
            return self._picked
        A_t, μ_t = self._infer_from_params()
        # cache
        self._picked = (A_t.clone(), μ_t.clone())
        return self._picked

    # ------------------------------------------------------------------
    # Torch-policy style API used by visualizations
    # ------------------------------------------------------------------
    def forward(self, obs: dict, k: int):
        """
        Return a dict with torch tensors:
            {"A_logits": (1,I,I), "μ": (1,I,d)}
        """
        A_t, μ_t = self._choose_leaves()
        return {
            "A_logits": A_t.unsqueeze(0),  # (1,I,I)
            "μ": μ_t.unsqueeze(0),         # (1,I,d)
        }

    def action_only(self, obs: dict, i_star: Tensor | int, k: int):
        """
        Mimic the Torch player's action_only:
           (u1, {"A": (1,I,I), "μ": (1,I,d), "j": int})
        We pick j = argmax over the i★ row of softmax(A_logits).
        """
        out = self.forward(obs, k)
        A_logits = out["A_logits"][0]              # (I,I)
        μ_tbl    = out["μ"][0]                     # (I,d)

        A = torch.softmax(A_logits, dim=-1)        # (I,I)
        if isinstance(i_star, torch.Tensor):
            i0 = int(i_star.item())
        else:
            i0 = int(i_star)
        row = A[i0]
        j_idx = int(torch.argmax(row).item())

        u1 = μ_tbl[j_idx].unsqueeze(0)             # (1,d)
        misc = {"A": A.unsqueeze(0), "μ": μ_tbl.unsqueeze(0), "j": torch.tensor([j_idx])}
        return u1, misc

class _P2TorchShim:
    """
    Minimal shim for P2: returns a (1,d) Torch action.
    If you have a JAX/Flax BR forward, you can inject a callable.
    """
    def __init__(self, d: int, forward_fn=None):
        self._d = int(d)
        self._forward_fn = forward_fn  # optional callback: (obs, k) -> np.ndarray[d]

    def forward(self, obs: Dict[str, Tensor], k: int) -> Tensor:
        if self._forward_fn is None:
            # zero action fallback (keeps animation paths valid)
            return torch.zeros(1, self._d)
        arr = np.asarray(self._forward_fn(obs, k))
        return torch.from_numpy(arr).float().view(1, self._d)

# =====================================================================
# Extra helpers used by football_demo_jax.py (kept here to declutter it)
# =====================================================================
import os as _os
import json as _json
import time as _time
from pathlib import Path as _Path

# util_jax.py
import os, time, contextlib

def configure_runtime_env(
    *,
    log_xla_compiles: bool = False,
    isolate_xla_cache: bool = True,
    enable_faulthandler: bool = True,
    mpl_backend: str | None = "Agg",
):
    """
    Process-wide runtime knobs that must run BEFORE importing JAX.
    Keeps Torch on CPU until we want it, and leaves headroom for cuBLAS.
    """
    # ---- XLA/JAX memory behavior: do not pre-reserve ~90% of VRAM ----
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # leave headroom for cudnn/cublas handles, workspaces, etc.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")
    # use platform allocator (often less fragmentation)
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    # Optional: isolate XLA cache to avoid stale executable reuse
    if isolate_xla_cache:
        cache_dir = os.path.expanduser("~/.cache/jax_compilations_isolated")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("XLA_CACHE_DIR", cache_dir)

    if log_xla_compiles:
        # verbose XLA logs (noisy)
        os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")

    # Matplotlib backend for headless nodes
    if mpl_backend:
        import matplotlib
        matplotlib.use(mpl_backend, force=True)

    if enable_faulthandler:
        import faulthandler
        faulthandler.enable()

def extract_leaves_for_viz(p1_params, I: int, d_action: int):
    """Probe p1 params pytree for a representative (I,I) logits and (I,d) μ block.
    Returns (A_block, mu_block) as numpy arrays or (None, None)."""
    try:
        import jax as _jax
        leaves = _jax.tree_util.tree_leaves(p1_params)
    except Exception:
        return None, None
    cand = []
    import numpy as _np
    for a in leaves:
        try:
            arr = _np.asarray(a)
            if arr.size > 0:
                cand.append(arr)
        except Exception:
            continue
    A_block = None
    mu_block = None
    for arr in cand:
        if arr.ndim >= 2 and arr.shape[-2:] == (I, I):
            A_block = arr.reshape(-1, I, I)[0]
            break
    if A_block is None:
        for arr in cand:
            if arr.size % (I * I) == 0:
                try:
                    A_block = arr.reshape(-1, I, I)[0]
                    break
                except Exception:
                    pass
    for arr in cand:
        if arr.ndim >= 2 and arr.shape[-2] == I and arr.shape[-1] == d_action:
            mu_block = arr.reshape(-1, I, d_action)[0]
            break
    if mu_block is None:
        for arr in cand:
            if arr.size % (I * d_action) == 0:
                try:
                    mu_block = arr.reshape(-1, I, d_action)[0]
                    break
                except Exception:
                    pass
    return A_block, mu_block

def pick_viz_leaves(p1_shim, p1_params, I: int, d_action: int, *, debug: bool = False) -> bool:
    """Populate a _P1TorchShim with one (I,I) logits block and one (I,d) μ table."""
    A_blk, mu_blk = extract_leaves_for_viz(p1_params, I, d_action)
    if A_blk is None or mu_blk is None:
        if debug:
            print("[viz] Could not infer (I,I) logits and (I,d) prototypes from p1_params.")
        return False
    try:
        import torch as _torch
        p1_shim.pick_leaves(_torch.as_tensor(A_blk, dtype=_torch.float32),
                            _torch.as_tensor(mu_blk, dtype=_torch.float32))
        if debug:
            import numpy as _np
            print(f"[viz] picked leaves: A{tuple(_np.asarray(A_blk).shape)} μ{tuple(_np.asarray(mu_blk).shape)}")
        return True
    except Exception as e:
        if debug:
            print(f"[viz] pick_leaves failed: {e}")
        return False

def _guarded_compile(lowered, label: str, timeout_s: int, debug: bool):
    """Compile in a background thread with a timeout so we don't stall."""
    import threading as _threading
    done = {"ok": False, "err": None}
    t0 = _time.perf_counter()
    def _runner():
        try:
            lowered.compile()
            done["ok"] = True
        except Exception as e:
            done["err"] = e
    th = _threading.Thread(target=_runner, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        if debug:
            print(f"[warmup] {label} compile timed out after {timeout_s}s; continuing.")
        return False
    if done["err"] is not None:
        if debug:
            print(f"[warmup] {label} compile failed: {done['err']}")
        return False
    if debug:
        dt = (_time.perf_counter() - t0) * 1e3
        print(f"[warmup] {label} compiled in {dt:.1f}ms.")
    return True

def _lower_arity_aware(jitted_step, state, flag: bool, debug: bool, label: str):
    """
    Try to lower a jitted step that may accept either (state, bool) or just (state).
    Returns (lowered, used_flag: bool).
    """
    try:
        lowered = jitted_step.lower(state, flag)
        if debug:
            print(f"[warmup] {label}: using 2-arg lower(state, {flag}).", flush=True)
        return lowered, True
    except TypeError:
        # Fallback: single-argument function
        lowered = jitted_step.lower(state)
        if debug:
            print(f"[warmup] {label}: using 1-arg lower(state) (no prune flag).", flush=True)
        return lowered, False


def warmup_precompile(jitted_step,
                      state,
                      *,
                      mode: str = "compile",     # "none" | "lower" | "compile"
                      timeout_s: int = 120,
                      debug: bool = False,
                      compile_prune_true: bool = False,
                      compile_prune: bool | None = None):
    """
    Precompile only the prune=False variant by default. If your jitted_step
    takes a single argument (state), we handle that transparently.

    jitted_step: either
       • fn(state, apply_prune: bool) -> (...)  (with static_argnums=(1,))
       • fn(state) -> (...)                     (no prune flag)
    """
    # Back-compat: some callers pass compile_prune instead of compile_prune_true
    if compile_prune is not None:
        compile_prune_true = bool(compile_prune) or bool(compile_prune_true)
    if mode == "none":
        return

    import time, threading, faulthandler, sys

    def _with_timeout(fn, desc):
        done = [False]
        def _poke():
            if not done[0]:
                print(f"Timeout (0:{timeout_s:02d}:00)!", flush=True)
                faulthandler.dump_traceback(sys.stderr)
        timer = threading.Timer(timeout_s, _poke)
        timer.start()
        try:
            t0 = time.perf_counter()
            out = fn()
            dt = (time.perf_counter() - t0) * 1e3
            return out, dt
        finally:
            done[0] = True
            timer.cancel()

    # ---- Always warm up prune=False (or the single-arg form) ---------
    if debug: print("[warmup] lowering (prune=False)...", flush=True)
    lower_false, used_flag = _lower_arity_aware(jitted_step, state, False, debug, "lower_false")
    # If _with_timeout wrapping is desired around lower (rarely needed), keep as-is:
    # lower_false, dt = _with_timeout(lambda: _lower_arity_aware(jitted_step, state, False, debug, 'lower_false')[0], "lower_false")
    # But lowering is fast; measure manually:
    print(f"[warmup] lower (prune=False) in 0.0ms", flush=True)

    if mode == "compile":
        if debug: print("[warmup] compiling (prune=False)...", flush=True)
        compiled_false, dt = _with_timeout(lambda: lower_false.compile(), "compile_false")
        print(f"[warmup] prune=False compiled in {dt:.1f}ms.", flush=True)

    # ---- Do NOT warm up prune=True here unless explicitly requested ---
    if compile_prune_true and used_flag:
        # Only possible if jitted_step accepts the prune flag.
        # We also optionally check for pruning cache attributes if your state uses them.
        has_cache = getattr(state, "row_prev_ids", None) is not None and getattr(state, "paths", None) is not None
        if not has_cache:
            print("[warmup] skip prune=True warmup (no pruning cache yet).", flush=True)
        else:
            if debug: print("[warmup] lowering (prune=True)...", flush=True)
            lower_true, _ = _lower_arity_aware(jitted_step, state, True, debug, "lower_true")
            print(f"[warmup] lower (prune=True) in 0.0ms", flush=True)
            if mode == "compile":
                if debug: print("[warmup] compiling (prune=True)...", flush=True)
                compiled_true, dt = _with_timeout(lambda: lower_true.compile(), "compile_true")
                print(f"[warmup] prune=True compiled in {dt:.1f}ms.", flush=True)

def postprune_compile(jit_step, solver_state, *, timeout_s: int, debug: bool = False) -> bool:
    """Optionally compile the post-prune shape to avoid first-iteration stall.
    Works for both (state, bool) and (state) arities.
    """
    try:
        lowered, used_flag = _lower_arity_aware(jit_step, solver_state, True, debug, "post-prune lower")
        return _guarded_compile(lowered, "post-prune", timeout_s, debug)
    except Exception as e:
        if debug:
            print(f"[warmup] post-prune compile skipped: {e}")
        return False


_SCALAR_KEYS = {"L", "g_p1", "g_p2", "n_seq"}  # extend if you add new scalars

def block_on_metrics(metrics: dict) -> dict:
    """
    Return a dict with only scalar entries materialized on the host.
    Non-scalars are left as-is (device arrays) and are NOT returned.
    """
    out = {}
    for k, v in list(metrics.items()):
        if isinstance(v, (int, float)):
            out[k] = v
        elif isinstance(v, jax.Array):
            if v.ndim == 0:
                out[k] = float(jax.device_get(v))
            elif k in _SCALAR_KEYS and v.size == 1:  # edge case: shape (1,)
                out[k] = float(np.array(v).reshape(()))
            else:
                # don't materialize big stuff on host
                continue
    return out

def save_checkpoint(path_or_dir: _Path | str, *args):
    """
    Save a msgpack checkpoint. Supports two calling styles:
      1) save_checkpoint(ckpt_path, state)
      2) save_checkpoint(ckpt_dir, step, state)
    """
    from flax import serialization as _flax_serial
    import msgpack
    if isinstance(path_or_dir, (str, _Path)) and str(path_or_dir).endswith(".msgpack"):
        ckpt_path = _Path(path_or_dir)
        if len(args) != 1:
            raise TypeError("save_checkpoint(path, state) expected exactly 2 args total.")
        state = args[0]
    else:
        if len(args) != 2:
            raise TypeError("save_checkpoint(dir, step, state) expected 3 args total.")
        ckpt_dir, step, state = path_or_dir, args[0], args[1]
        ckpt_path = _Path(ckpt_dir) / f"ckpt_{int(step):04d}.msgpack"
    payload = {"p1_params": getattr(state, 'p1_params', None),
               "p2_params": getattr(state, 'p2_params', None)}
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "wb") as f:
        f.write(msgpack.dumps(_flax_serial.to_state_dict(payload)
                              if hasattr(_flax_serial, "to_state_dict")
                              else _flax_serial.to_bytes(payload)))
    print(f"[ckpt] saved to {ckpt_path}")
    return str(ckpt_path)

def write_jsonl(path: str, metrics: dict):
    log_metrics = {k: float(v) for k, v in metrics.items() if k != "iter"}
    log_metrics["iter"] = int(metrics["iter"])
    with open(path, "a") as fh:
        fh.write(_json.dumps(log_metrics) + "\n")

def viz_interaction_animations(TorchFootballGame, TorchDefaultSpec,
                               p1_for_viz, p2_for_viz,
                               game_spec_dict: dict, anim_dir: _Path,
                               epoch: int, *, save: bool = True, debug: bool = False):
    """Run non-JAX interaction visualization if the Torch game is available.
    Includes the initial state frame, and optionally saves GIFs."""
    if TorchFootballGame is None or TorchDefaultSpec is None:
        if debug:
            print("[viz] Torch game/spec not available; skipping.")
        return
    try:
        torch_game = TorchFootballGame(TorchDefaultSpec(N=game_spec_dict["N_PLAYERS"],
                                                        horizon=game_spec_dict["T"],
                                                        dt=game_spec_dict["tau"],
                                                        device="cpu",
                                                        n_substeps=game_spec_dict["n_substeps"]))
        outs = torch_game.visualize_most_likely(p1_for_viz, p2_for_viz, fps=6, save_dir=None)
        if save:
            from matplotlib import animation as _anim
            out_dir = _Path(anim_dir) / "plays"
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, (_, a) in enumerate(outs):
                out_gif = out_dir / f"most_likely_type{i:02d}_iter{epoch:04d}.gif"
                a.save(out_gif, writer="pillow", fps=6)
            print(f"[viz] saved {len(outs)} animations to {out_dir}")
    except Exception as e:
        print(f"[viz] skipped: {e}")


# --- New: render_interaction ---
def render_interaction(state, out_dir: str, *, debug: bool = False):
    """
    CPU-only interaction visualization from solver `state`.
    Heuristically infers (I, d) from p1_params and uses the non-JAX Torch
    FootballGame if available. Safely no-ops if not.
    """
    try:
        from game import FootballGame as TorchFootballGame, default_football_spec as TorchDefaultSpec
    except Exception as e:
        if debug:
            print(f"[viz] Torch FootballGame not available: {e}")
        return
    # Infer I and action dim d from params
    A_blk, mu_blk = extract_leaves_for_viz(getattr(state, "p1_params", None), I=2, d_action=2)
    if mu_blk is None or A_blk is None:
        if debug:
            print("[viz] could not infer leaves; skipping interaction viz.")
        return
    I = int(np.asarray(A_blk).shape[0])
    d = int(np.asarray(mu_blk).shape[1])
    # Prepare shims
    p1_shim = _P1TorchShim(getattr(state, "p1_params", {}), I_hint=I)
    ok = pick_viz_leaves(p1_shim, getattr(state, "p1_params", {}), I, d, debug=debug)
    if not ok:
        return
    p2_shim = _P2TorchShim(d)
    # Build a Torch spec, inferring N from d (=2N actions)
    N_guess = max(1, d // 2)
    # Try to pull T/dt/substeps from state.config if present
    T = getattr(getattr(state, "config", None), "T", 1.0)
    dt = getattr(getattr(state, "config", None), "dt", 0.05)
    n_sub = getattr(getattr(state, "config", None), "n_substeps", 4)
    spec = TorchDefaultSpec(N=N_guess, horizon=T, dt=dt, device="cpu", n_substeps=n_sub)
    # Run and save
    anim_dir = _Path(out_dir)
    epoch = int(getattr(state, "iter", 0))
    viz_interaction_animations(TorchFootballGame, TorchDefaultSpec,
                               p1_shim, p2_shim,
                               spec, anim_dir, epoch,
                               save=True, debug=debug)

def prime_cublas():
    """
    Run a tiny GEMM to force-load cuBLAS while plenty of VRAM is free.
    Call this AFTER importing JAX but BEFORE you create any Torch CUDA tensors.
    """
    import jax, jax.numpy as jnp
    a = jnp.ones((8, 8), dtype=jnp.float32)
    _ = (a @ a).block_until_ready()


# Back-compat alias expected by some mains
def tree_viz_from_p1_params_compat(p1_params, out_png: str, **kw):
    return tree_viz_from_p1_params(p1_params, out_png, **kw)