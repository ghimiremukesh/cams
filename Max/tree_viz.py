from __future__ import annotations
"""tree_viz.py
────────────────────────────────────────────────────────
Visualisation helper for the collapsing-tree differential games.
Generates a NetworkX directed graph where
    • nodes  = (t, idx)   idx = base-I encoding of message history
    • node attributes:
        ─ belief      : public belief vector p ∈ Δᴵ
        ─ pure        : bool  (completely revealing)
        ─ revealing   : bool  (entropy < ent_thr)
        ─ leaf        : bool  (t == K)
    • edge attributes:
        ─ prob        : conditional probability of the column j

Colour map (Matplotlib hex codes):
    root              →  '#8e44ad' (purple)
    pure              →  '#27ae60' (green)
    revealing (not pure) → '#f39c12' (orange)
    non-revealing     →  '#95a5a6' (grey)
    leaf              →  '#2c3e50' (navy)

Functions
---------
    build_tree(p1, prior, I, K, ent_thr=0.2) -> nx.DiGraph
    draw_tree(G, out_png)                    -> str

Requires networkx ≥ 2.6 and matplotlib.
"""

from typing import List, Tuple
import math
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import Tensor

# -----------------------------------------------------------

_COLORS = {
    "root"      : "#8e44ad",
    "pure"      : "#27ae60",
    "reveal"    : "#f39c12",
    "non"       : "#95a5a6",
    "leaf"      : "#2c3e50",
}

# -----------------------------------------------------------

def _entropy(p: Tensor, eps: float = 1e-12):
    p = p.clamp(min=eps)
    return -(p * p.log()).sum(-1)

# ---------- parameters you may tune ---------------------------------
onehot_thr   = 1e-3       # tolerance for identifying one-hot rows
ent_thr_bel  = 1e-3       # belief entropy threshold for deep-pure
mu_thr       = 1e-2       # μ-row diversity threshold
# --------------------------------------------------------------------

def build_tree(p1, prior: Tensor, I: int, K: int) -> nx.DiGraph:
    dev = prior.device
    G   = nx.DiGraph()

    def _entropy(p: Tensor, eps: float = 1e-12):
        return -(p.clamp(min=eps) * (p+eps).log()).sum()

    # ── helper to classify and add a node ───────────────────────────
    def _add(t: int, idx: int, belief: Tensor) -> str:
        if t == 0:
            ntype = "root"
        elif t == K:
            ntype = "leaf"
        else:
            Λσ, μ_tbl = p1._slice(t, idx)
            A   = torch.softmax(Λσ, dim=-1)                 # (I,I)

            row_max, arg = A.max(dim=-1)                    # each row’s peak & col
            one_hot = (row_max > 1 - onehot_thr).all() and \
                    (len(torch.unique(arg)) == I)         # permutation check

            ent = _entropy(belief).item()

            if one_hot:
                ntype = "deep" if ent < ent_thr_bel else "first"
            else:
                max_dist = (μ_tbl.unsqueeze(0) - μ_tbl.unsqueeze(1)) \
                        .norm(dim=-1).max().item()
                ntype = "reveal" if max_dist > mu_thr else "non"

        G.add_node((t, idx), belief=belief.cpu(), ntype=ntype)
        return ntype

    # ── BFS over tree ------------------------------------------------
    queue = [(0, 0, prior)]
    _add(0, 0, prior)

    while queue:
        t, idx, belief = queue.pop(0)
        if t == K:
            continue

        Λσ, _ = p1._slice(t, idx)
        A     = torch.softmax(Λσ, dim=-1)              # (I,I)

        q = (belief.unsqueeze(0) @ A).squeeze(0)
        q = q / q.sum().clamp_min(1e-12)

        # decide branching: one column if node is 'first' or 'deep'
        ntype = G.nodes[(t, idx)]["ntype"]
        if ntype in {"first", "deep"}:
            cols = [torch.argmax(belief).item()]
        else:
            cols = list(range(I))

        for j in cols:
            child_idx = idx * I + j
            Aj = A[:, j]
            child_belief = (Aj * belief) / (Aj@belief).clamp_min(1e-12)

            prob_j = q[j].item()
            G.add_edge((t, idx), (t+1, child_idx), prob=prob_j)
            _add(t+1, child_idx, child_belief)
            queue.append((t+1, child_idx, child_belief))

    return G

# -----------------------------------------------------------

def draw_tree(G: nx.DiGraph, out_png: str | None = None):
    # pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except Exception:
        # fallback: force-directed layout
        pos = nx.spring_layout(G, seed=0)

    # node colours
    ncolor = [_COLORS[G.nodes[n]["ntype"]] for n in G.nodes]
    sizes  = [300 if G.nodes[n]["ntype"] == "root" else 150 for n in G.nodes]

    # edge colours = greyscale prob
    ecolor = []
    for u, v, d in G.edges(data=True):
        p = d.get("prob", 0.0)
        shade = 1.0 - p  # 0 prob → white, 1 prob → black
        ecolor.append((shade, shade, shade))

    nx.draw(G, pos, with_labels=False, node_size=sizes, node_color=ncolor,
            edge_color=ecolor, arrows=False, linewidths=0.5)

    # legend --------------------------------------------------
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=_COLORS[k], label=k) for k in _COLORS]
    plt.legend(handles=handles, fontsize=8, loc="upper right")
    plt.axis("off")
    if out_png:
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        return out_png
    return None