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

# -----------------------------------------------------------

def build_tree(p1, prior: Tensor, I: int, K: int, *, ent_thr: float = 0.2):
    """Return a directed NetworkX graph with node / edge attrs."""
    dev = prior.device
    root_idx = 0
    G = nx.DiGraph()

    # node attr helper --------------------------------------
    def _add_node(t: int, idx: int, belief: Tensor):
        pure   = p1._pure[t][idx].item() if t < K else False
        if t == 0:
            ntype = "root"
        elif t == K:
            ntype = "leaf"
        elif pure:
            ntype = "pure"
        else:
            # prototype diversity test
            _, mu_tbl = p1._slice(t, idx)           # (I, d)
            # pair-wise max distance
            max_dist = (mu_tbl.unsqueeze(0) - mu_tbl.unsqueeze(1)) \
                        .norm(dim=-1).max().item()
            mu_thr = 1e-2                           # tweak to your units
            ntype = "reveal" if max_dist > mu_thr else "non"
        G.add_node((t, idx), belief=belief.cpu(), ntype=ntype)

    # --------------------------------------------------------
    # BFS over tree with pruning when node is pure
    # --------------------------------------------------------
    _add_node(0, root_idx, prior)
    queue: List[Tuple[int, int, Tensor]] = [(0, root_idx, prior)]

    while queue:
        t, idx, belief = queue.pop(0)
        if t == K:
            continue

        pure = p1._pure[t][idx].item()
        Λσ, _ = p1._slice(t, idx)
        A     = torch.softmax(Λσ, dim=-1)                   # (I,I)

        # effective message distribution  q_j = Σ_i p_i A_ij
        q = (belief.unsqueeze(0) @ A).squeeze(0)            # (I,)
        q = q / q.sum().clamp_min(1e-12)

        cols = [torch.argmax(belief).item()] if pure else list(range(I))
        for j in cols:
            child_idx = idx * I + j     # base-I shift
            child_belief = belief.clone()
            # Bayes update --------------------------------------
            Aj = A[:, j]
            numer = Aj * belief
            child_belief = numer / numer.sum().clamp_min(1e-12)

            prob_j = q[j].item()
            G.add_edge((t, idx), (t+1, child_idx), prob=prob_j)
            _add_node(t+1, child_idx, child_belief)

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