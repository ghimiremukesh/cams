# football_demo_jax.py
# =========================================================
# Driver for JAX-based training, configured for multi-GPU.
# =========================================================

from __future__ import annotations
import os, json, math, time, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax import jax_utils as flax_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

DM_COLORS = {"off": "red", "def": "blue", "star": "#FFC107"}
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 11, "axes.facecolor": "#FFFFFF",
                     "axes.edgecolor": "#E0E0E0", "axes.grid": True, "grid.color": "#EAEAEA",
                     "grid.linewidth": 0.8, "grid.linestyle": "-"})

from game_jax import default_football_spec, FootballGame
from player_jax import CAMSInformed, BR
from solver_jax_multi_gpu import DSGDASolver





# =========================================================
# Global switches & defaults
# =========================================================
SEED        = 1
# ----- Multi-GPU Batching Configuration -----
# Total effective batch size across all devices
GLOBAL_BATCH_SIZE = 1000
# Batch size per device for a single forward/backward pass. Tune this to maximize VRAM usage.
PER_DEVICE_MICRO_BATCH_SIZE = 50

EPOCHS      = 50_000
VIS_EVERY   = 2500
LOG_ROOT    = "multi_gpu_test/"
HORIZON     = 2.5
DT          = 0.25
N_SUBSTEPS  = 4
N_PLAYERS   = 11

POS_STD = 0.05  # default = 0.05

PRINT_EVERY = 100
SHOW_DEBUG  = False

PLAYER_SPEC = {"hidden": 128, "temperature": 1.0, "init_scale": 1e-1, "ent_thr_belief": 1e-2}
SOLVER_SPEC = {"lr_p1": 3e-3, "lr_p2": 1e-2, "momentum": 0.6, "C2_p1": 10.0, "C2_p2": 10.0}

# =========================================================
# Initialisation and Visualization Helpers (Unchanged)
# =========================================================
def _init_cams_params(rng, model: CAMSInformed, game: FootballGame):
    B = 1
    dummy_obs = {"x": jnp.zeros((B, game.STATE_DIM)), "p": jnp.tile(game.P0, (B, 1)), "t": jnp.zeros((B,))}
    params = model.init(rng, dummy_obs, 0)["params"]
    base = unfreeze(params)
    for k in range(1, game.K):
        vk = model.init(jax.random.fold_in(rng, k), dummy_obs, k)["params"]
        add = unfreeze(vk)
        if f"step{k}" in add: base[f"step{k}"] = add[f"step{k}"]
    return freeze(base)

def _init_br_params(rng, model: BR, game: FootballGame):
    B = 1
    dummy_obs = {"x": jnp.zeros((B, game.STATE_DIM)), "p": jnp.tile(game.P0, (B, 1)), "t": jnp.zeros((B,))}
    params = model.init(rng, dummy_obs, 0)["params"]
    base = unfreeze(params)
    for k in range(1, game.K):
        vk = model.init(jax.random.fold_in(rng, k), dummy_obs, k)["params"]
        add = unfreeze(vk)
        if f"step{k}" in add: base[f"step{k}"] = add[f"step{k}"]
    return freeze(base)

def _split_state(game: FootballGame, x: jnp.ndarray):
    B = x.shape[0]
    pos1 = x[:, 0:2*game.N].reshape(B, game.N, 2)
    vel1 = x[:, 2*game.N:4*game.N].reshape(B, game.N, 2)
    pos2 = x[:, 4*game.N:6*game.N].reshape(B, game.N, 2)
    vel2 = x[:, 6*game.N:8*game.N].reshape(B, game.N, 2)
    return pos1, vel1, pos2, vel2

def visualize_most_likely(
    game: FootballGame,
    p1_model: CAMSInformed, p1_params,
    p2_model: BR,           p2_params,
    *,
    fps: int = 6,
):
    """
    CPU-only rendering of one rollout per hidden type i★ with persistent 'blob' heat.

    Change requested:
      • Blob patches appear ONLY AFTER their time-step (i.e., a blob stamped at t_k
        becomes visible starting at frame k+1, not during frame k).
      • Fixed blob radius (data units), not magnitude-scaled.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle

    # ------------------------- Styling -------------------------
    _rc = {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "figure.dpi": 120,
    }
    prev_rc = {k: plt.rcParams.get(k, None) for k in _rc}
    plt.rcParams.update(_rc)

    # Team colors / colormaps
    OFF_LINE = "#c62828"      # offense red
    DEF_LINE = "#1565c0"      # defense blue
    OFF_CMAP = plt.cm.Reds
    DEF_CMAP = plt.cm.Blues

    # Markers / sizes (pt^2 where applicable)
    S_OUTLINE = 160.0
    S_POS_DOT = 28.0
    DOT_INSIDE = 36.0
    OUTLINE_LW = 1.6
    TRAIL_LW = 1.6

    # ---- Blob settings (in DATA units) ----
    HEAT_ALPHA = 0.35         # per-blob opacity
    HEAT_GAMMA = 0.70         # <1 brightens low magnitudes (for color)
    HEAT_BASE  = 0.30         # color floor to avoid near-white
    Z_BLOBS    = 2.0          # zorder below outlines/trails but above grid

    # Fixed radius (data units). Tweak this one knob:
    R_FIXED = 0.05            # e.g., 0.05 of field units

    # Delay so blobs appear only AFTER their step
    APPEAR_DELAY = 1          # stamp frame (t - APPEAR_DELAY)

    # Helpers ---------------------------------------------------
    def _split_state(game, x):
        return globals()["_split_state"](game, x)

    def _team_action_per_player(u: jnp.ndarray, N: int) -> np.ndarray:
        """Return per-player L2 magnitudes (N,) from a team action vector."""
        u = jnp.asarray(u)
        u = jnp.squeeze(u)
        D = int(u.shape[-1]) if u.ndim >= 1 else 0
        if D == 2 * N:
            arr = u.reshape(N, 2)
        elif N > 0 and D % N == 0 and D > 0:
            arr = u.reshape(N, D // N)
        else:
            mag = float(jnp.linalg.norm(u)) / max(N, 1)
            return np.full((N,), mag, dtype=float)
        return np.asarray(jnp.linalg.norm(arr, axis=1), dtype=float)

    # Indices for highlights
    RB_IDX = getattr(game, "RB_IDX", getattr(game, "BALL_IDX", 0))
    QB_IDX = getattr(game, "QB_IDX", getattr(game, "BALL_IDX", 0))
    REC_IDX = None
    for cand in ("REC_IDX", "WR_IDX", "RECV_IDX"):
        if hasattr(game, cand):
            REC_IDX = getattr(game, cand)
            break

    
    # Visualization is CPU-only and needs a single copy of the parameters
    p1_params_host = flax_utils.unreplicate(p1_params)
    p2_params_host = flax_utils.unreplicate(p2_params)

    outs = []

    for i_star in range(game.I):
        # Reset and set hidden type deterministically
        state, _ = game.reset(batch_size=1)
        state = state.tree_replace(i_star=jnp.full((1,), i_star, dtype=jnp.int32))
        obs = {"x": state.x, "p": state.p, "t": state.t}

        pos1, _, pos2, _ = _split_state(game, state.x)
        traj_off = [jnp.array(pos1[0]).copy()]
        traj_def = [jnp.array(pos2[0]).copy()]
        p_traj   = [float(state.p[0, 0])]
        times    = [0.0]

        off_mags_seq = [np.zeros((game.N,), dtype=float)]
        def_mags_seq = [np.zeros((game.N,), dtype=float)]

        # Rollout under most-likely row j_k for this type
        for k in range(game.K):
            out = p1_model.apply({"params": p1_params_host}, obs, k)
            A = jax.nn.softmax(out["A_logits"][0], axis=-1)
            row = A[i_star]
            j_k = int(jnp.argmax(row))

            mu_tbl = out["μ"][0]            # [J, D_off]
            u1 = mu_tbl[j_k][None, :]       # (1, D_off)
            u2 = p2_model.apply({"params": p2_params_host}, obs, k)

            # Step + Bayes update
            state, _ = game.step(state, u1, u2)
            state = state.tree_replace(
                p=game._bayes_update(state.p, A[None, ...], jnp.array([j_k]))
            )
            obs = {"x": state.x, "p": state.p, "t": state.t}

            pos1, _, pos2, _ = _split_state(game, state.x)
            traj_off.append(jnp.array(pos1[0]).copy())
            traj_def.append(jnp.array(pos2[0]).copy())
            p_traj.append(float(state.p[0, 0]))
            times.append((k + 1) * game.dt)

            off_mags_seq.append(_team_action_per_player(u1, game.N))
            def_mags_seq.append(_team_action_per_player(u2, game.N))

        # Stacks
        off_np = np.stack([np.array(t) for t in traj_off], axis=0)  # (T, N, 2)
        def_np = np.stack([np.array(t) for t in traj_def], axis=0)  # (T, N, 2)
        off_mags_np = np.stack(off_mags_seq, axis=0)                # (T, N)
        def_mags_np = np.stack(def_mags_seq, axis=0)                # (T, N)
        T_frames = off_np.shape[0]
        max_off = float(np.max(off_mags_np)) if off_mags_np.size else 1.0
        max_def = float(np.max(def_mags_np)) if def_mags_np.size else 1.0
        max_off = max(max_off, 1e-12)
        max_def = max(max_def, 1e-12)

        # Figure
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(6.5, 8.2), gridspec_kw={"height_ratios": [4, 1]}
        )
        for side in ("top", "right"):
            ax_top.spines[side].set_visible(False)
            ax_bot.spines[side].set_visible(False)

        # Axes limits
        xmin, xmax = -game.BOX_POS - 0.2, game.BOX_POS + 0.2
        ymin, ymax = -game.BOX_POS - 0.2, game.BOX_POS + 0.2
        ax_top.set_xlim(xmin, xmax)
        ax_top.set_ylim(ymin, ymax)
        ax_top.set_aspect("equal")
        ax_top.set_title(f"Most-likely path – type {i_star}", fontweight="bold")
        ax_top.set_xlabel("x (field units)")
        ax_top.set_ylabel("y (field units)")

        # Belief subplot
        ax_bot.set_xlim(0, game.T)
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.set_xlabel("time (s)")
        ax_bot.set_ylabel("belief p[type=0]")
        ax_bot.plot(times, p_traj, color="#333333", lw=1.6)

        # Dotted trajectory lines
        off_trails = [
            ax_top.plot([], [], lw=TRAIL_LW, alpha=0.75, color=OFF_LINE,
                        linestyle=":", solid_capstyle="round", zorder=2.6)[0]
            for _ in range(game.N)
        ]
        def_trails = [
            ax_top.plot([], [], lw=TRAIL_LW, alpha=0.75, color=DEF_LINE,
                        linestyle=":", solid_capstyle="round", zorder=2.6)[0]
            for _ in range(game.N)
        ]

        # Outline markers (no fill)
        off_outline = ax_top.scatter(
            [], [], s=S_OUTLINE, facecolors="none", edgecolors="black",
            linewidths=OUTLINE_LW, marker="o", zorder=3.4
        )
        def_outline = ax_top.scatter(
            [], [], s=S_OUTLINE, facecolors="none", edgecolors="black",
            linewidths=OUTLINE_LW, marker="s", zorder=3.4
        )

        # Position dots (fixed size, no outline)
        off_pos = ax_top.scatter(
            [], [], s=S_POS_DOT, c=OFF_LINE, edgecolors="none", alpha=0.95, zorder=3.2
        )
        def_pos = ax_top.scatter(
            [], [], s=S_POS_DOT, c=DEF_LINE, edgecolors="none", alpha=0.95, zorder=3.2
        )

        # Persistent blob storage (as Patches)
        off_blob_patches = []
        def_blob_patches = []

        # Highlights
        rb_circle = ax_top.scatter([], [], s=S_OUTLINE * 1.05, facecolors="none",
                                   edgecolors="black", linewidths=OUTLINE_LW + 0.2, marker="o", zorder=3.8)
        # RB cross via plot() (always visible)
        MS_CROSS = float(np.sqrt(S_OUTLINE) * 1.25)  # points (not pt^2)
        rb_cross, = ax_top.plot(
            [], [], linestyle="None",
            marker="x", markersize=MS_CROSS, markeredgewidth=2.8,
            color="black", zorder=4.0
        )
        qb_circle = ax_top.scatter([], [], s=S_OUTLINE * 1.05, facecolors="none",
                                   edgecolors="black", linewidths=OUTLINE_LW + 0.2, marker="o", zorder=3.8)
        qb_dot    = ax_top.scatter([], [], s=DOT_INSIDE, c="black", marker="o", edgecolors="none", zorder=3.9)
        rec_circle = ax_top.scatter([], [], s=S_OUTLINE * 1.05, facecolors="none",
                                    edgecolors="black", linewidths=OUTLINE_LW + 0.2, marker="o", zorder=3.8)
        rec_dot    = ax_top.scatter([], [], s=DOT_INSIDE, c="black", marker="o", edgecolors="none", zorder=3.9)

        # ---------- bind everything into defaults to avoid late-binding ----------
        def init(off_np=off_np, def_np=def_np,
                 off_outline=off_outline, def_outline=def_outline,
                 off_pos=off_pos, def_pos=def_pos,
                 off_trails=off_trails, def_trails=def_trails,
                 rb_circle=rb_circle, rb_cross=rb_cross,
                 qb_circle=qb_circle, qb_dot=qb_dot,
                 rec_circle=rec_circle, rec_dot=rec_dot):
            empty_xy = np.empty((0, 2))
            off_outline.set_offsets(empty_xy)
            def_outline.set_offsets(empty_xy)
            off_pos.set_offsets(empty_xy)
            def_pos.set_offsets(empty_xy)
            for ln in off_trails + def_trails:
                ln.set_data([], [])
            # clear any pre-existing blobs (when re-running in notebooks)
            for p in off_blob_patches + def_blob_patches:
                try:
                    p.remove()
                except Exception:
                    pass
            off_blob_patches.clear()
            def_blob_patches.clear()

            # clear highlights
            rb_circle.set_offsets(empty_xy); rb_cross.set_data([], [])
            qb_circle.set_offsets(empty_xy); qb_dot.set_offsets(empty_xy)
            rec_circle.set_offsets(empty_xy); rec_dot.set_offsets(empty_xy)
            return (
                off_outline, def_outline, off_pos, def_pos,
                rb_circle, rb_cross, qb_circle, qb_dot, rec_circle, rec_dot,
                *off_trails, *def_trails
            )

        def _add_blobs(ax, centers, mags, cmap, wmax, zorder_list, patch_store):
            """Create Circle patches (persistent) and add to ax.
               Fixed-radius blobs; color still reflects magnitude."""
            if centers.size == 0:
                return
            w = np.clip(mags / (wmax + 1e-12), 0.0, 1.0)
            w_plot = np.power(w, HEAT_GAMMA)
            # Fixed radius (data units)
            r = float(R_FIXED)
            # Colormap colors (avoid white end; set alpha)
            w_col = HEAT_BASE + (1.0 - HEAT_BASE) * w_plot
            cols = cmap(w_col)
            if cols.ndim == 1: cols = cols[None, :]
            for (x, y), c in zip(centers, cols):
                color = (float(c[0]), float(c[1]), float(c[2]), HEAT_ALPHA)
                circ = Circle((float(x), float(y)), radius=r,
                              facecolor=color, edgecolor='none', zorder=Z_BLOBS)
                ax.add_patch(circ)
                patch_store.append(circ)

        def update(frame,
                   off_np=off_np, def_np=def_np,
                   off_mags_np=off_mags_np, def_mags_np=def_mags_np,
                   max_off=max_off, max_def=max_def,
                   off_outline=off_outline, def_outline=def_outline,
                   off_pos=off_pos, def_pos=def_pos,
                   off_trails=off_trails, def_trails=def_trails,
                   rb_circle=rb_circle, rb_cross=rb_cross,
                   qb_circle=qb_circle, qb_dot=qb_dot,
                   rec_circle=rec_circle, rec_dot=rec_dot,
                   i_star=i_star, RB_IDX=RB_IDX, QB_IDX=QB_IDX, REC_IDX=REC_IDX,
                   ax_top=ax_top, APPEAR_DELAY=APPEAR_DELAY):
            # outlines + position dots
            off_outline.set_offsets(off_np[frame])
            def_outline.set_offsets(def_np[frame])
            off_pos.set_offsets(off_np[frame])
            def_pos.set_offsets(def_np[frame])

            # dotted trails
            xs_off = off_np[:frame + 1, :, 0]; ys_off = off_np[:frame + 1, :, 1]
            xs_def = def_np[:frame + 1, :, 0]; ys_def = def_np[:frame + 1, :, 1]
            for i in range(xs_off.shape[1]):
                off_trails[i].set_data(xs_off[:, i], ys_off[:, i])
                def_trails[i].set_data(xs_def[:, i], ys_def[:, i])

            # PERSISTENT BLOBS:
            #   stamp ONLY the frame that just finished (frame - APPEAR_DELAY)
            t_stamp = frame - APPEAR_DELAY
            if t_stamp >= 0:
                _add_blobs(ax_top, off_np[t_stamp], off_mags_np[t_stamp], OFF_CMAP, max_off, Z_BLOBS, off_blob_patches)
                _add_blobs(ax_top, def_np[t_stamp], def_mags_np[t_stamp], DEF_CMAP, max_def, Z_BLOBS, def_blob_patches)

            # clear highlights
            empty_xy = np.empty((0, 2))
            rb_circle.set_offsets(empty_xy); rb_cross.set_data([], [])
            qb_circle.set_offsets(empty_xy); qb_dot.set_offsets(empty_xy)
            rec_circle.set_offsets(empty_xy); rec_dot.set_offsets(empty_xy)

            # add highlights for this type/frame
            if i_star == 0:
                x, y = off_np[frame, RB_IDX]
                rb_circle.set_offsets(np.array([[x, y]]))
                rb_cross.set_data([x], [y])     # visible “×”
            else:
                qb_xy = off_np[frame, QB_IDX].reshape(1, 2)
                qb_circle.set_offsets(qb_xy)
                qb_dot.set_offsets(qb_xy)
                if REC_IDX is not None:
                    rec_xy = off_np[frame, REC_IDX].reshape(1, 2)
                else:
                    dists = np.linalg.norm(off_np[frame] - qb_xy[0], axis=1)
                    rec_guess = int(np.argmax(dists))
                    rec_xy = off_np[frame, rec_guess].reshape(1, 2)
                rec_circle.set_offsets(rec_xy)
                rec_dot.set_offsets(rec_xy)

            # return animated artists (patches live on the Axes)
            return (
                off_outline, def_outline, off_pos, def_pos,
                rb_circle, rb_cross, qb_circle, qb_dot, rec_circle, rec_dot,
                *off_trails, *def_trails
            )

        ani = animation.FuncAnimation(
            fig, update, frames=T_frames, init_func=init,
            blit=False, interval=1000 / fps, repeat=False
        )

        # Notebook display helper
        try:
            from IPython.display import HTML
            html = HTML(ani.to_jshtml())
        except Exception:
            html = None

        outs.append((html, ani))

    # Restore rcParams
    for k, v in prev_rc.items():
        if v is not None:
            plt.rcParams[k] = v

    return outs


# =========================================================
# Training curves plotter (reads JSONL like PyTorch util)
# =========================================================
def plot_run_jsonl(log_path: str, save_png: str | None = None):
    import pandas as pd
    import json
    import matplotlib.pyplot as plt

    with open(log_path) as fh:
        records = [json.loads(line) for line in fh]
    if not records:
        return
    df = pd.DataFrame(records)

    # Make sure 'iter' column exists and is suitable for x-axis
    if "iter" not in df.columns:
        print("Log file does not contain an 'iter' column.")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 10)) # Increased size for better readability
    ax = axes.ravel()
    
    # --- Unchanged Plots ---
    # Assuming 'L', 'g_p1', 'g_p2', 'n_seq' columns exist
    if "L" in df.columns:
      ax[0].plot(df["iter"], df["L"])
      ax[0].set_title("Loss L")

    if "g_p1" in df.columns and "g_p2" in df.columns:
      ax[1].plot(df["iter"], df["g_p1"], label="‖g₁‖")
      ax[1].plot(df["iter"], df["g_p2"], label="‖g₂‖")
      ax[1].legend()
      ax[1].set_title("Gradient norms")
    
    if "n_seq" in df.columns:
      ax[2].plot(df["iter"], df["n_seq"])
      ax[2].set_title("# active sequences S")

    # --- Fixed Time Plot ---
    phases = ["t_prune", "t_loss", "t_backward", "t_momentum", "t_step"]
    
    # Filter for phases that actually exist in the DataFrame
    existing_phases = [p for p in phases if p in df.columns]
    
    if existing_phases:
        # Prepare data for the stackplot
        y_data = [df[ph] for ph in existing_phases]
        labels = [ph.replace("t_", "") for ph in existing_phases]

        # Use a stacked area plot instead of a bar plot
        ax[3].stackplot(df["iter"], y_data, labels=labels)
        
        # Calculate a reasonable y-limit to ignore initial spikes
        total_time = df[existing_phases].sum(axis=1)
        if not total_time.empty:
            # Clip the y-axis at 110% of the 98th percentile for a clearer view
            upper_limit = total_time.quantile(0.98) * 1.1
            ax[3].set_ylim(0, upper_limit)
            
    ax[3].legend()
    ax[3].set_title("Per-iteration time (ms)")
    
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=150)
    # plt.show() # Uncomment to display plot interactively
    plt.close(fig)

# =========================================================
# Main
# =========================================================
def main():
    # Detect available devices
    num_devices = jax.device_count()
    if num_devices > 1:
        print(f"Detected {num_devices} GPUs. Running in multi-GPU mode.")
    else:
        print("Detected 1 device. Running in single-device mode.")

    key = jax.random.PRNGKey(SEED)

    spec = default_football_spec(N=N_PLAYERS, horizon=HORIZON, dt=DT, n_substeps=N_SUBSTEPS, lineup_pos_std=POS_STD)
    game = FootballGame(spec, batch_size=PER_DEVICE_MICRO_BATCH_SIZE)
    demo = FootballGame(spec, batch_size=1)

    p1_model = CAMSInformed(I=game.I, d=game.ACTION_DIM, feat_dim=game.FEAT_DIM, K=game.K,
                            box_acc=game.BOX_ACC, hidden=PLAYER_SPEC["hidden"])
    p2_model = BR(d=game.ACTION_DIM, feat_dim=game.FEAT_DIM, K=game.K,
                  box_acc=game.BOX_ACC, hidden=PLAYER_SPEC["hidden"])

    key, k1, k2 = jax.random.split(key, 3)
    p1_params = _init_cams_params(k1, p1_model, game)
    p2_params = _init_br_params(k2, p2_model, game)

    solver = DSGDASolver(
        game, p1_model, p1_params, p2_model, p2_params,
        SOLVER_SPEC, log_root=LOG_ROOT, seed=SEED,
        global_batch_size=GLOBAL_BATCH_SIZE,
        per_device_micro_batch_size=PER_DEVICE_MICRO_BATCH_SIZE,
    )

    print(f"[run] dir: {solver.run_dir}")
    print(f"[cfg] I={game.I}  K={game.K}  N={game.N}  d={game.ACTION_DIM}")
    print(f"[cfg] Global Batch: {solver.global_batch_size}, Per-Device Micro-Batch: {solver.per_device_micro_batch_size}")
    print(f"[cfg] Grad Accumulation Steps: {solver.n_accum_steps}")

    for epoch in tqdm(range(EPOCHS)):
        stats = solver.step()
        if (epoch % PRINT_EVERY == 0):
            print(f"\n[{epoch:05d}] L={stats['L']:+.4f} ||g_p1||={stats['g_p1']:.4f} ||g_p2||={stats['g_p2']:.4f} wall={stats['wall_ms']:.1f}ms")
        if (epoch % VIS_EVERY == 0) or epoch == EPOCHS-1:
            ckpt_path = solver.save_checkpoint(epoch)
            print(f"[ckpt] saved {ckpt_path}")
            animations = visualize_most_likely(
                demo, p1_model, solver.p1_params, p2_model, solver.p2_params, fps=6
            )
            for i, (_, ani) in enumerate(animations):
                gif_path = Path(solver.anim_dir) / f"type{i}_iter{epoch:06d}.gif"
                ani.save(gif_path, writer="pillow", fps=10)
                print(f"[viz] Saved animation to {gif_path}")

    print("Training complete.")
       # ----- training curves -----
    plot_run_jsonl(solver.log_path, save_png=str(Path(solver.run_dir) / "training_curves.png"))
    print(f"[plot] training_curves.png saved in {solver.run_dir}")

if __name__ == "__main__":
    main()

