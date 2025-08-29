# football_demo_jax.py
# =========================================================
# Clean driver for JAX-based training of the football game.
#  • No tree pruning, no tree visualisation.
#  • Training in JAX; visualisation CPU-only with Matplotlib.
#  • Unified switches at the top.
#  • Logs JSONL and saves checkpoints like the PyTorch version.
# =========================================================

from __future__ import annotations
import os, json, math, time, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---- JAX/Flax ---------------------------------------------------------
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

# ---- Plotting (CPU) ---------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # headless/back-end agnostic CPU rendering
import matplotlib.pyplot as plt
from matplotlib import animation

# add tqdm for progress bar
from tqdm import tqdm

# ---- Style (DeepMind-ish minimal aesthetic) --------------------------
DM_COLORS = {
    "off": "red",       # offence in red
    "def": "blue",      # defence in blue
    "star": "#FFC107",  # amber for RB/QB highlight
}
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E0E0E0",
    "axes.grid": True,
    "grid.color": "#EAEAEA",
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
})

# ---- Local modules ----------------------------------------------------
from game_jax import default_football_spec, FootballGame
from player_jax import CAMSInformed, BR
from solver_jax_eff_test import DSGDASolver


# =========================================================
# Global switches & defaults
# =========================================================
SEED        = 1
BATCH_SIZE  = 1
EPOCHS      = 25_000
VIS_EVERY   = 2500

LOG_ROOT    = "Max/runs"   # run folders like PyTorch version
HORIZON     = 1.5
DT          = 0.5
N_SUBSTEPS  = 4
N_PLAYERS   = 11

# Debug / verbosity
PRINT_EVERY = 100    # iteration print cadence (stdout)
SHOW_DEBUG  = False # extra shape/NaN checks (kept minimal here)

# Player/solver hyperparams (parity with PyTorch)
PLAYER_SPEC = {
    "hidden": 128,
    "temperature": 1.0,     # used in viz misc; training uses raw logits
    "init_scale": 1e-1,
    "ent_thr_belief": 1e-2, # kept for parity; unused (no pruning)
}
SOLVER_SPEC = {
    "lr_p1": 3e-3,
    "lr_p2": 1e-2,
    "momentum": 0.6,
    "C2_p1": 10.0,
    "C2_p2": 10.0,
}


# =========================================================
# Initialisation helpers (Flax params for all steps)
# =========================================================
def _init_cams_params(rng, model: CAMSInformed, game: FootballGame):
    """Initialise root + all step nets so params exist for k=0..K-1."""
    B = 1
    dummy_obs = {
        "x": jnp.zeros((B, game.STATE_DIM), dtype=jnp.float32),
        "p": jnp.tile(game.P0.astype(jnp.float32), (B, 1)),
        "t": jnp.zeros((B,), dtype=jnp.float32),
    }

    params = model.init(jax.random.split(rng)[0], dummy_obs, 0)["params"]  # root
    base = unfreeze(params)
    for k in range(1, game.K):
        vk = model.init(jax.random.fold_in(rng, k), dummy_obs, k)["params"]
        add = unfreeze(vk)
        step_name = f"step{k}"
        if step_name in add:
            base[step_name] = add[step_name]
    return freeze(base)


def _init_br_params(rng, model: BR, game: FootballGame):
    B = 1
    dummy_obs = {
        "x": jnp.zeros((B, game.STATE_DIM), dtype=jnp.float32),
        "p": jnp.tile(game.P0.astype(jnp.float32), (B, 1)),
        "t": jnp.zeros((B,), dtype=jnp.float32),
    }
    params = model.init(jax.random.split(rng)[0], dummy_obs, 0)["params"]  # root
    base = unfreeze(params)
    for k in range(1, game.K):
        vk = model.init(jax.random.fold_in(rng, k), dummy_obs, k)["params"]
        add = unfreeze(vk)
        step_name = f"step{k}"
        if step_name in add:
            base[step_name] = add[step_name]
    return freeze(base)


# =========================================================
# CPU visualisation (no JIT; small batches; Matplotlib)
# =========================================================
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
    CPU-only rendering of one rollout per hidden type i★.
    • Type 0 (RB push): highlight RB.
    • Type 1 (QB throw): highlight QB.
    • Draw semi-transparent trajectories for all players.
    Returns a list of (html, ani).
    """
    outs = []

    for i_star in range(game.I):
        # Reset and set the hidden type deterministically
        state, _ = game.reset(batch_size=1)
        state = type(state)(x=state.x, t=state.t, p=state.p,
                            i_star=jnp.full((1,), i_star, dtype=jnp.int32),
                            w_last=state.w_last)
        obs = {"x": state.x, "p": state.p, "t": state.t}

        pos1, _, pos2, _ = _split_state(game, state.x)
        traj_off = [jnp.array(pos1[0]).copy()]
        traj_def = [jnp.array(pos2[0]).copy()]
        p_traj   = [float(state.p[0, 0])]
        times    = [0.0]

        for k in range(game.K):
            out = p1_model.apply({"params": p1_params}, obs, k)
            A = jax.nn.softmax(out["A_logits"][0], axis=-1)
            row = A[i_star]
            j_k = int(jnp.argmax(row))

            mu_tbl = out["μ"][0]
            u1 = mu_tbl[j_k][None, :]
            u2 = p2_model.apply({"params": p2_params}, obs, k)

            state, _ = game.step(state, u1, u2)
            state = type(state)(x=state.x, t=state.t,
                                p=game._bayes_update(state.p, A[None, ...], jnp.array([j_k], dtype=jnp.int32)),
                                i_star=state.i_star, w_last=state.w_last)
            obs = {"x": state.x, "p": state.p, "t": state.t}

            pos1, _, pos2, _ = _split_state(game, state.x)
            traj_off.append(jnp.array(pos1[0]).copy())
            traj_def.append(jnp.array(pos2[0]).copy())
            p_traj.append(float(state.p[0, 0]))
            times.append((k + 1) * game.dt)

        # numpy stacks for animation
        import numpy as np
        off_np = np.stack([np.array(t) for t in traj_off], axis=0)  # (T+1, N, 2)
        def_np = np.stack([np.array(t) for t in traj_def], axis=0)
        T_frames = off_np.shape[0]

        # Choose which player to highlight per type
        star_idx = game.BALL_IDX if i_star == 0 else getattr(game, "QB_IDX", game.BALL_IDX)

        # ---- build figure ----
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(6, 8), gridspec_kw={"height_ratios": [4, 1]}
        )
        for side in ("top", "right"):
            ax_top.spines[side].set_visible(False)
            ax_bot.spines[side].set_visible(False)
        ax_top.set_xlim(-game.BOX_POS - .2, game.BOX_POS + .2)
        ax_top.set_ylim(-game.BOX_POS - .2, game.BOX_POS + .2)
        ax_top.set_aspect("equal")
        ax_top.set_title(f"Most-likely path – type {i_star}")

        # Static receivers/ball carrier highlight color
        scat_off = ax_top.scatter([], [], s=70, c=DM_COLORS["off"], edgecolors="black", linewidths=0.5)
        scat_def = ax_top.scatter([], [], s=70, c=DM_COLORS["def"], edgecolors="black", linewidths=0.5)
        star_sc  = ax_top.scatter([], [], s=140, marker="*", c=DM_COLORS["star"], edgecolors="#111111", lw=.6)

        # Trajectory lines (semi-transparent trails)
        off_trails = [ax_top.plot([], [], lw=1.5, alpha=0.35, color=DM_COLORS["off"], solid_capstyle="round")[0]
                      for _ in range(game.N)]
        def_trails = [ax_top.plot([], [], lw=1.5, alpha=0.35, color=DM_COLORS["def"], solid_capstyle="round")[0]
                      for _ in range(game.N)]

        # Belief subplot
        ax_bot.set_xlim(0, game.T)
        ax_bot.set_ylim(-0.05, 1.05)
        ax_bot.set_xlabel("time (s)")
        ax_bot.set_ylabel(f"belief p[0]")
        ax_bot.plot(times, p_traj, color="#222222")

        def init(off_np=off_np, def_np=def_np, scat_off=scat_off, scat_def=scat_def, star_sc=star_sc, off_trails=off_trails, def_trails=def_trails):
            empty = np.empty((0, 2))
            scat_off.set_offsets(empty)
            scat_def.set_offsets(empty)
            star_sc.set_offsets(empty)
            for ln in off_trails + def_trails:
                ln.set_data([], [])
            return (scat_off, scat_def, star_sc, *off_trails, *def_trails)

        def update(frame, off_np=off_np, def_np=def_np, star_idx=star_idx, scat_off=scat_off, scat_def=scat_def, star_sc=star_sc, off_trails=off_trails, def_trails=def_trails):
            # points
            scat_off.set_offsets(off_np[frame])
            scat_def.set_offsets(def_np[frame])
            star_sc.set_offsets(off_np[frame, star_idx])
            # trails up to current frame
            xs_off = off_np[:frame+1, :, 0]
            ys_off = off_np[:frame+1, :, 1]
            xs_def = def_np[:frame+1, :, 0]
            ys_def = def_np[:frame+1, :, 1]
            for i in range(game.N):
                off_trails[i].set_data(xs_off[:, i], ys_off[:, i])
                def_trails[i].set_data(xs_def[:, i], ys_def[:, i])
            return (scat_off, scat_def, star_sc, *off_trails, *def_trails)

        ani = animation.FuncAnimation(
            fig, update, frames=T_frames, init_func=init,
            blit=True, interval=1000 / fps, repeat=False
        )
        # NOTE: we intentionally do not close the figure here to avoid single-frame exports.

        # For notebook display compatibility (optional)
        try:
            from IPython.display import HTML
            html = HTML(ani.to_jshtml())
        except Exception:
            html = None

        outs.append((html, ani))

    return outs


# =========================================================
# Visual design smoketest (random walk trajectories)
# =========================================================
def save_visual_style_smoketest(out_dir: str, *, N: int = 11, T: int = 30, box: float = 1.6, fps: int = 8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    rng = np.random.default_rng(0)
    # random walks (bounded)
    pos_off = np.zeros((T, N, 2), dtype=np.float32)
    pos_def = np.zeros((T, N, 2), dtype=np.float32)
    pos_off[0] = rng.uniform(-box*0.9, -box*0.4, size=(N, 2))
    pos_def[0] = rng.uniform(box*0.2, box*0.9, size=(N, 2))
    for t in range(1, T):
        pos_off[t] = np.clip(pos_off[t-1] + 0.08 * rng.normal(size=(N, 2)), -box, box)
        pos_def[t] = np.clip(pos_def[t-1] + 0.08 * rng.normal(size=(N, 2)), -box, box)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-box-0.2, box+0.2); ax.set_ylim(-box-0.2, box+0.2); ax.set_aspect("equal")
    scat_off = ax.scatter([], [], s=70, c=DM_COLORS["off"], edgecolors="black", linewidths=0.5)
    scat_def = ax.scatter([], [], s=70, c=DM_COLORS["def"], edgecolors="black", linewidths=0.5)
    trails_off = [ax.plot([], [], lw=1.5, alpha=0.35, color=DM_COLORS["off"])[0] for _ in range(N)]
    trails_def = [ax.plot([], [], lw=1.5, alpha=0.35, color=DM_COLORS["def"])[0] for _ in range(N)]

    def init():
        scat_off.set_offsets(np.empty((0,2))); scat_def.set_offsets(np.empty((0,2)))
        for ln in trails_off + trails_def: ln.set_data([], [])
        return (scat_off, scat_def, *trails_off, *trails_def)

    def update(f):
        scat_off.set_offsets(pos_off[f]); scat_def.set_offsets(pos_def[f])
        for i in range(N):
            trails_off[i].set_data(pos_off[:f+1, i, 0], pos_off[:f+1, i, 1])
            trails_def[i].set_data(pos_def[:f+1, i, 0], pos_def[:f+1, i, 1])
        return (scat_off, scat_def, *trails_off, *trails_def)

    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=1000/fps)
    out_path = out_dir / "visual_style_smoketest.gif"
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    return str(out_path)


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
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    # ----- seed -----
    key = jax.random.PRNGKey(SEED)

    # ----- game & spec -----
    spec = default_football_spec(
        N=N_PLAYERS, horizon=HORIZON, dt=DT, device="cpu",
        n_substeps=N_SUBSTEPS
    )
    game = FootballGame(spec, batch_size=BATCH_SIZE)
    demo = FootballGame(spec, batch_size=1)  # separate env for viz

    # ----- models -----
    p1_model = CAMSInformed(
        I=game.I, d=game.ACTION_DIM, feat_dim=game.FEAT_DIM,
        K=game.K, box_acc=game.BOX_ACC, hidden=PLAYER_SPEC["hidden"]
    )
    p2_model = BR(
        d=game.ACTION_DIM, feat_dim=game.FEAT_DIM,
        K=game.K, box_acc=game.BOX_ACC, hidden=PLAYER_SPEC["hidden"]
    )

    # ----- params -----
    key, k1, k2 = jax.random.split(key, 3)
    p1_params = _init_cams_params(k1, p1_model, game)
    p2_params = _init_br_params(k2, p2_model, game)

    # ----- solver -----
    solver = DSGDASolver(
        game, p1_model, p1_params, p2_model, p2_params,
        SOLVER_SPEC, log_root=LOG_ROOT
    )

    print(f"[run] dir: {solver.run_dir}")
    print(f"[cfg] I={game.I}  K={game.K}  N={game.N}  d={game.ACTION_DIM}  S={solver.S_paths}")

    # quick one-off style smoketest (can be commented out later)
    try:
        test_gif = save_visual_style_smoketest(solver.anim_dir)
        print(f"[viz-test] wrote {test_gif}")
    except Exception as _e:
        pass

    # ----- training loop -----
    for epoch in tqdm(range(EPOCHS)):  # add tqdm for progress bar
        stats = solver.step()

        if (epoch % PRINT_EVERY) == 0:
            print(
                f"[{epoch:04d}] L={stats['L']:+.4f} "
                f"||g_p1||={stats['g_p1']:.4f} "
                f"||g_p2||={stats['g_p2']:.4f} "
                f"S={stats['n_seq']}  "
                f"t_loss={stats['t_loss']:.1f}ms "
                f"t_mom={stats['t_momentum']:.1f}ms "
                f"wall={stats['wall_ms']:.1f}ms"
            )

        if (epoch % VIS_EVERY) == 0 or (epoch == EPOCHS-1):  # also save last checkpoint + anims
            # 1) save checkpoint BEFORE viz (parity with PyTorch)
            ckpt_path = solver.save_checkpoint(epoch)
            print(f"[ckpt] saved {ckpt_path}")

            # 2) CPU viz of most-likely sequence per type, save GIFs
            html_list = visualize_most_likely(
                demo, p1_model, solver.p1_params, p2_model, solver.p2_params, fps=6
            )
            for i, (html, ani) in enumerate(html_list):
                gif_path = Path(solver.anim_dir) / f"type{i:02d}_{solver.stamp}_iter{epoch:04d}.gif"
                ani.save(gif_path, writer="pillow", fps=6)
            print(f"[viz] wrote {len(html_list)} animations → {solver.anim_dir}")

    print("Training complete.")

    # ----- training curves -----
    plot_run_jsonl(solver.log_path, save_png=str(Path(solver.run_dir) / "training_curves.png"))
    print(f"[plot] training_curves.png saved in {solver.run_dir}")


if __name__ == "__main__":
    main()
