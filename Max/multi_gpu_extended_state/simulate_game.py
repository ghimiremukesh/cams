#!/usr/bin/env python3
# =========================================================
# football_rollout_all_gifs.py
# CPU-only: load ckpt, sample hidden type per play, sample rows each step,
# roll out N plays, and save a GIF for EVERY play.
# Optional: also save an NPZ with all rollouts.
# =========================================================

from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import Any, Dict, List

# ---- Force CPU (single device) BEFORE importing jax ----
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization as flax_serial
from flax import jax_utils as flax_utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

# ---- Project imports: match your training driver ----
from game_jax import default_football_spec, FootballGame
from player_jax import CAMSInformed, BR


# =========================================================
# Helpers aligned with your demo
# =========================================================
def _split_state(game: FootballGame, x: jnp.ndarray):
    B = x.shape[0]
    pos1 = x[:, 0:2*game.N].reshape(B, game.N, 2)
    vel1 = x[:, 2*game.N:4*game.N].reshape(B, game.N, 2)
    pos2 = x[:, 4*game.N:6*game.N].reshape(B, game.N, 2)
    vel2 = x[:, 6*game.N:8*game.N].reshape(B, game.N, 2)
    return pos1, vel1, pos2, vel2


def _find_latest_ckpt(ckpt_dir: Path) -> Path:
    """Prefer latest by meta 'iter', else newest msgpack mtime."""
    ckpt_dir = Path(ckpt_dir)
    metas = sorted(ckpt_dir.glob("ckpt_*.meta.json"))
    best_path, best_iter = None, -1
    for mp in metas:
        try:
            meta = json.loads(mp.read_text())
            it = int(meta.get("iter", -1))
            p = ckpt_dir / f"ckpt_{it}.msgpack"
            if it > best_iter and p.exists():
                best_iter, best_path = it, p
        except Exception:
            pass
    if best_path:
        return best_path
    # fallback
    cands = sorted(ckpt_dir.glob("ckpt_*.msgpack"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return cands[0]


def _load_ckpt(ckpt_path: Path) -> Dict[str, Any]:
    with open(ckpt_path, "rb") as fh:
        blob = fh.read()
    state_tree = flax_serial.from_bytes(None, blob)
    if not isinstance(state_tree, dict):
        raise ValueError("Unexpected checkpoint format (not a dict).")
    if "p1_params" not in state_tree or "p2_params" not in state_tree:
        raise ValueError("Checkpoint missing p1_params/p2_params.")
    return state_tree


def _load_meta(ckpt_path: Path) -> Dict[str, Any] | None:
    stem = ckpt_path.stem  # e.g., "ckpt_25000"
    meta_path = ckpt_path.with_name(stem + ".meta.json")
    return json.loads(meta_path.read_text()) if meta_path.exists() else None


def build_game_and_models(meta: Dict[str, Any] | None,
                          N: int | None, horizon: float | None, dt: float | None, n_substeps: int | None):
    # Prefer meta["game_spec"] if available
    if meta and "game_spec" in meta:
        gs = meta["game_spec"]
        N = int(gs.get("N", N if N is not None else 11))
        horizon = float(gs.get("horizon", horizon if horizon is not None else 2.5))
        dt = float(gs.get("dt", dt if dt is not None else 0.25))
        n_substeps = int(gs.get("n_substeps", n_substeps if n_substeps is not None else 4))
    else:
        N = 11 if N is None else N
        horizon = 2.5 if horizon is None else horizon
        dt = 0.25 if dt is None else dt
        n_substeps = 4 if n_substeps is None else n_substeps

    spec = default_football_spec(N=N, horizon=horizon, dt=dt, n_substeps=n_substeps)
    game = FootballGame(spec, batch_size=1)  # single-CPU rollout
    p1_model = CAMSInformed(I=game.I, d=game.ACTION_DIM, feat_dim=game.FEAT_DIM, K=game.K,
                            box_acc=game.BOX_ACC, hidden=128)
    p2_model = BR(d=game.ACTION_DIM, feat_dim=game.FEAT_DIM, K=game.K,
                  box_acc=game.BOX_ACC, hidden=128)
    return game, p1_model, p2_model


@jax.jit
def _softmax(logits):
    return jax.nn.softmax(logits, axis=-1)


def _sample_type_from_prior(game: FootballGame, rng: jax.Array) -> tuple[int, jax.Array]:
    """Sample hidden type i⋆ from prior P0."""
    prior = game.P0
    prior = prior[0] if prior.ndim == 2 else prior  # shape (I,)
    rng, sub = jax.random.split(rng)
    i_star = int(jax.random.categorical(sub, jnp.log(prior)))
    return i_star, rng


def _team_action_per_player(u: np.ndarray, N: int) -> np.ndarray:
    """Per-player L2 magnitudes (N,) from a team action vector."""
    u = np.asarray(u).squeeze()
    D = int(u.shape[-1]) if u.ndim >= 1 else 0
    if D == 2 * N:                # (N,2)
        return np.linalg.norm(u.reshape(N, 2), axis=1)
    elif N > 0 and D % N == 0 and D > 0:
        return np.linalg.norm(u.reshape(N, D // N), axis=1)
    # fallback: average magnitude
    mag = float(np.linalg.norm(u)) / max(N, 1)
    return np.full((N,), mag, dtype=float)


# =========================================================
# Rollout: sample hidden type once; sample rows at each step
# =========================================================
def simulate_one_play(
    game: FootballGame,
    p1_model: CAMSInformed,
    p2_model: BR,
    p1_params,
    p2_params,
    rng: jax.Array,
) -> tuple[Dict[str, Any], jax.Array]:
    # 1) sample hidden type from prior
    i_star, rng = _sample_type_from_prior(game, rng)

    # 2) reset environment and set type
    state, _ = game.reset(batch_size=1)
    state = state.tree_replace(i_star=jnp.full((1,), i_star, dtype=jnp.int32))
    obs = {"x": state.x, "p": state.p, "t": state.t}

    # storage
    x_seq = [np.array(state.x[0])]
    p_traj = [float(state.p[0, 0])]
    times  = [0.0]
    j_seq  = []
    u1_seq, u2_seq = [], []

    # 3) rollout with categorical row sampling each step
    for k in range(game.K):
        out = p1_model.apply({"params": p1_params}, obs, k)
        A = _softmax(out["A_logits"][0])          # (I, J)
        row_probs = np.array(A[i_star])           # (J,)
        rng, sub = jax.random.split(rng)
        j_k = int(jax.random.categorical(sub, jnp.log(row_probs)))

        mu_tbl = out["μ"][0]                      # [J, D_off]
        u1 = mu_tbl[j_k][None, :]                 # (1, D_off)
        u2 = p2_model.apply({"params": p2_params}, obs, k)

        state, _ = game.step(state, u1, u2)
        state = state.tree_replace(p=game._bayes_update(state.p, A[None, ...], jnp.array([j_k])))
        obs = {"x": state.x, "p": state.p, "t": state.t}

        x_seq.append(np.array(state.x[0]))
        p_traj.append(float(state.p[0, 0]))
        u1_seq.append(np.array(u1[0]))
        u2_seq.append(np.array(u2[0] if getattr(u2, "ndim", 1) == 2 else u2))
        times.append((k + 1) * game.dt)
        j_seq.append(j_k)

    # convenience position tensors for visualization
    x_stack = np.stack(x_seq, axis=0)   # (T, STATE_DIM)
    pos1, _, pos2, _ = _split_state(game, jnp.asarray(x_stack[None, ...])[0])

    play = {
        "type": int(i_star),
        "x_seq": x_stack,                    # (T, STATE_DIM)
        "pos_off_seq": np.array(pos1),      # (T, N, 2)
        "pos_def_seq": np.array(pos2),      # (T, N, 2)
        "p_traj": np.array(p_traj),         # (T,)
        "u1_seq": np.stack(u1_seq, axis=0), # (K, D_off)
        "u2_seq": np.stack(u2_seq, axis=0), # (K, D_def)
        "j_seq": np.array(j_seq, dtype=np.int32),  # (K,)
        "times": np.array(times),
        "dt": float(game.dt),
        "K": int(game.K),
    }
    return play, rng


def simulate_batch(
    game: FootballGame,
    p1_model: CAMSInformed,
    p2_model: BR,
    p1_params,
    p2_params,
    n_plays: int,
    seed: int,
) -> List[Dict[str, Any]]:
    plays: List[Dict[str, Any]] = []
    rng = jax.random.PRNGKey(seed)
    for _ in range(n_plays):
        play, rng = simulate_one_play(game, p1_model, p2_model, p1_params, p2_params, rng)
        plays.append(play)
    return plays


# =========================================================
# GIFs for ALL plays (heatmaps appear AFTER step)
# =========================================================
def _build_mags_TxN_from_actions(play: Dict[str, Any], N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert action sequences u1_seq/u2_seq (K steps) to per-player magnitudes
    aligned with T=K+1 frames. Prepend zeros at t=0 so blobs appear only after step.
    """
    u1_seq = play["u1_seq"]  # (K, D_off)
    u2_seq = play["u2_seq"]  # (K, D_def)
    K = u1_seq.shape[0]
    off = np.zeros((K + 1, N), dtype=float)
    dff = np.zeros((K + 1, N), dtype=float)
    for k in range(K):
        off[k + 1] = _team_action_per_player(u1_seq[k], N)
        dff[k + 1] = _team_action_per_player(u2_seq[k], N)
    return off, dff


def animate_one_play_to_file(play: Dict[str, Any], game: FootballGame,
                             out_path: Path, fps: int = 8, fmt: str = "gif"):
    off_np = play["pos_off_seq"]  # (T, N, 2)
    def_np = play["pos_def_seq"]  # (T, N, 2)
    T_frames, N = off_np.shape[0], off_np.shape[1]
    if T_frames <= 1:
        print(f"[viz] skip (not enough frames): {out_path.name}")
        return

    off_mags_np, def_mags_np = _build_mags_TxN_from_actions(play, N)
    max_off = float(np.max(off_mags_np)) if off_mags_np.size else 1.0
    max_def = float(np.max(def_mags_np)) if def_mags_np.size else 1.0
    max_off = max(max_off, 1e-12)
    max_def = max(max_def, 1e-12)

    # Styling
    OFF_LINE = "#c62828"; DEF_LINE = "#1565c0"
    OFF_CMAP = plt.cm.Reds; DEF_CMAP = plt.cm.Blues
    S_OUTLINE = 160.0; S_POS_DOT = 28.0; TRAIL_LW = 1.6; OUTLINE_LW = 1.6
    R_FIXED = 0.05; HEAT_ALPHA = 0.35; HEAT_GAMMA = 0.70; HEAT_BASE = 0.30
    Z_BLOBS = 2.0; APPEAR_DELAY = 1

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    xmin, xmax = -game.BOX_POS - 0.2, game.BOX_POS + 0.2
    ymin, ymax = -game.BOX_POS - 0.2, game.BOX_POS + 0.2
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.set_title(f"Sampled rollout (type={play['type']})")
    for side in ("top", "right"): ax.spines[side].set_visible(False)

    off_trails = [
        ax.plot([], [], lw=TRAIL_LW, alpha=0.75, color=OFF_LINE, linestyle=":", solid_capstyle="round", zorder=2.6)[0]
        for _ in range(N)
    ]
    def_trails = [
        ax.plot([], [], lw=TRAIL_LW, alpha=0.75, color=DEF_LINE, linestyle=":", solid_capstyle="round", zorder=2.6)[0]
        for _ in range(N)
    ]
    off_outline = ax.scatter([], [], s=S_OUTLINE, facecolors="none", edgecolors="black",
                             linewidths=OUTLINE_LW, marker="o", zorder=3.4)
    def_outline = ax.scatter([], [], s=S_OUTLINE, facecolors="none", edgecolors="black",
                             linewidths=OUTLINE_LW, marker="s", zorder=3.4)
    off_pos = ax.scatter([], [], s=S_POS_DOT, c=OFF_LINE, edgecolors="none", alpha=0.95, zorder=3.2)
    def_pos = ax.scatter([], [], s=S_POS_DOT, c=DEF_LINE, edgecolors="none", alpha=0.95, zorder=3.2)

    off_blob_patches: List[Circle] = []
    def_blob_patches: List[Circle] = []

    def _add_blobs(ax_, centers, mags, cmap, wmax, patch_store):
        if centers.size == 0:
            return
        w = np.clip(mags / (wmax + 1e-12), 0.0, 1.0)
        w_plot = np.power(w, HEAT_GAMMA)
        w_col = HEAT_BASE + (1.0 - HEAT_BASE) * w_plot
        cols = cmap(w_col)
        if cols.ndim == 1: cols = cols[None, :]
        for (x, y), c in zip(centers, cols):
            color = (float(c[0]), float(c[1]), float(c[2]), HEAT_ALPHA)
            circ = Circle((float(x), float(y)), radius=R_FIXED, facecolor=color, edgecolor='none', zorder=Z_BLOBS)
            ax_.add_patch(circ)
            patch_store.append(circ)

    def init():
        empty = np.empty((0, 2))
        off_outline.set_offsets(empty); def_outline.set_offsets(empty)
        off_pos.set_offsets(empty); def_pos.set_offsets(empty)
        for ln in off_trails + def_trails: ln.set_data([], [])
        for p in off_blob_patches + def_blob_patches:
            try: p.remove()
            except Exception: pass
        off_blob_patches.clear(); def_blob_patches.clear()
        return off_outline, def_outline, off_pos, def_pos, *off_trails, *def_trails

    def update(frame):
        # positions for current frame
        off_outline.set_offsets(off_np[frame])
        def_outline.set_offsets(def_np[frame])
        off_pos.set_offsets(off_np[frame])
        def_pos.set_offsets(def_np[frame])

        # dotted trails up to current frame
        xs_off = off_np[:frame+1, :, 0]; ys_off = off_np[:frame+1, :, 1]
        xs_def = def_np[:frame+1, :, 0]; ys_def = def_np[:frame+1, :, 1]
        for i in range(N):
            off_trails[i].set_data(xs_off[:, i], ys_off[:, i])
            def_trails[i].set_data(xs_def[:, i], ys_def[:, i])

        # stamp only the step that just completed (appear AFTER step)
        t_stamp = frame - APPEAR_DELAY
        if t_stamp >= 0:
            _add_blobs(ax, off_np[t_stamp], off_mags_np[t_stamp], OFF_CMAP, max_off, off_blob_patches)
            _add_blobs(ax, def_np[t_stamp], def_mags_np[t_stamp], DEF_CMAP, max_def, def_blob_patches)

        return off_outline, def_outline, off_pos, def_pos, *off_trails, *def_trails

    ani = animation.FuncAnimation(fig, update, frames=T_frames, init_func=init,
                                  blit=False, interval=1000 / fps, repeat=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if fmt == "mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            ani.save(out_path, writer=writer)
        else:
            writer = animation.PillowWriter(fps=fps)
            ani.save(out_path, writer=writer)
        print(f"[viz] saved {out_path}")
    except Exception as e:
        print(f"[viz:warn] could not save {out_path}: {e}")
    plt.close(fig)


def render_all_gifs(plays: List[Dict[str, Any]], game: FootballGame,
                    out_dir: Path, fps: int = 8, fmt: str = "gif"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, play in enumerate(plays):
        fname = f"play_{i:03d}_type{play['type']}.{fmt}"
        animate_one_play_to_file(play, game, out_dir / fname, fps=fps, fmt=fmt)


# =========================================================
# Optional NPZ saver (zipped NumPy archive)
# =========================================================
def save_plays_npz(plays: List[Dict[str, Any]], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packed = {f"play_{i}": np.array(plays[i], dtype=object) for i in range(len(plays))}
    np.savez_compressed(out_path, **packed)
    print(f"[save] plays -> {out_path}  (npz = zipped NumPy arrays)")


# =========================================================
# CLI
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Sample hidden type, roll out plays, and save GIF for EVERY play.")
    ap.add_argument("--ckpt_dir", type=str, default="ckpt", help="Directory with ckpt_*.msgpack (+ meta).")
    ap.add_argument("--tag", type=str, default="", help="Checkpoint tag (iter). If empty, pick latest.")
    ap.add_argument("--n_plays", type=int, default=16, help="# plays to simulate.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed.")
    ap.add_argument("--save_dir", type=str, default="out_rollouts", help="Output directory.")
    ap.add_argument("--fps", type=int, default=8, help="FPS for GIFs.")
    ap.add_argument("--video_format", type=str, default="gif", choices=["gif", "mp4"], help="Animation format.")
    # fallbacks if meta missing
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--n_substeps", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)

    # choose checkpoint
    if args.tag:
        ckpt_path = ckpt_dir / f"ckpt_{args.tag}.msgpack"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    else:
        ckpt_path = _find_latest_ckpt(ckpt_dir)

    meta = _load_meta(ckpt_path)
    state = _load_ckpt(ckpt_path)
    print(f"[ckpt] loaded {ckpt_path} (iter={state.get('iter', '?')})")

    # build env + models on CPU
    game, p1_model, p2_model = build_game_and_models(meta, args.N, args.horizon, args.dt, args.n_substeps)


    p1_params = state["p1_params"]
    p2_params = state["p2_params"]

    # simulate all plays
    plays = simulate_batch(game, p1_model, p2_model, p1_params, p2_params,
                           n_plays=args.n_plays, seed=args.seed)

    # save GIF/MP4 for EVERY play
    out_dir = Path(args.save_dir)
    render_all_gifs(plays, game, out_dir=out_dir, fps=args.fps, fmt=args.video_format)


    print(f"[done] generated {len(plays)} plays and saved {args.video_format.upper()}s to {out_dir}")


if __name__ == "__main__":
    main()
