"""
football_demo.ipynb
================================================
"""
import importlib
from IPython.display import display
import torch
from torch import Tensor
from pathlib import Path

import numpy as np
import math
import random
SEED       = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------
# 1. Instantiate the game (Football realisation) ----------------------
# ---------------------------------------------------------------------
import game
importlib.reload(game)
from game import FootballGame, default_football_spec

spec   = default_football_spec(N=5, horizon=1.5, dt=0.5, device="cuda", n_substeps=4)

batch_size = 1
game   = FootballGame(spec, batch_size)
demo   = FootballGame(spec, batch_size)

# ---------------------------------------------------------------------
# 2. Instantiate the players ------------------------------------
# ---------------------------------------------------------------------
import player
importlib.reload(player)
from player import CAMS_INFORMED, BR

player_spec = {
    'hidden': 32,           # policy network width
    'temperature': 1.0,     # logit temperature for mixed strategy: higher = less entropy
    "init_scale": 1e-2,     # random initialization for root strategy parameters
    "ent_thr_belief": 1e-2  # threshold of belief entropy for pruning
}

p1 = CAMS_INFORMED(game, player_spec)
p2 = BR(game, player_spec)

# ---------------------------------------------------------------------
# 3. Instantiate the DS‑GDA solver ------------------------------------
# ---------------------------------------------------------------------
import dsgda_solver
importlib.reload(dsgda_solver)
from dsgda_solver import DSGDASolver

solver_spec = {
    'lr_p1': 3e-3,
    'lr_p2': 1e-2,
    'momentum': 0.6,
    'C2_p1': 10.0,
    'C2_p2': 10.0,
}

solver = DSGDASolver(game, p1, p2, solver_spec, prune=True, prune_every=100, prune_warmup=1000)

# ---------------------------------------------------------------------
# 4. Training loop  ----------------------------------------------------
# ---------------------------------------------------------------------
EPOCHS     = 1000      # number of DSGDA iterations
VIS_EVERY  = 100        # visualize solution frequency

for epoch in range(EPOCHS):
    stats = solver.step()

    if epoch % VIS_EVERY == 0:
        # save checkpoint **before** visualising
        ckpt_path = solver.save_checkpoint(epoch)
        print(f"[ckpt] saved {ckpt_path}")

        # ── helper: count paths given P1’s collapse mask ───────────────────────
        # active = solver._count_active_p1()
        # print(f"#P1 active parameters: {active}")

        print(f"[{epoch:04d}] L={stats['L']:+.4f} "
                f"||g_p1||={stats['g_p1']:.4f} "
                f"||g_p2||={stats['g_p2']:.4f} "
                f"S={stats['n_seq']}  "
                f"t_prune={stats['t_prune']:.1f}ms "
                f"t_loss={stats['t_loss']:.1f}ms "
                f"t_back={stats['t_backward']:.1f}ms "
                f"t_mom={stats['t_momentum']:.1f}ms "
                f"t_step={stats['t_step']:.1f}ms "
                f"wall={stats['wall_ms']:.1f}ms")
        
        html_list = demo.visualize_most_likely(solver.p1, solver.p2, fps=5, save_dir="animations")
        for i, (html, ani) in enumerate(html_list):
            gif_path = Path(solver.anim_dir) / f"type{i:02d}_{solver.stamp}_iter{epoch:04d}.gif"
            ani.save(gif_path, writer="pillow", fps=5)
            display(html)

        # 1) build the current tree (uses P1’s collapse flags automatically)
        # G = build_tree(
        #     solver.p1,               # current Player-1 strategy
        #     game.P0[0],              # prior belief vector
        #     game.I, game.K,          # roster size & horizon
        #     ent_thr=1e-3              # entropy threshold for “revealing”
        # )

        # 2) draw into a temporary PNG and show it
        # with tempfile.TemporaryDirectory() as tmp:
            # png_path = os.path.join(tmp, f"tree_epoch{epoch:04d}.png")
        # draw_tree(G, None)
        # plt.show()

print("Training complete.")


## Save training stats as an image
from util import plot_run
plot_run(solver.log_path, save_png=Path(solver.run_dir)/"training_curves.png")
