"""
football_demo.ipynb
================================================
"""
import importlib
from IPython.display import display
import torch
from torch import Tensor

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

spec   = default_football_spec(N=5, horizon=1.2, dt=0.12, device="cpu", n_substeps=4)

batch_size = 1
game   = FootballGame(spec, batch_size)

# ---------------------------------------------------------------------
# 2. Instantiate the players ------------------------------------
# ---------------------------------------------------------------------
import player
importlib.reload(player)
from player import CAMS_INFORMED, BR

player_spec = {
    'hidden': 32,       # policy network width
    'temperature': 1.0  # logit temperature for mixed strategy: higher = less entropy
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
    'lr_p1': 3e-4,
    'lr_p2': 1e-3,
    'momentum': 0.6,
    'C2_p1': 10.0,
    'C2_p2': 10.0,
}

solver = DSGDASolver(game, p1, p2, solver_spec)

# ---------------------------------------------------------------------
# 4. Training loop  ----------------------------------------------------
# ---------------------------------------------------------------------
EPOCHS     = 100_000      # number of DSGDA iterations
VIS_EVERY  = 100        # visualize solution frequency

for epoch in range(EPOCHS):
    stats = solver.step()

    if epoch % VIS_EVERY == 0:
        print(f"[{epoch:04d}]  L = {stats['L']:+.4f}  "
              f"‖m₁‖={stats['g_p1']:.3f}  ‖m₂‖={stats['g_p2']:.3f}")
        gif_files = solver.game.save_type_animations(
            solver.p1, solver.p2,
            iteration=epoch, fps=5)
        print("saved:", gif_files)

print("Training complete.")