import hexner_full_fix_3
from open_spiel.python import policy
from open_spiel.python.jax import deep_cfr
import pyspiel
import pickle
from flax.training import checkpoints
import os
import shutil

full_game = pyspiel.load_game_as_turn_based("python_hexner_full")

deep_cfr_solver = deep_cfr.DeepCFRSolver(
    full_game, 
    policy_network_layers=(256, 256),
    advantage_network_layers=(256, 256), 
    num_iterations=1000, 
    num_traversals=5, 
    learning_rate=1e-3,
    batch_size_advantage=1024, 
    batch_size_strategy=10000, 
    memory_capacity=1e7,
    policy_network_train_steps=5000,
    advantage_network_train_steps=1000,
    reinitialize_advantage_networks=True,
)

logging_root='cfr_policy_3/'

if os.path.exists(logging_root):
    shutil.rmtree(logging_root)

os.makedirs(logging_root)

chkpt_pth = os.path.abspath(logging_root)
save_iterations = 100
_, advantage_losses, policy_loss = deep_cfr_solver.solve(chkpt_pth, save_iterations)

# cfr_policy = policy.tabular_policy_from_callable(
#             full_game, deep_cfr_solver.action_probabilities)

# file = open('deep_cfr_policy', 'wb')
# pickle.dump(cfr_policy, file)
