from functools import partial

import numpy as np
import torch
from Gym_Envs.HexnerEnv_w_reward_n_belief import GymHexnerEnv
from algorithms.PPO.PPO_Pub_Belief import PPOAgent
import matplotlib.pyplot as plt
from torch.utils._pytree import tree_map


def plot_simulation(traj_player1, traj_player2, actions_player1, actions_player2, ptype, beliefs):
    """
    Plots simulation results in a single figure with subplots:
      - Top: Trajectories for player1 and player2 (x-y positions).
      - Bottom left: Time series of player1's actions.
      - Bottom right: Time series of player2's actions.

    Parameters:
    - traj_player1: array of shape (num_steps, 2) for player1 positions.
    - traj_player2: array of shape (num_steps, 2) for player2 positions.
    - actions_player1: array of shape (num_steps, 2) for player1 actions.
    - actions_player2: array of shape (num_steps, 2) for player2 actions.
    """
    num_steps = traj_player1.shape[0]
    steps = np.arange(num_steps)

    # Create one figure with a 2-row grid, where the bottom row has 2 subplots.
    # fig = plt.figure(figsize=(6, 5))
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2)

    # Top subplot: Trajectories
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(traj_player1[:, 0], traj_player1[:, 1], 'r.-', label="Player1 Trajectory")
    ax1.plot(traj_player2[:, 0], traj_player2[:, 1], 'b.-', label="Player2 Trajectory")
    ax1.scatter(traj_player1[-1, 0], traj_player1[-1, 1], color='red', marker='*')
    ax1.scatter(traj_player2[-1, 0], traj_player2[-1, 1], color='blue', marker='*')
    ax1.scatter(0, 1, edgecolor='magenta', marker = 'o', facecolor='white')
    ax1.scatter(0, -1, edgecolor='magenta', marker='o', facecolor='white')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"Player Trajectories; Type: {1 if (ptype == np.array([1, 0])).all() else 2}")
    ax1.legend()
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_xlim([-1, 1])
    # ax1.grid(True)

    # Bottom subplots in one row for actions.
    # Create two subplots side by side.
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(actions_player1[:, 0], actions_player1[:, 1], 'r.-', label="Action (Player1)")
    # ax2.set_xlabel("Step")
    # ax2.set_ylabel("Action Value")
    # ax2.set_title("Player1 Actions")
    # ax2.set_xlim([-1, 1])
    # ax2.set_ylim(-2, 2)
    ax2.legend()
    ax2.grid(True)

    ax2.plot( actions_player2[:, 0], actions_player2[:, 1], 'b.-', label="Action (Player2)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Actions")
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(beliefs)
    ax3.set_xlabel("time")
    ax3.set_ylabel("p")
    ax3.set_title("Belief")

    plt.tight_layout()
    plt.show()

# def process_obs_player1(obs, include_history):
#     """Process the observation for player1.
#     If include_history is True, it concatenates state (8), player_type (2) and flattened action_history (max_game_length*2).
#     Otherwise, it concatenates state and player_type only.
#     """
#     if include_history:
#         return np.concatenate([
#             obs["player1"]["state"],
#             obs["player1"]["player_type"],
#             obs["player1"]["action_history"].reshape(-1, )
#         ])
#     else:
#         return np.concatenate([obs["player1"]["state"], obs["player1"]["player_type"]])
#
#
# def process_obs_player2(obs, include_history):
#     """Process the observation for player2.
#     If include_history is True, it concatenates state (8) and the flattened action_history (max_game_length*2).
#     Otherwise, it simply returns the state.
#     """
#     if include_history:
#         return np.concatenate([
#             obs["player2"]["state"],
#             obs["player2"]["action_history"].reshape(-1, )
#         ])
#     else:
#         return obs["player2"]["state"]



def run_episode(env, agent1, agent2, device, include_history, seed=42):
    """
    Runs one evaluation episode while recording:
      - trajectory: the (x, y) positions of each player over time.
      - actions: the actions taken by each player at every step.
    Returns:
      total_reward (scalar, from player1's perspective),
      traj_player1, traj_player2: arrays of shape (num_steps, 2),
      actions_player1, actions_player2: arrays of shape (num_steps, 2).
    """
    # seed = 10 for type 2 and 42 for type 1
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    traj_player1 = []  # record player1's positions (x,y)
    traj_player2 = []  # record player2's positions (x,y)
    traj_player1.append(obs["player1"]["state"][1:3])
    traj_player2.append(obs["player2"]["state"][5:7])
    actions_player1 = []  # record actions (2-dim) for player1
    actions_player2 = []  # record actions (2-dim) for player2
    beliefs = []
    player_type = obs["player1"]["player_type"]
    beliefs.append(obs["player1"]["belief"])

    while not done:
        # Process observation for each player.
        cast_to_tensor = partial(torch.tensor, dtype=torch.float32)
        obs_p1 = tree_map(cast_to_tensor, obs["player1"])
        obs_p1 = tree_map(torch.flatten, obs_p1)
        reshape = partial(torch.reshape, shape=(1, -1))
        obs_p1 = tree_map(reshape, obs_p1)
        obs_p2 = tree_map(cast_to_tensor, obs["player2"])
        obs_p2 = tree_map(torch.flatten, obs_p2)
        obs_p2 = tree_map(reshape, obs_p2)
        # obs_p1 = process_obs_player1(obs, include_history=include_history)
        # obs_p2 = process_obs_player2(obs, include_history=include_history)
        # obs_tensor1 = torch.tensor(obs_p1, dtype=torch.float32).to(device)
        # obs_tensor2 = torch.tensor(obs_p2, dtype=torch.float32).to(device)

        with torch.no_grad():
            a1, _, _, _, b = agent1.model.get_action(obs_p1)
            a2, _, _, _, _ = agent2.model.get_action(obs_p2)

        a1_np = a1.cpu().numpy().flatten()
        a2_np = a2.cpu().numpy().flatten()

        # Record actions.
        actions_player1.append(a1_np)
        actions_player2.append(a2_np)
        beliefs.append(b.item())

        # Build action dictionary.
        actions_dict = {"player1": a1_np, "player2": a2_np, "belief": b}

        # Gymnasium API: step returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(actions_dict)
        # Record trajectory; assume obs["player1"]["state"] contains [x1, y1, vx1, vy1, ...]
        traj_player1.append(obs["player1"]["state"][1:3])
        traj_player2.append(obs["player2"]["state"][5:7])
        done = terminated or truncated
        total_reward += reward  # since reward is scalar (player1's perspective)

    return total_reward, np.array(traj_player1), np.array(traj_player2), np.array(actions_player1), np.array(
        actions_player2), np.array(beliefs), player_type


if __name__ == "__main__":
    device = torch.device("cpu") # always cpu for eval

    # Create the environment.
    env = GymHexnerEnv(
        dt=0.1,
        max_game_length=None,  # let the env compute max_game_length as int(1/dt) if None
        ux_max= 12,
        uy_max= 12,
        dx_max = 12,
        dy_max= 12,
        p=0.5,
        render_on=False,  # No rendering in this eval script.
        include_action_history=False  # Change to True if you want to use action history.
    )

    # Reset the environment once.
    # env.reset(seed=42)

    # Set the input dimensions based on the flag.
    include_action_history = False
    if include_action_history:
        action_hist_dim = env.max_game_length * 4
    else:
       action_hist_dim = 0

    # Instantiate agents (using the same hyperparameters as training if desired).
    agent1 = PPOAgent(
        state_dim=9,
        belief_dim=1,
        action_dim=2,
        num_type=2,
        action_bounds=[12, 12],
        hidden_sizes=[64, 64],
        action_hist_dim=action_hist_dim,
        learning_rate=0.00025,  # The learning rate is not important here, as we won't update.
        clip_coef=0.2,
        ppo_epochs=10,
        mini_batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        device=device
    )
    agent2 = PPOAgent(
        state_dim=9,
        belief_dim=1,
        action_dim=2,
        num_type=3,
        action_bounds=[12, 12],
        action_hist_dim=action_hist_dim,
        hidden_sizes=[64, 64],
        has_private_type=False,
        learning_rate=0.00025,
        clip_coef=0.2,
        ppo_epochs=10,
        mini_batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        device=device
    )

    # Load the trained models from checkpoint files.
    # Adjust these paths as needed.
    # checkpoint_agent1 = "saved_models/Pub_belief/MMD/4-steps/agent1_final.pt"
    # checkpoint_agent2 = "saved_models/Pub_belief/MMD/4-steps/agent2_final.pt"

    checkpoint_agent1 = "saved_models/FINAL/PPO_SIMUL/10-steps/agent1_final.pt"
    checkpoint_agent2 = "saved_models/FINAL/PPO_SIMUL/10-steps/agent2_final.pt"

    agent1.model.load_state_dict(torch.load(checkpoint_agent1, map_location=device))
    agent2.model.load_state_dict(torch.load(checkpoint_agent2, map_location=device))
    agent1.model.eval()
    agent2.model.eval()

    # Run one evaluation episode.
    # Run one evaluation episode while recording data.
    seed = 42
    total_reward, traj_player1, traj_player2, actions_player1, actions_player2, beliefs, ptype = run_episode(
        env, agent1, agent2, device, include_action_history, seed=seed
    )

    print(f"Episode reward (player1 perspective): {total_reward}")
    # print(actions_player1)
    print(beliefs)
    plot_simulation(traj_player1, traj_player2, actions_player1, actions_player2, ptype, beliefs)

    # for saving
    import pandas as pd
    df = pd.DataFrame({'x1': traj_player1[:, 0],
                       'y1': traj_player1[:, 1],
                       'x2': traj_player2[:, 0],
                      'y2': traj_player2[:, 1],
                       'b': beliefs})
    df.to_csv(f'traj_viz_PPO_{seed}.csv', index=False)
    # df.to_csv(f'MMD_camsdrl.csv', index=False)
