
# NO EXPLOITABILITY COMPUTATION FOR NOW. 
# TODO: Add Later
# from algorithms.eas_exploitability import (
#     build_traverser,
#     compute_exploitability_cached,
# )

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../IIG-RL-Benchmark/')

import pyspiel

import argparse
import yaml
import pickle
import torch
import numpy as np
import time
from torch import nn
import hexners_game_fixed_state
from torch.distributions.categorical import Categorical
import random
import algorithms
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as scio
# matplotlib.use('Agg')
# must import games

# create a dictionary of int-action pairs
ux_max = 12
uy_max = 12
dx_max = 12
dy_max = 12
n = 10
_uxs = np.linspace(-ux_max, ux_max, n)
_uys = np.linspace(-uy_max, uy_max, n)
_us = list(product(_uxs, _uys))
_umap = {k: v for (k, v) in enumerate(_us)}


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PGModel(torch.nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(PGModel, self).__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 512)), # 512
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )

    def forward(self, x):
        return self.model(x)


def load_models(agent_path_dict, game_str):
    agent_dict = {"player_0": [], "player_1": []}

    for agent_path in agent_path_dict:
        if agent_path["algo"] in ["psro", "escher", "escher", "nfsp", "rnad"]:
            with open(agent_path["path"], "rb") as f:
                # ipdb.set_trace()
                agents = pickle.load(f)
            models = [ag.get_model() for ag in agents]
            agent_dict["player_0"].append(
                {
                    "algo": agent_path["algo"],
                    "model": models[0],
                    "agent": agents[0],
                    "path": agent_path["path"],
                    "probs": None,
                }
            )
            agent_dict["player_1"].append(
                {
                    "algo": agent_path["algo"],
                    "model": models[1],
                    "agent": agents[1],
                    "path": agent_path["path"],
                    "probs": None,
                }
            )
        else:
            with open(agent_path["path"], "rb") as f:
                model_dict = torch.load(f, weights_only=True)
            if '0.weight' in model_dict:
                model_dict = {f"model.{k}": v for k, v in model_dict.items()}
            elif '0.0.weight' in model_dict: # PPG
                model_dict = {f"model.{k[2:]}" if k.count('.') == 2 else 'model.' + k.replace('1', '6'): v for k, v in model_dict.items()}
            else:
                raise ValueError(f"Unrecognized model dict keys: {model_dict.keys()}")

            game = pyspiel.load_game_as_turn_based(game_str)
            models = [PGModel(observation_shape=game.information_state_tensor_size(), num_actions=game.num_distinct_actions()) for _ in range(2)]
            for i in range(2):
                models[i].load_state_dict(model_dict)

            agent_dict["player_0"].append(
                {
                    "algo": agent_path["algo"],
                    "model": models[0],
                    "path": agent_path["path"],
                    "probs": None,
                }
            )
            agent_dict["player_1"].append(
                {
                    "algo": agent_path["algo"],
                    "model": models[1],
                    "path": agent_path["path"],
                    "probs": None,
                }
            )

            # assert False, "fix hard code for phantom_ttt"
    return agent_dict


def split_array(num_agents, k, part):
    chunk_size = num_agents // k
    chunks = []

    X = np.arange(num_agents)
    Y = np.arange(num_agents)

    X, Y = np.meshgrid(X, Y)

    for i in range(0, num_agents, chunk_size):
        for j in range(0, num_agents, chunk_size):
            X_ind = X[i : i + chunk_size, j : j + chunk_size]
            Y_ind = Y[i : i + chunk_size, j : j + chunk_size]
            chunks.append((X_ind.flatten(), Y_ind.flatten()))
    return chunks[part]


def main(args):
    start_time = time.time()
    with open(args.agents_yaml, "r") as f:
        agents_yaml = yaml.load(f, Loader=yaml.FullLoader)

    game = agents_yaml["game"]

    game_to_osgame = {
        'classical_phantom_ttt': 'phantom_ttt(obstype=reveal-nothing)',
        'abrupt_phantom_ttt': 'phantom_ttt(obstype=reveal-nothing,gameversion=abrupt)',
        'classical_dark_hex': 'dark_hex(gameversion=cdh,board_size=3,obstype=reveal-nothing)',
        'abrupt_dark_hex': 'dark_hex(gameversion=adh,board_size=3,obstype=reveal-nothing)',
        'kuhn_poker': 'kuhn_poker(players=2)',
        'leduc_poker': 'leduc_poker(players=2)',
        'hexners_game_entire_space': 'python_hexner_entire_space_test', 
        'hexners_game_fixed_state': 'python_hexner_fixed_state'
    }

    agent_dict = load_models(agents_yaml["agents"], game_to_osgame[game])

    # given a set of initial states, return the trajectory (states, p1_actions, p2_actions)
    model_p0 = agent_dict["player_0"][0]["model"]
    model_p1 = agent_dict["player_1"][0]["model"]


    # for now run for one initial state
    # init_states = [[-0.5, 0., 0., 0., 0.5, 0., 0., 0.]]

    
    states = []
    # rewards = []
    n = 1
    chosen_type = 1 # 1 is type 1; 0 is type 2
    Us = []
    for _ in range(n):
        U = []
        # new_game = pyspiel.convert_to_turn_based_game(hexners_game_entire_state_test.HexnerGame(init_state=init_state))
        new_game = pyspiel.load_game_as_turn_based(game_to_osgame[game])
        # let nature pick a type
        state = new_game.new_initial_state()
        # ipdb.set_trace()
        game_states = [state.information_state_tensor(0)][6:6+8]
        
        while not state.is_terminal():
            if state.current_player() == 0:
                game_states.append(state.information_state_tensor(0)[6:6+8])     

            if state.is_chance_node():
                action = chosen_type # manually pick P1's type
                state.apply_action(action)
                if action == 0:
                    target = 2 # goal 2
                else:
                    target = 1
            else:
                cur_player = state.current_player()
                info_state_vector = torch.tensor(state.information_state_tensor(cur_player), dtype=torch.float32)
                if cur_player == 0:
                    logits = model_p0(info_state_vector)
                else:
                    logits = model_p1(info_state_vector)
                
                probs = Categorical(logits=logits)
                # ipdb.set_trace()
                # probs = probs.
                # legal_actions = state.legal_actions(cur_player)
                # action = random.choices(legal_actions, probs)[0]
                action = probs.sample()

                if state.current_player() == 0:
                    U.append(_umap[int(action)])

                state.apply_action(action)
        game_states.append(state.information_state_tensor(0)[6:6 + 8])
        states.append(game_states)
        Us.append(U)

    # data = {'states': np.array(states), 'Us': np.array(Us)}
    # scio.savemat(f'{agents_yaml["agents"][0]["algo"]}_type_{"1" if chosen_type == 1 else "2"}_final.mat', data)

    game_states = np.vstack(game_states)
    x1 = game_states[:, 0]
    y1 = game_states[:, 1]
    x2 = game_states[:, 4]
    y2 = game_states[:, 5]

    # ipdb.set_trace()
    fig, ax = plt.subplots(1)

    ax.plot(x1, y1, label='P1')
    ax.plot(x2, y2, label='P2')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.legend()
    plt.show()
    # #
    # plt.savefig('test_run_type_10_2_10M.jpg')
    # save for plotting comparison
    import pandas as pd
    data = {'x1': x1, 'y1': y1, 'x2': x2, 'y2':y2}
    df = pd.DataFrame(data)
    df.to_csv('rnad_D_traj.csv')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-yaml", type=str, default='hexner.yaml', required=False)
    parser.add_argument("--save-dir", type=str, default='test/', required=False)
    parser.add_argument("--disc", type=int, default=1)
    parser.add_argument("--part", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)