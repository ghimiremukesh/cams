import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import hydra
import json
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pyspiel
import random
import time
import torch
import traceback
import uuid

from open_spiel.python import policy
import ipdb


# from algorithms.eas_exploitability import compute_exploitability, build_traverser
from algorithms.runner import get_runner_cls
from utils import log_to_csv, get_metadata, log_memory_usage_periodically

from open_spiel.python.algorithms import exploitability, get_all_states
from torch.distributions.categorical import Categorical

from open_spiel_games import hexners_game_fixed_state

OPENSPIEL_GAMES = {
    "classical_phantom_ttt": "phantom_ttt(obstype=reveal-nothing)",
    "abrupt_phantom_ttt": "phantom_ttt(obstype=reveal-nothing,gameversion=abrupt)",
    "classical_dark_hex": "dark_hex(gameversion=cdh,board_size=3,obstype=reveal-nothing)",
    "abrupt_dark_hex": "dark_hex(gameversion=adh,board_size=3,obstype=reveal-nothing)",
    "kuhn_poker": "kuhn_poker(players=2)",
    "leduc_poker": "leduc_poker(players=2)",
    "hexners_game_fixed_state": "python_hexner_fixed_state", 
}


def set_seed(seed):
    # set the random seed for torch, numpy, and python
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def create_custom_policy(game, infostate_prob_map):
    """Create TabularPolicy with custom action probabilities.
    
    Args:
        game: OpenSpiel game object
        infostate_prob_map: Dict mapping {infostate_str: {action: prob}}
    
    Returns:
        Configured TabularPolicy
    """
    # Get all states and initialize tabular policy
    all_states = get_all_states.get_all_states(game)
    tabular_policy = policy.TabularPolicy(game, states=all_states)
    
    # Populate action probabilities
    for state_str, state_idx in tabular_policy.state_lookup.items():
        # ipdb.set_trace()
        infostate_key = state_str
        
        if infostate_key in infostate_prob_map:
            action_probs = infostate_prob_map[infostate_key]
            for action, prob in action_probs.items():
                # ipdb.set_trace()
                tabular_policy.action_probability_array[state_idx, action] = prob

    # ipdb.set_trace()         
    return tabular_policy

def compute_exploitability_wrapper(
    log_file,
    game_name,
    algorithm,
    model_p0,
    model_p1,
    step,
    action_selection=["sto", "sto"],
):
    # print("Computing exploitability... (this can take a few minutes)")
    t0_exploitability = time.time()
    # ev0, expl0, expl1 = compute_exploitability(
    #     model_p0,
    #     model_p1,
    #     traverser=traverser,
    #     batch_size=400_000,
    #     action_selection=action_selection,
    #     game_name=game_name,
    # )
    
    # manually get all infostates set in the game and set the policy
    game = pyspiel.load_game_as_turn_based(OPENSPIEL_GAMES[game_name])
    p1_infostates = []
    p1_infostates_str = []
    p2_infostates = []
    p2_info_states_str = []
    policy_dict = {}

    # if isinstance(model_p0, dict) and isinstance(model_p1, dict):
    #     model_p0 = model_p0['models']
    #     model_p1 = model_p1['models']

    for p1_type in [1, 0]:
        initial_state = game.new_initial_state()
        initial_state.apply_action(p1_type)
        # ipdb.set_trace()
        p1_infostates.append(initial_state.information_state_tensor(0))
        p1_infostates_str.append(initial_state.information_state_string(0))
        # take random action for p1
        initial_state.apply_action(1)  # doesn't matter
        if len(p2_infostates) == 0: # just need to do it once as P2 doens't see the type
            p2_infostates.append(initial_state.information_state_tensor(1))
            p2_info_states_str.append(initial_state.information_state_string(1))

    # once we have the infostate, get the action probabilities for each of the infostates
    # p1_probs = Categorical(logits=model_p0(torch.tensor(p1_infostates, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu'))).probs
    # p2_probs = Categorical(logits=model_p1(torch.tensor(p2_infostates, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu'))).probs

    # ipdb.set_trace()
    # switch to cpu for exploibability computation
    # model_p0 = model_p0.to('cpu')
    # model_p1 = model_p1.to('cpu')
    # ipdb.set_trace()
    if algorithm['algorithm_name'] == 'mmd':
        # p1_probs = Categorical(logits=model_p0(torch.tensor(p1_infostates, dtype=torch.float32, device=model_p0[0].weight.get_device()))).probs
        # p2_probs = Categorical(logits=model_p1(torch.tensor(p2_infostates, dtype=torch.float32, device=model_p1[0].weight.get_device()))).probs
        p1_probs = Categorical(logits=model_p0(torch.tensor(p1_infostates, dtype=torch.float32))).probs
        p2_probs = Categorical(logits=model_p1(torch.tensor(p2_infostates, dtype=torch.float32))).probs
    else:
        p1_probs = Categorical(logits=model_p0(torch.tensor(p1_infostates, dtype=torch.float32))).probs
        p2_probs = Categorical(logits=model_p1(torch.tensor(p2_infostates, dtype=torch.float32))).probs

    # get the action probabilities for each of the infostates
    for i in range(len(p1_infostates)):
        policy_dict[p1_infostates_str[i]] = {j: p1_probs[i, j].item() for j in range(p1_probs.shape[1])}
    
    policy_dict[p2_info_states_str[0]] = {i: p2_probs.reshape(-1, )[i].item() for i in range(p1_probs.shape[1])}

    ## now create a tabular policy from the policy dict
    curr_policy = create_custom_policy(game, policy_dict)
    # ipdb.set_trace()

    # don't compute for large action space game
    # exp = exploitability.exploitability(game, curr_policy)

    # get the distance to gt
    a1_gt = np.array([[0.833, 1.818]])
    a2_gt = np.array([[0.833, -1.818]])

    # get the actions
    from open_spiel_games.hexners_game_fixed_state import _umap
    us = np.array(list(_umap.values()))
    pi_1 = np.array(list(policy_dict[p1_infostates_str[0]].values()))
    pi_2 = np.array(list(policy_dict[p1_infostates_str[1]].values()))

    a1_mean = (pi_1.reshape(-1, 1) * us).sum(0)
    a2_mean = (pi_2.reshape(-1, 1) * us).sum(0)

    dist_to_gt_mean = (np.linalg.norm(a1_mean - a1_gt) + np.linalg.norm(a2_mean - a2_gt))/2


    log_data = {
        "global_step": step,
        # "avg_score_response": (expl0 + expl1) / 2,
        # "avg_score_p0": ev0 + expl1,
        # "avg_score_p1": -ev0 + expl0,
        # "ev0": ev0,
        # "expl0": expl0,
        # "expl1": expl1,
        # "expl": exp, # only for small action space game
        "dist_to_gt": dist_to_gt_mean.item(),
        "timestamp": time.time(),
        "computation_duration": time.time() - t0_exploitability,
    }
    # print(f'avg_score_response={(expl0 + expl1) / 2}')
    # print(f'Exploitability: {exp}')
    
    log_to_csv(log_data, log_file)


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def main(cfg: DictConfig):
    # checks
    assert cfg.game in list(OPENSPIEL_GAMES.keys())
    assert cfg.compute_exploitability in [True, False]

    # seed
    set_seed(cfg.seed)

    # setup logging
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    random_str = uuid.uuid4().hex[:6]
    experiment_dir = os.path.join(
        cfg.save_dir, cfg.group_name, cfg.algorithm.algorithm_name, cfg.game, f'{time_str}_{random_str}'
    )
    os.makedirs(experiment_dir, exist_ok=False)
    cfg.experiment_dir = experiment_dir
    print(f"Logging at {cfg.experiment_dir}")

    # save config
    config_path = os.path.join(cfg.experiment_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)

    # save metadata
    metadata = get_metadata()
    metadata['status'] = 'Running'
    metadata['start_date'] = datetime.now().strftime("%B %d, %Y at %-I:%M:%S %p")
    metadata['start_timestamp'] = time.time()
    metadata_path = os.path.join(cfg.experiment_dir, "metadata.json")
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    # setup exploitability computation
    if cfg.compute_exploitability:
        t0_traverser = time.time()
        # print("Building traverser... (this can take a few minutes)")
        # traverser = build_traverser(cfg.game)
        # print("Done")
        # metadata['eas_traverser_build_time'] = time.time() - t0_traverser
        with open(metadata_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        exploitability_log_file = os.path.join(cfg.experiment_dir, "exploitability.csv")

        def compute_exploitability_callback(*args, **kwargs):
            compute_exploitability_wrapper(
                exploitability_log_file, cfg.game, cfg.algorithm, *args, **kwargs
            )
    else:
        compute_exploitability_callback = None

    # load game
    print("Loading game")
    openspiel_game = OPENSPIEL_GAMES[cfg.game]
    # try catch to load the game as turn-based if simultaneous
    if cfg.turn_based:
        game = pyspiel.load_game(openspiel_game)
    else:
        game = pyspiel.load_game_as_turn_based(openspiel_game)

    # load algorithm
    print("Loading runner")
    algorithm = cfg.algorithm.algorithm_name
    runner = get_runner_cls(algorithm)(cfg, game, compute_exploitability_callback)

    # start memory logging thread
    memory_usage_file_path = os.path.join(cfg.experiment_dir, "memory_usage.log")
    thread, stop_event = log_memory_usage_periodically(log_file_path=memory_usage_file_path, log_interval_s=10, runner=runner)

    # train
    print(
        f'Training {cfg.algorithm.algorithm_name.upper()} on {cfg.game} for {cfg.max_steps} steps '
        f'{"with" if cfg.compute_exploitability else "without"} exploitability computation'
    )

    if cfg.run_in_safe_mode:
        print("Running in safe mode")
        try:
            runner.run()
        # update metadata
        except KeyboardInterrupt:
            metadata['status'] = 'Stopped'
            print('Training stopped')
        except Exception as e:
            metadata['status'] = 'Crashed'
            metadata['exception'] = repr(e)
            print('Training crashed')
            tb = traceback.format_exc()
            print(tb)
            err_path = os.path.join(cfg.experiment_dir, "error.txt")
            with open(err_path, 'w') as f:
                f.write(tb)
            metadata['error'] = tb
        else:
            metadata['status'] = 'Finished'
            print('Training finished!')
    else:
        runner.run()
        metadata['status'] = 'Finished'
        print('Training finished!')
    metadata['end_date'] = datetime.now().strftime("%B %d, %Y at %-I:%M:%S %p")
    metadata['end_timestamp'] = time.time()
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    # stop memory loggging thread peacefully
    stop_event.set()
    thread.join()


if __name__ == "__main__":
    main()
