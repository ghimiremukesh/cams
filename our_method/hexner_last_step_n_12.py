# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Hexner's one-step game for T-dt implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum

import numpy as np

import pyspiel
from itertools import product
import torch
from typing import Any, Iterable, List, Mapping, Optional, Set

# create a dictionary of int-action pairs
ux_max = 0.99
uy_max = 3.947
dx_max = 0.99
dy_max = 0.99
n = 12
_uxs = np.linspace(-ux_max, ux_max, n)
_uys = np.linspace(-uy_max, uy_max, n)
_dxs = np.linspace(-dx_max, dx_max, n)
_dys = np.linspace(-dy_max, dy_max, n)
_us = list(product(_uxs, _uys))
_ds = list(product(_dxs, _dys))
_umap = {k: v for (k, v) in enumerate(_us)}
_dmap = {k: v for (k, v) in enumerate(_ds)}

_NUM_PLAYERS = 2
_TYPE = frozenset([0, 1])
_dt = 1
_R1 = np.array([[0.05, 0], [0, 0.025]])
_R2 = np.array([[0.05, 0], [0, 0.1]])
# _BELIEF = frozenset([0.5, 0.5])
_GAME_TYPE = pyspiel.GameType(
    short_name="python_hexner_onestep",
    long_name="Python Hexner's Game",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=n * n,
    max_chance_outcomes=len(_TYPE),  # p1's types
    num_players=_NUM_PLAYERS,
    min_utility=-1e6,
    max_utility=1e6,
    utility_sum=0.0,
    max_game_length=1)  # e.g. total time-steps


class HexnerGame(pyspiel.Game):
    """A Python version of Hexner game."""
    initial_state: List
    value_fn: Optional
    def __init__(self,
                 initial_state: List = None):
        self.initial_state = initial_state[:-1]
        self.p = initial_state[-1]
        super().__init__(_GAME_TYPE, _GAME_INFO, dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return HexnerState(self, self.initial_state)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return HexnerObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)


def _go_forward(states, actions):
    """Applies point dynamics to advance the system, i.e. states of each player"""

    u = _umap[actions[0]]
    d = _dmap[actions[1]]

    ux, uy = u
    dx, dy = d

    x1 = states[0]
    y1 = states[1]
    vx1 = states[2]
    vy1 = states[3]

    x2 = states[4]
    y2 = states[5]
    vx2 = states[6]
    vy2 = states[7]

    x1_next = x1 + vx1 * _dt + (1/2) * ux * _dt ** 2
    y1_next = y1 + vy1 * _dt + (1/2) * uy * _dt ** 2
    vx1_next = vx1 +  ux * _dt
    vy1_next = vy1 +  uy * _dt

    x2_next = x2 + vx2 * _dt + (1/2) * dx * _dt ** 2
    y2_next = y2 + vy2 * _dt + (1/2) * dy * _dt ** 2
    vx2_next = vx2 + dx * _dt
    vy2_next = vy2 +  dy * _dt

    ins_reward = (_dt * (np.sum((np.array(u) ** 2) * np.diag(_R1)) - np.sum((np.array(d) ** 2) * np.diag(_R2))))

    x1_next = np.clip(x1_next, -1, 1)
    y1_next = np.clip(y1_next, -1, 1)
    x2_next = np.clip(x2_next, -1, 1)
    y2_next = np.clip(y2_next, -1, 1)

    return [x1_next, y1_next, vx1_next, vy1_next, x2_next, y2_next, vx2_next, vy2_next], \
           np.array([ins_reward, -ins_reward])


class HexnerState(pyspiel.State):
    """A python version of the Kuhn poker state."""

    def __init__(self, game, initial_state):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        # self.position_1 = np.random.uniform(-1, 1, 2)  # initialize position for p1
        # self.velocity_1x = np.random.uniform(-6, 6, 1)  # initialize velocity in x
        # self.velocity_1y = np.random.uniform(-12, 12, 1)  # initialize velocity in y
        #
        # self.position_2 = np.random.uniform(-1, 1, 2)  # initialize position for p2
        # self.velocity_2x = np.random.uniform(-6, 6, 1)  # initialize velocity in x
        # self.velocity_2y = np.random.uniform(-4, 4, 1)  # initialize velocity in y
        #
        # # now concat them
        # self.states = np.concatenate((self.position_1, self.velocity_1x, self.velocity_1y,
        #                               self.position_2, self.velocity_2x, self.velocity_2y)).reshape(1, -1)
        self.states = np.array(initial_state).reshape(1, -1)

        self.actions = []
        self._game_over = False
        self._next_player = 0
        self._is_chance = True  # no chance here
        self.t_step = 0  # initial time step is 0
        self._rewards = np.zeros(_NUM_PLAYERS)
        self._returns = np.zeros(_NUM_PLAYERS)
        self.p1type = [0, 0]  # first index represents type: left, and second type:right
        self.p = 0.5
        # self.time = 0

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every sequential-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self._is_chance:
            return pyspiel.PlayerId.CHANCE
        else:
            return pyspiel.PlayerId.SIMULTANEOUS

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0
        return list(_umap.keys()) if player == 0 else list(_dmap.keys())

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        assert self._is_chance
        outcomes = _TYPE
        p = [self.p, 1 - self.p]
        return list(zip(outcomes, p))

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        # This is not called at simultaneous-move states
        assert self._is_chance and not self._game_over
        self._is_chance = False
        self.p1type[action] = 1

    def _apply_actions(self, actions):
        """Applies the specified actions (per player) to the state."""
        assert not self._is_chance and not self._game_over
        _next_states, _ins_costs = _go_forward(self.states[-1], actions)
        self.states = np.vstack((self.states, _next_states))
        # self.states.append(_next_states)
        self._rewards = _ins_costs
        self.t_step += 1
        self._game_over = True if self.t_step >= self.get_game().max_game_length() else False
        if self._game_over:
            self._returns -= (self._terminal_cost(_next_states) + self._rewards)

    def _terminal_cost(self, states):
        """Computes terminal cost of the game based on the p1's type."""
        goal_1 = np.array([0, 1])
        goal_2 = np.array([0, -1])
        x1 = states[:2]
        x2 = states[4:6]

        dist_to_goal_1 = np.linalg.norm(x1 - goal_1) ** 2 - np.linalg.norm(x2 - goal_1) ** 2
        dist_to_goal_2 = np.linalg.norm(x1 - goal_2) ** 2 - np.linalg.norm(x2 - goal_2) ** 2

        if self.p1type == [0, 1]:
            terminal = dist_to_goal_1
        else:
            terminal = dist_to_goal_2

        return np.array([terminal, -terminal])

    def _action_to_string(self, player, action):
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return f"Goal:{'2' if action == 0 else '1'}"
        if player == 0:
            return f"{action}"  # just return the action idx
        else:
            return f"{action}"

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def rewards(self):
        """Reward at the previous step"""
        return self._rewards

    def returns(self):
        """Total reward for each player over the course of the game."""
        return self._returns  # round off for better graph viz.

    def __str__(self):
        """String for debug purposes. No particular semantics are required"""
        return (f"p1:{self.action_history_string(0)}"
                f"p2:{self.action_history_string(1)}")

    def action_history_string(self, player):
        return "".join(
            self._action_to_string(pa.player, pa.action)[0]
            for pa in self.full_history()
            if pa.player == player
        )


class HexnerObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""
    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        assert not bool(params)
        self.iig_obs_type = iig_obs_type
        pieces = []
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("player_type", 2, (2,)))

        # build the single flat tensor
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)
        # build the named & reshaed views of the bits of the flat tensor.
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index:index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        self.tensor.fill(0)
        if "player_type" in self.dict and player == 0:
            self.dict["player_type"][state.p1type[player]] = 1

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        if "player_type" in self.dict and player == 0:
            pieces.append(f"Goal:{'2' if state.p1type == [1, 0] else '1'}")
        pieces.append(f"us:{state.action_history_string(player)} "
                      f"op:{state.action_history_string(1 - player)}")
        return " ".join(p for p in pieces)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, HexnerGame)
