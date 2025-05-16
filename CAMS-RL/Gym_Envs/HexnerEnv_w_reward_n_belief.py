import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time


class GymHexnerEnv(gym.Env):
    """
    A Gymnasium environment mimicking Hexner's game in OpenSpiel with continuous actions.

    - Continuous actions: Player 1 (attacker control u = [ux, uy]) and Player 2 (defender control d = [dx, dy])
    - Dynamics: Updates positions and velocities using a fixed time step (dt) and clips positions within [-1, 1].
    - Reward: Instantaneous costs are accumulated; at the terminal step a terminal cost based on the distance
      to a goal (which depends on player1's type) is added to yield a zero-sum payoff.
    - Observations: By default, player1 sees a dict with "state" (8-dim) and "player_type" (2-dim); player2 sees "state" (8-dim).
      If `include_action_history` is enabled, then both observations also include an "action_history" field of shape (max_game_length, 2).
      (The history is padded with zeros until all timesteps are filled.)

    The environment also supports rendering.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, dt=0.25, max_game_length=None,
                 ux_max=12, uy_max=12, dx_max=12, dy_max=12, p=0.5,
                 render_on=True, include_action_history=False):
        super().__init__()
        self.dt = dt
        if max_game_length is None:
            self.max_game_length = int(1 / dt)
        else:
            self.max_game_length = max_game_length
        self.ux_max = ux_max
        self.uy_max = uy_max
        self.dx_max = dx_max
        self.dy_max = dy_max
        self.render_on = render_on
        self.include_action_history = include_action_history

        # sample p if not set
        if p is not None:
            self.p = p
        else:
            self.p = np.random.uniform(0, 1)

        self.init_p = self.p


        # Define action space: continuous 2-D control for each player also add belief
        self.action_space = spaces.Dict({
            "player1": spaces.Box(
                low=np.array([-ux_max, -uy_max], dtype=np.float32),
                high=np.array([ux_max, uy_max], dtype=np.float32),
                shape=(2,),
                dtype=np.float32),
            "player2": spaces.Box(
                low=np.array([-dx_max, -dy_max], dtype=np.float32),
                high=np.array([dx_max, dy_max], dtype=np.float32),
                shape=(2,),
                dtype=np.float32),
            "belief": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(),
                dtype=np.float32)
        })

        # Define observation space.
        # Base fields: For player1, "state" (9-dim) (time and state) and "player_type" (2-dim); for player2, "state" (9-dim).
        player1_obs = {
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            "player_type": spaces.Box(low=0, high=1, shape=(2,), dtype=np.int32),
            "belief": spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
        }
        player2_obs = {
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            "belief": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
        }
        # If action history is included, add an "action_history" field of fixed shape (max_game_length, 2*2) both players
        if self.include_action_history:
            action_history_space = spaces.Box(low=-np.inf, high=np.inf,
                                              shape=(self.max_game_length, 2*2),
                                              dtype=np.float32)
            player1_obs["action_history"] = action_history_space
            player2_obs["action_history"] = action_history_space

        self.observation_space = spaces.Dict({
            "player1": spaces.Dict(player1_obs),
            "player2": spaces.Dict(player2_obs)
        })

        # For rendering.
        self.fig = None
        self.ax = None

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # sample p again if keep_p is not passed
        if options is None:
            self.p = self.init_p

        self.t_step = 0
        self.cumulative_cost = 0.0
        # State: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        self.state = np.zeros(8, dtype=np.float32)
        self.state[0] = -0.5  # Player1's x position.
        self.state[4] = 0.5  # Player2's x position.
        # self.state[1] = -0.5 # Player1's y position
        # self.state[5] = -0.5 # Player2's y position
        # Randomly set player1's type.
        # self.p1type = [0, 1] if np.random.rand() < self.p else [1, 0]  # flip this
        self.p1type = [1, 0] if np.random.rand() < self.p else [0, 1]
        # Initialize action history if enabled.
        if self.include_action_history:
            self.action_history = []  # List of dicts; each dict: {"player1": action, "player2": action}
        return self._get_obs(), {}

    def step(self, actions):
        # Ensure actions are numpy arrays.
        action1 = np.array(actions["player1"], dtype=np.float32)
        action2 = np.array(actions["player2"], dtype=np.float32)
        belief1 = np.array(actions["belief"], dtype=np.float32)

        next_state, ins_cost = self._go_forward(self.state, action1, action2)
        # self.cumulative_cost += ins_cost
        self.state = next_state
        self.p = belief1
        self.t_step += 1

        if self.include_action_history:
            self.action_history.append(actions)

        terminated = False
        reward = ins_cost  # not zero
        if self.t_step >= self.max_game_length:
            terminated = True
            terminal_cost = self._terminal_cost(self.state)
            total_cost = reward + terminal_cost
            reward = total_cost
            # reward["player2"] = total_cost
        return self._get_obs(), -reward, terminated, False, {}

    def _go_forward(self, state, action1, action2):
        dt = self.dt
        x1, y1, vx1, vy1 = state[0:4]
        x2, y2, vx2, vy2 = state[4:8]
        x1_next = x1 + vx1 * dt + 0.5 * action1[0] * dt ** 2
        y1_next = y1 + vy1 * dt + 0.5 * action1[1] * dt ** 2
        vx1_next = vx1 + action1[0] * dt
        vy1_next = vy1 + action1[1] * dt
        x2_next = x2 + vx2 * dt + 0.5 * action2[0] * dt ** 2
        y2_next = y2 + vy2 * dt + 0.5 * action2[1] * dt ** 2
        vx2_next = vx2 + action2[0] * dt
        vy2_next = vy2 + action2[1] * dt
        # Clip positions.
        x1_next = np.clip(x1_next, -1, 1)
        y1_next = np.clip(y1_next, -1, 1)
        x2_next = np.clip(x2_next, -1, 1)
        y2_next = np.clip(y2_next, -1, 1)
        next_state = np.array([x1_next, y1_next, vx1_next, vy1_next,
                               x2_next, y2_next, vx2_next, vy2_next],
                              dtype=np.float32)
        # Instantaneous cost computation (quadratic cost on controls).
        cost_u = action1[0] ** 2 * 0.05 + action1[1] ** 2 * 0.025
        cost_d = action2[0] ** 2 * 0.05 + action2[1] ** 2 * 0.1
        ins_cost = dt * (cost_u - cost_d)
        return next_state, ins_cost

    def _terminal_cost(self, state):
        # may be include belief?
        # goal1 = np.array([0, 1])
        # goal2 = np.array([0, -1])
        goal = np.array([0, 1], dtype=np.float32) if self.p1type == [1, 0] else np.array([0, -1], dtype=np.float32)
        x1 = state[0:2]
        x2 = state[4:6]

        # cost1 = np.linalg.norm(x1 - goal1) ** 2 - np.linalg.norm(x2 - goal1) ** 2
        # cost2 = np.linalg.norm(x1 - goal2) ** 2 - np.linalg.norm(x2 - goal2) ** 2
        #
        # return self.p * cost1 + (1 - self.p) * cost2
        return np.linalg.norm(x1 - goal) ** 2 - np.linalg.norm(x2 - goal) ** 2

    def _get_obs(self):
        # Base observation.
        obs_player1 = {
            "state": np.concatenate(([self.t_step * self.dt], self.state.copy())),
            "player_type": np.array(self.p1type, dtype=np.int32),
            "belief": self.p,
        }
        obs_player2 = {
            "state": np.concatenate(([self.t_step * self.dt], self.state.copy())),
            "belief": self.p,
        }
        if self.include_action_history:
            # Build padded history arrays (shape: (max_game_length, 2*2)) for each player (both players actions)
            history_len = len(self.action_history) if hasattr(self, "action_history") else 0
            if history_len > 0:
                hist1 = np.array([a["player1"] for a in self.action_history], dtype=np.float32)
                hist2 = np.array([a["player2"] for a in self.action_history], dtype=np.float32)
            else:
                hist1 = np.zeros((0, 2), dtype=np.float32)
                hist2 = np.zeros((0, 2), dtype=np.float32)
            hist = np.hstack((hist1, hist2))
            padded1 = np.zeros((self.max_game_length, 4), dtype=np.float32)
            padded2 = np.zeros((self.max_game_length, 4), dtype=np.float32)
            if history_len > 0:
                padded1[-history_len:] = hist
                padded2[-history_len:] = hist
            obs_player1["action_history"] = padded1
            obs_player2["action_history"] = padded2
        return {"player1": obs_player1, "player2": obs_player2}

    def render(self, mode="human"):
        if not self.render_on:
            return
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.set_title(f"Time step: {self.t_step}")
        x1, y1 = self.state[0], self.state[1]
        x2, y2 = self.state[4], self.state[5]
        self.ax.scatter(x1, y1, color="red", s=200, label="Player1 (Attacker)")
        self.ax.scatter(x2, y2, color="blue", s=200, label="Player2 (Defender)")
        goal = np.array([0, 1]) if self.p1type == [0, 1] else np.array([0, -1])
        self.ax.scatter(goal[0], goal[1], color="green", marker="*", s=250, label="Player1's Goal")
        self.ax.annotate("P1", (x1, y1), textcoords="offset points", xytext=(5, 5), color="red", fontsize=12)
        self.ax.annotate("P2", (x2, y2), textcoords="offset points", xytext=(5, 5), color="blue", fontsize=12)
        self.ax.legend(loc="upper right")
        plt.draw()
        plt.pause(0.1)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None
    def render(self, mode="human"):
        if not self.render_on:
            return
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.clear()
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.set_title(f"Time step: {self.t_step}")
        x1, y1 = self.state[0], self.state[1]
        x2, y2 = self.state[4], self.state[5]
        self.ax.scatter(x1, y1, color="red", s=200, label="Player1 (Attacker)")
        self.ax.scatter(x2, y2, color="blue", s=200, label="Player2 (Defender)")
        goal = np.array([0, 1]) if self.p1type == [0, 1] else np.array([0, -1])
        self.ax.scatter(goal[0], goal[1], color="green", marker="*", s=250, label="Player1's Goal")
        self.ax.annotate("P1", (x1, y1), textcoords="offset points", xytext=(5, 5), color="red", fontsize=12)
        self.ax.annotate("P2", (x2, y2), textcoords="offset points", xytext=(5, 5), color="blue", fontsize=12)
        self.ax.legend(loc="upper right")
        plt.draw()
        plt.pause(0.1)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None
