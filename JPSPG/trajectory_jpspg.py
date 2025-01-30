import jax
import jax.numpy as jnp
import numpy as np
from flax import linen
from flax.training import checkpoints
from jpspg_minimal_example_continuous_incomplete_2d_multi_stage import NormalFormCTGame, utility_function
import matplotlib.pyplot as plt
import scipy.io as scio
from utils import util_funcs
from tqdm import tqdm


def scaled_tanh(x, a, b):
    return a + (b - a) * (jnp.tanh(x) + 1)/2

class strategy_nn(linen.Module):
    num_hidden_layers: int
    output_dim: int
    low: int
    high: int

    @linen.compact
    def __call__(self, x):
        key = self.make_rng("noise")

        # 2) Learnable distribution parameters
        mu = self.param('mu', linen.initializers.zeros, (self.output_dim, ))
        log_sigma = self.param('log_sigma', linen.initializers.zeros, (self.output_dim, ))

        # 3) Reparameterize: z = mu + sigma * eps
        eps = jax.random.normal(key, shape=(self.output_dim, ))
        z = (mu + jnp.exp(log_sigma) * eps)

        z = jax.tree_util.tree_map(lambda x, z: jnp.hstack((x, z)), x, z)
        x = linen.relu(z)
        x = linen.Dense(64, kernel_init=linen.initializers.he_normal())(x)
        # x = linen.relu(x)
        x = linen.Dense(self.output_dim, kernel_init=linen.initializers.he_normal())(x)
        x = scaled_tanh(x, self.low, self.high)
        return x


def simulate_game(init_state, model, p1_action_nn, p2_action_nn, key, p1_type):
    p1_state = jnp.hstack((init_state[:4], p1_type))
    p2_state = init_state[4:]
    # Compute actions using the model
    p1_states = [p1_state[:4]]
    p2_states = [p2_state]
    p1_actions = []
    for _ in range(4):
        p1_action = model.apply(p1_action_nn, p1_state.reshape(-1, ), rngs={'noise': key})
        p2_action = model.apply(p2_action_nn, p2_state.reshape(-1, ), rngs={'noise': key})
        # Update p1 state
        x1 = p1_state[0:2] + p1_state[2:4] * 0.25 + 0.5 * p1_action * 0.25 ** 2
        vx1 = p1_state[2:4] + p1_action * 0.25
        p1_state = jnp.hstack((x1, vx1, p1_type))
        p1_states.append(p1_state[:-1])
        p1_actions.append(p1_action)

        # Update p2 state
        x2 = p2_state[0:2] + p2_state[2:4] * 0.25 + 0.5 * p2_action * 0.25 ** 2
        vx2 = p2_state[2:4] + p2_action * 0.25
        p2_state = jnp.hstack((x2, vx2))
        p2_states.append(p2_state)

    return jnp.vstack(p1_states), jnp.vstack(p2_states), jnp.vstack(p1_actions)


if __name__ == '__main__':
    game = NormalFormCTGame(utility_function, 1, [(-12, 12)])

    p1_params = checkpoints.restore_checkpoint('jpspg_models/p1/checkpoint_0', target=None)
    p2_params = checkpoints.restore_checkpoint('jpspg_models/p2/checkpoint_0', target=None)

    p1_type = 1

    num_iters = 100
    # keys = jax.random.split(jax.random.PRNGKey(p1_type))
    # keys = jax.random.PRNGKey(0)

    Us = []
    U_GTs = []

    R1 = jnp.array([[0.05, 0], [0, 0.025]])
    R2 = jnp.array([[0.05, 0], [0, 0.1]])

    ys = [0, -0.5, 0.5]
    keys = jax.random.split(jax.random.PRNGKey(p1_type), len(ys))


    # for run in tqdm(range(num_iters)):
    for run in tqdm(range(len(ys))):
        # game.sample_init_states(keys[run])
        init_state = jnp.array([-0.5, ys[run], 0, 0, 0.5, ys[run], 0, 0])
        game.states = init_state
        p1_states, p2_states, actions = simulate_game(game.states, game.model, p1_params, p2_params, keys[run], p1_type)

        Us.append(actions)
        # gt solution
        tau = 0.25
        A = jnp.eye(4) + jnp.array([[0, 0, tau, 0], [0, 0, 0, tau], [0, 0, 0, 0], [0, 0, 0, 0]])
        B = jnp.array([[0.5 * tau ** 2, 0], [0, 0.5 * tau ** 2], [tau, 0], [0, tau]])
        Qf = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        Q = jnp.zeros((4, 4))
        dt = tau
        N = 4
        R1 = jnp.array([[0.05, 0], [0, 0.025]]) * tau
        R2 = jnp.array([[0.05, 0], [0, 0.1]]) * tau

        K1 = util_funcs.discrete_lqr(A, B, Q, R1, Qf, N)
        K2 = util_funcs.discrete_lqr(A, B, Q, R2, Qf, N)

        p = 0.5
        # change index for different time steps

        target = p1_type

        trajs = []
        U_GT = []
        D_GT = []
        # trajs.append(states_)
        N = 4
        for i in range(4, 0, -1):
            states_ = p1_states[-i-1]
            if i <= 2:
                p = 0 if target == 0 else 1
            x1 = states_[:4]
            goal = jnp.array([0, 2 * p - 1, 0, 0]) * jnp.ones_like(p)
            u = -K1[-i] @ (x1 - goal).T
            U_GT.append(u)

        U_GTs.append(jnp.vstack(U_GT))

        # plots if needed
        fig, (state_ax, action_ax) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=plt.figaspect(0.4),
        )

        state_ax.plot(p1_states[:, 0], p1_states[:, 1], label='p1', color='red')
        state_ax.plot(p2_states[:, 0], p2_states[:, 1], label='p2', color='blue')
        action_ax.plot(actions[:, 0], label='a_x')
        action_ax.plot(actions[:, 1], label='a_y')
        state_ax.set_xlim([-1, 1])
        state_ax.set_ylim([-1, 1])

        state_ax.legend()
        action_ax.legend()
        plt.show()








