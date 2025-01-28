import argparse
import copy
from functools import partial
from timeit import default_timer
from turtledemo.penrose import start

import jax
import optax
from jax import lax, nn, numpy as jnp, random
from matplotlib import pyplot as plt
from optax import tree_utils as otu
from flax import linen
import numpy as np


# jax.config.update("jax_disable_jit", True)



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


def utility_function(state, action, r, p1_type):
    goal = jnp.where(p1_type == 0, np.array([0, -1]), jnp.array([0, 1]))
    x1 = state[0:2] + 0.5 * action

    return jnp.sum((x1 - goal) ** 2) + jnp.sum(jnp.diag(r) * action ** 2)

def get_exploitability(utility, action_0, action_1, states):
    # hard code r1 and r2
    r1 = jnp.array([[0.05, 0.],
                    [0., 0.025]])

    util_0 = -utility(states[0:2], action_0, r1, 0)
    util_1 = -utility(states[0:2], action_1, r1, 1)
    gt_action_0 = jnp.array([0.833, -1.818])
    gt_action_1 = jnp.array([0.833, 1.818])
    exploitability_0 = (-utility(states[0:2], gt_action_0, r1, 0) - util_0)
    exploitability_1 = (-utility(states[0:2], gt_action_1, r1, 1) - util_1)

    return (exploitability_1 + exploitability_0)/2


def pseudo_gradient(f, x, key, scale):
    # https://arxiv.org/abs/1703.03864
    key, subkey = random.split(key)
    z = otu.tree_random_like(subkey, x)
    zs = jax.tree.map(lambda z: jnp.stack([z, -z]), z)
    xs = otu.tree_add_scalar_mul(x, scale, zs)
    ys = jax.vmap(f, [0, None], axis_size=2)(xs, key)
    scale_factor = 1 / (len(ys) * scale)

    def combine(zs):
        return jnp.tensordot(ys, zs, (0, 0)) * scale_factor

    return jax.tree.map(combine, zs)


def tree_get_index(tree, index):
    return jax.tree.map(lambda x: x[index], tree)


def joint_perturbation_simultaneous_pseudo_gradient(u, x, key, scale):
    # https://arxiv.org/abs/2408.09306
    jacobian = pseudo_gradient(u, x, key, scale)
    jacobian_diag = [tree_get_index(si, i) for i, si in enumerate(jacobian)]
    return jacobian_diag


def get_nfg_ct_utilities(utility, model, params, states, p1_type, key):
    # just for 2 players
    p1_action_nn = params[0]
    p2_action_nn = params[1]
    p1_state = jax.tree_util.tree_map(lambda a, b: jnp.hstack((a, b)), jnp.array(states[0:2]), p1_type)
    p2_state = jnp.array(states[2:4]).reshape(-1, )


    p1_action = model.apply(p1_action_nn, p1_state.reshape(-1, ), rngs={'noise': key})
    p2_action = model.apply(p2_action_nn, p2_state.reshape(-1, ), rngs={'noise': key})

    # hard code r1 and r2
    r1 = jnp.array([[0.05, 0.],
                    [0., 0.025]])

    r2 = jnp.array([[0.05, 0],
                    [0., 0.1]])

    p1_util = utility(p1_state, p1_action, r1, p1_type) # should return both instant. cost and final cost
    p2_util = utility(p2_state, p2_action, r2, p1_type)

    return -jnp.array([p1_util - p2_util, p2_util - p1_util])

class NormalFormCTGame:
    def __init__(self, utility_f, adim, abounds, states, seed=0):
        self.utility = utility_f
        self.action_dim = adim
        self.action_bounds = abounds
        self.states = states
        self.seed = seed
        key = random.PRNGKey(self.seed)
        key1, self.noise_key = random.split(key, 2)

        # self.p1_type = jnp.array(np.random.choice(jnp.array([0, 1]), p=jnp.array([0.5, 0.5])))
        self.p1_type = None
        self.model = strategy_nn(num_hidden_layers=1, output_dim=2, low=self.action_bounds[0][0],
                                   high=self.action_bounds[0][1])

    def set_p1_type(self, p1_type):
        self.p1_type = p1_type

    def init_params(self, key):
        keys = random.split(key, 2) # there's always two players
        p1_dummy_state = jnp.zeros((3, ))
        p2_dummy_state = jnp.zeros((2, ))
        states = [p1_dummy_state, p2_dummy_state]

        return [self.model.init(keys[i], states[i]) for i in range(2)]


    def get_utilities(self, params, key):

        def f(x, _):
            return jax.tree.map(lambda x: x + 1e-8, x), None

        params, _ = lax.scan(f, params, length=1)

        return get_nfg_ct_utilities(self.utility, self.model, params, self.states, self.p1_type, self.noise_key)

    def get_metrics(self, params):
        p1_state = jnp.hstack((jnp.array(self.states[0:2]), self.p1_type))
        p2_state = jnp.array(self.states[2:4])
        action_1 = self.model.apply(params[0], p1_state.reshape(-1, ), rngs={'noise': self.noise_key})
        action_2 = self.model.apply(params[1], p2_state.reshape(-1, ), rngs={'noise': self.noise_key})

        # for exploitability
        s_0 = jnp.hstack((jnp.array(self.states[0:2]), 0))
        s_1 = jnp.hstack((jnp.array(self.states[0:2]), 1))
        a_0 = self.model.apply(params[0], s_0.reshape(-1, ), rngs={'noise': self.noise_key})
        a_1 = self.model.apply(params[0], s_1.reshape(-1, ), rngs={'noise': self.noise_key})
        exploitability = get_exploitability(self.utility, a_0, a_1, jnp.array(self.states))
        return {'p1_action': action_1, 'p2_action': action_2, 'type': self.p1_type, 'exploitability': exploitability}


def train(game, key, optimizer, iters, scale, solver):

    def update(state, key):

        params, opt_state = state

        key, subkey = random.split(key, 2)
        p1_type = random.choice(subkey, jnp.array([0, 1]), p=jnp.array([0.5, 0.5]))
        game.set_p1_type(p1_type)

        metrics = game.get_metrics(params)

        match solver:
            case "JPSPG":
                grads = joint_perturbation_simultaneous_pseudo_gradient(
                    game.get_utilities, params, key, scale
                )
            case _:
                raise NotImplementedError

        grads = jax.tree.map(jnp.negative, grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # params = jax.tree_util.tree_map(lambda params: jnp.squeeze(params, axis=0), params) # to maintain the shape
        return (params, opt_state), metrics

    key, subkey = random.split(key)
    params = game.init_params(subkey)
    opt_state = optimizer.init(params)
    keys = random.split(key, iters)
    (params, _), history = lax.scan(update, (params, opt_state), keys)

    return params, history


def get_game(args):
    match args.game:
        case "ct":
            util = utility_function
            return NormalFormCTGame(util, 1, [(-12, 12)], [-0.5, 0, 0.5, 0])
        case _:
            raise NotImplementedError


def get_optimizer(args):
    match args.opt:
        case "sgd":
            return optax.sgd(args.lr)
        case "adabelief":
            return optax.adabelief(args.lr)
        case "adam":
            return optax.adam(args.lr)
        case _:
            raise NotImplementedError


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--game", default="ct")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--actions", type=int, default=2)
    p.add_argument("--delay", type=int, default=10**3)

    p.add_argument("--opt", default="sgd")
    p.add_argument("--lr", type=float, default=2e-6)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=20*10**5)
    p.add_argument("--scale", type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()

    game = get_game(args)

    optimizer = get_optimizer(args)

    fig, (ax_iter, ax_runtime) = plt.subplots(
        ncols=2,
        sharey=False,
        constrained_layout=True,
        figsize=plt.figaspect(0.4),
    )

    key = random.key(args.seed)

    for solver in ["JPSPG"]:
        print(f"{solver=}")

        start_time = default_timer()
        params, hist = train(
            game=game,
            key=key,
            optimizer=optimizer,
            iters=args.iters,
            scale=args.scale,
            solver=solver,
        )
        hist = jax.block_until_ready(hist)
        end_time = default_timer()

        # game = get_game(args)
        # print("P1 Type: ", hist['type'][-1])
        # print(hist['type'])
        print("total time: ", end_time - start_time)
        # print("P1 Action: ", hist['p1_action'][-1])
        # print("P2 Action: ", hist['p2_action'][-1])
        p1_actions = hist['p1_action']
        p2_actions = hist['p2_action']
        p1_types = hist['type']

        p1_0_actions = p1_actions[jnp.where(p1_types == 0)]
        p1_1_actions = p1_actions[jnp.where(p1_types == 1)]

        # ax_iter.plot(p1_0_actions[:, 0], p1_0_actions[:, 1], label=f"P1 type:0")
        # ax_iter.plot(p1_1_actions[:, 0], p1_1_actions[:, 1], label='P1 type:1')
        ax_iter.scatter(p1_0_actions[:, 0], p1_0_actions[:, 1], c=range(len(p1_0_actions)), cmap='Blues', label="P1 type:0")
        ax_iter.scatter(p1_1_actions[:, 0], p1_1_actions[:, 1], c=range(len(p1_1_actions)), cmap='Oranges', label="P1 type:1")
        ax_iter.scatter(p2_actions[:, 0], p2_actions[:, 1], c=range(len(p2_actions)), cmap='Greens', label="P2")
        # ax_iter.plot(p2_actions[:, 0], p2_actions[:, 1], label='P2')
        # ax_iter.set_ylim([-0.8, 1.5])
        x = jnp.linspace(0, end_time - start_time, len(p2_actions))
        # ax_runtime.plot(hist['type'])

        ax_iter.scatter(0.833, -1.818, marker='x', label='P1 type:0 GT', color='Teal', s=200)
        ax_iter.scatter(0.833, 1.818, marker='x', label='P1 type:1 GT', color='Orange', s=200)
        ax_iter.scatter(-0.833, 0, marker='x', label='P2 GT', color='Green', s=200)
        # ax_runtime.plot(p1_0_actions)
        # ax_runtime.plot(p1_1_actions)
        # ax_runtime.plot(p2_actions)
        ax_runtime.plot(hist['exploitability'])
        ax_runtime.set_title('Exploitability')
        ax_iter.set_title('Actions')


        new_game = get_game(args)
        new_game.set_p1_type(0)
        print(new_game.get_metrics(params))
        new_game.set_p1_type(1)
        print(new_game.get_metrics(params))



    ax_iter.legend()
    # ax_runtime.legend(title="solver")

    ax_iter.set(xlabel="a_x", ylabel="a_y")
    ax_runtime.set(xlabel="iterations", ylabel="exploitability")
    plt.show()


if __name__ == "__main__":
    main()
