import argparse
import copy
import os.path
from functools import partial
from timeit import default_timer
import jax
import optax
from jax import lax, nn, numpy as jnp, random
from optax import tree_utils as otu
from flax import linen
import numpy as np
from flax.training import checkpoints
import flax
from jax.experimental import host_callback as hcb

flax.config.update('flax_use_orbax_checkpointing', False)


# jax.config.update("jax_disable_jit", True)

def progress_bar(arg, transforms):
    idx, n_iter, print_rate = arg
    if idx % print_rate == 0:
        print(f"Completed iteration {idx}/{n_iter}")


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
    x1 = state[0:2] + state[2:4] * 0.25 + 0.5 * action * 0.25 ** 2

    return jnp.sum((x1 - goal) ** 2) + 0.25 * jnp.sum(jnp.diag(r) * action ** 2)


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
    p1_state = jax.tree_util.tree_map(lambda a, b: jnp.hstack((a, b)), jnp.array(states[0:4]), p1_type)
    p2_state = jnp.array(states[4:8]).reshape(-1, )

    # 4-stage game
    t = 0
    dt = 0.25
    # hard code r1 and r2
    r1 = jnp.array([[0.05, 0.],
                    [0., 0.025]])

    r2 = jnp.array([[0.05, 0],
                    [0., 0.1]])

    def step(carry, _):
        p1_state, p2_state, key = carry

        # Compute actions using the model
        p1_action = model.apply(p1_action_nn, p1_state.reshape(-1, ), rngs={'noise': key})
        p2_action = model.apply(p2_action_nn, p2_state.reshape(-1, ), rngs={'noise': key})

        # Update p1 state
        x1 = p1_state[0:2] + p1_state[2:4] * 0.25 + 0.5 * p1_action * 0.25 ** 2
        vx1 = p1_state[2:4] + p1_action * 0.25
        new_p1_state = jnp.hstack((x1, vx1, p1_type))

        # Update p2 state
        x2 = p2_state[0:2] + p2_state[2:4] * 0.25 + 0.5 * p2_action * 0.25 ** 2
        vx2 = p2_state[2:4] + p2_action * 0.25
        new_p2_state = jnp.hstack((x2, vx2))

        # New carry for next iteration
        new_carry = (new_p1_state, new_p2_state, key)
        return new_carry, None

    # Initialize carry with starting states and key
    carry_init = (p1_state, p2_state, key)

    # Run lax.scan for 3 iterations
    final_carry, _ = lax.scan(step, carry_init, None, length=3)

    # Extract final states after 3 iterations
    p1_state, p2_state, _ = final_carry

    p1_action = model.apply(p1_action_nn, p1_state.reshape(-1, ), rngs={'noise': key})
    p2_action = model.apply(p2_action_nn, p2_state.reshape(-1, ), rngs={'noise': key})

    p1_util = utility(p1_state, p1_action, r1, p1_type) # should return both instant. cost and final cost
    p2_util = utility(p2_state, p2_action, r2, p1_type)

    return -jnp.array([p1_util - p2_util, p2_util - p1_util])

class NormalFormCTGame:
    def __init__(self, utility_f, adim, abounds, seed=0):
        self.utility = utility_f
        self.action_dim = adim
        self.action_bounds = abounds
        self.seed = seed
        key = random.PRNGKey(self.seed)
        self.key1, self.noise_key = random.split(key, 2)
        self.states = None
        # self.p1_type = jnp.array(np.random.choice(jnp.array([0, 1]), p=jnp.array([0.5, 0.5])))
        self.p1_type = None
        self.model = strategy_nn(num_hidden_layers=1, output_dim=2, low=self.action_bounds[0][0],
                                   high=self.action_bounds[0][1])

    def sample_init_states(self, key):
        pos = random.uniform(key, shape=(4,), minval=-1, maxval=1)
        self.states = jnp.array([pos[0], pos[1], 0, 0, pos[2], pos[3], 0, 0])

    def set_p1_type(self, p1_type):
        self.p1_type = p1_type

    def init_params(self, key):
        keys = random.split(key, 2) # there's always two players
        p1_dummy_state = jnp.zeros((5, ))
        p2_dummy_state = jnp.zeros((4, ))
        states = [p1_dummy_state, p2_dummy_state]

        return [self.model.init(keys[i], states[i]) for i in range(2)]


    def get_utilities(self, params, key):

        def f(x, _):
            return jax.tree.map(lambda x: x + 1e-8, x), None

        params, _ = lax.scan(f, params, length=1)

        return get_nfg_ct_utilities(self.utility, self.model, params, self.states, self.p1_type, self.noise_key)

    def get_metrics(self, params):
        p1_state = jnp.hstack((jnp.array(self.states[0:4]), self.p1_type))
        p2_state = jnp.array(self.states[4:8])
        p1_action_nn = params[0]
        p2_action_nn = params[1]

        dt = 0.25
        # hard code r1 and r2
        p1_actions = []
        p2_actions = []

        # for i in range(4):
        #     p1_action = self.model.apply(p1_action_nn, p1_state.reshape(-1, ), rngs={'noise': self.noise_key})
        #     p2_action = self.model.apply(p2_action_nn, p2_state.reshape(-1, ), rngs={'noise': self.noise_key})
        #     x1 = p1_state[0:2] + p1_state[2:4] * 0.25 + 0.5 * p1_action * 0.25 ** 2
        #     vx1 = p1_state[2:4] + p1_action * 0.25
        #     new_p1_state = jnp.hstack((x1, vx1, self.p1_type))
        #
        #     x2 = p2_state[0:2] + p2_state[2:4] * 0.25 + 0.5 * p2_action * 0.25 ** 2
        #     vx2 = p2_state[2:4] + p2_action * 0.25
        #     new_p2_state = jnp.hstack((x2, vx2))
        #     p1_actions.append(p1_action)
        #     p2_actions.append(p2_action)
        #
        #     p1_state, p2_state = new_p1_state, new_p2_state

        def step(carry, _):
            # Unpack the carry
            p1_state, p2_state, noise_key = carry

            # Compute actions using the model
            p1_action = self.model.apply(p1_action_nn, p1_state.reshape(-1, ), rngs={'noise': noise_key})
            p2_action = self.model.apply(p2_action_nn, p2_state.reshape(-1, ), rngs={'noise': noise_key})

            # Update p1 state
            x1 = p1_state[0:2] + p1_state[2:4] * 0.25 + 0.5 * p1_action * 0.25 ** 2
            vx1 = p1_state[2:4] + p1_action * 0.25
            new_p1_state = jnp.hstack((x1, vx1, self.p1_type))

            # Update p2 state
            x2 = p2_state[0:2] + p2_state[2:4] * 0.25 + 0.5 * p2_action * 0.25 ** 2
            vx2 = p2_state[2:4] + p2_action * 0.25
            new_p2_state = jnp.hstack((x2, vx2))

            # Prepare the new carry for next iteration
            new_carry = (new_p1_state, new_p2_state, noise_key)

            # Return the new carry and actions taken in this step
            return new_carry, (p1_action, p2_action)

        # Initialize the carry with starting states and noise key
        carry_init = (p1_state, p2_state, self.noise_key)

        # Execute lax.scan for 4 iterations and collect actions
        final_carry, actions = lax.scan(step, carry_init, None, length=4)

        # Unpack the results
        p1_state, p2_state, _ = final_carry
        p1_actions, p2_actions = actions

        return {'p1_states': p1_state, 'p1_actions': p1_actions, 'p2_states': p2_state, 'p2_actions': p2_actions, 'type': self.p1_type}


def train(game, key, optimizer, iters, scale, solver):
    n_iter = iters
    print_rate = 1000  # Adjust this value as needed for frequency of printing

    def update(state, key_and_idx):
        key, idx = key_and_idx
        params, opt_state = state

        # Progress reporting using id_tap
        hcb.id_tap(progress_bar, (idx, n_iter, print_rate))

        key, subkey1, subkey2 = random.split(key, 3)
        p1_type = random.choice(subkey1, jnp.array([0, 1]), p=jnp.array([0.5, 0.5]))
        game.set_p1_type(p1_type)
        game.sample_init_states(subkey2)

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
    indices = jnp.arange(iters)
    # Combine keys and indices to pass both to the update function
    keys_and_indices = (keys, indices)
    (params, _), history = lax.scan(update, (params, opt_state), keys_and_indices)

    return params, history


def get_game(args):
    match args.game:
        case "ct":
            util = utility_function
            return NormalFormCTGame(util, 1, [(-12, 12)])
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
    p.add_argument("--iters", type=int, default=10**8)
    p.add_argument("--scale", type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()

    game = get_game(args)

    optimizer = get_optimizer(args)

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

        ckpt_dir_1 = os.path.abspath('jpspg_models/p1/')
        ckpt_dir_2 = os.path.abspath('jpspg_models/p2/')
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir_1, target=params[0], step=0, overwrite=True)
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir_2, target=params[1], step=0, overwrite=True)
        # game = get_game(args)
        # print("P1 Type: ", hist['type'][-1])
        # print(hist['type'])
        print("total time: ", end_time - start_time)



if __name__ == "__main__":
    main()
