"""
Util functions for optimization problems
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from .util_funcs import x_next, compute_bounds, normalize_to_max_1d, running_cost
from .game_params import Hexner
from . import nn_modules
from functools import partial
from tqdm import tqdm
from flax.training import checkpoints
import flax

dt = Hexner.dt
EPSILON = 1e-6

try:
    pinn_model = nn_modules.PICNN(nn_modules.ModelConfig)
    bc_model = nn_modules.PICNN(nn_modules.NoTimeConfig)
    load_dir = os.path.abspath(f'../PINN/logs/_final/checkpoint_250000')
    pinn_state_dict = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=None)
    pinn_state_dict = flax.core.FrozenDict(pinn_state_dict)
except:
    print('some models not loaded. Make sure this is desired behavior')

# bounds for dsgda steps
bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 4),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-12., 12.]] * 4),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]

@partial(jit, static_argnums=(3, 4))
def gradient_step(params, x, p, loss_fn, t):
    c = 1e-1
    alpha = 5e-2
    val_fn = partial(loss_fn, t=t)
    grad = jax.vmap(jax.grad(val_fn, argnums=0))(params, x, p)
    p1_params = params[:, :6]
    p2_params = params[:, 6:10]
    new_p1_params = (p1_params - c * grad[:, :6]).clip(min_bounds[:6], max_bounds[:6])
    new_params = jnp.concatenate((new_p1_params, p2_params), axis=1)
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(new_params, x, p)
    new_p2_params = (p2_params + alpha * grad[:, 6:]).clip(min_bounds[6:], max_bounds[6:])
    new_params = jnp.concatenate((new_p1_params, new_p2_params), axis=1)

    return new_params

@partial(jit, static_argnums=(3, ))
def value_inter(params, x, p, t=0.5):
    u1 = params[:2]
    u2 = params[2:4]
    a1 = params[4]
    a2 = params[5]
    v1 = params[6:8]
    v2 = params[8:10]


    p_u1 = a1 * p + a2 * (1 - p)
    p_u2 = 1 - p_u1

    # posteriors
    pos_1 = a1 * p / p_u1

    pos_2 = (1 - a1) * p / p_u2

    x_next_1 = x_next(x, u1, v1)
    x_next_2 = x_next(x, u2, v2)

    input_1 = jnp.concat((x_next_1, pos_1))
    input_2 = jnp.concat((x_next_2, pos_2))

    # v_bound_next = compute_bounds(1 - t + dt, 12)
    v_bound_next = 12

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_1d(input_, v_bound_next, v_bound_next, v_bound_next, v_bound_next)
        t_ = jnp.array([t - dt])
        input_ = jnp.concatenate((t_, rescaled_input))
        return pinn_model.apply(pinn_state_dict, input_)

    val_1 = apply_to_model(input_1)
    val_2 = apply_to_model(input_2)

    ins_cost_1 = dt * running_cost(u1, v1).reshape(-1, )
    ins_cost_2 = dt * running_cost(u2, v2).reshape(-1, )

    final_cost_1 = val_1 + ins_cost_1
    final_cost_2 = val_2 + ins_cost_2

    objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

    return objective.reshape(())


class dsgda_solver:
    def __init__(self, num_points, t=0.5):
        self.t = t
        pos_bound = 1
        vel_bound = compute_bounds(1 - t, 12)
        key = jax.random.PRNGKey(0)
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        pos = jax.random.uniform(key1, (num_points, 4), minval=-pos_bound, maxval=pos_bound)
        vel_1 = jax.random.uniform(key2, (num_points, 2), minval=-vel_bound, maxval=vel_bound)
        vel_2 = jax.random.uniform(key3, (num_points, 2), minval=-vel_bound, maxval=vel_bound)
        self.p = jax.random.uniform(key4, (num_points, 1), minval=EPSILON, maxval=1 - EPSILON)

        self.states_ = jnp.concatenate((pos[:, :2], vel_1, pos[:, 2:4], vel_2), axis=1)

        params_u = jax.random.uniform(key5, (1, 4), minval=-1, maxval=1)
        params_up = jnp.array([[EPSILON, 1 - EPSILON]])
        params_d = jax.random.uniform(key6, (1, 4), minval=-1, maxval=1)

        params = jnp.concatenate((params_u, params_up, params_d), axis=1)
        self.params = np.repeat(params, repeats=num_points, axis=0)


    def solve_minimax(self):
        curr_params = self.params
        obj_fn = value_inter
        iters = 120000
        iter = 0
        with tqdm(total=iters) as pbar:
            while iter <= iters:
                new_params = gradient_step(curr_params, self.states_, self.p, obj_fn, self.t)
                curr_params = new_params
                pbar.update(1)
                iter += 1

        val_fn = partial(obj_fn, t=self.t)
        value = jax.vmap(val_fn)(curr_params, self.states_, self.p)

        return curr_params, value




