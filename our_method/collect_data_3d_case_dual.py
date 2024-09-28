"""
Script to collect data for training dual value network at a specified time.
Uses alternating descent-ascent algorithm to simultaneously compute optimal policy for the players.
"""
import os
import sys
from typing import final

import flax.core
import jax.random
import utils_jax


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax.numpy as jnp
import configargparse
import scipy.io as scio
from tqdm import tqdm
import time
from jax import jit
from utils_jax import final_cost_function_3d_dual, normalize_to_max_3d, normalize_to_max_final_3d

from utils_jax import x_next_3d, running_cost_3d
from functools import partial
from flax_picnn_3d_dual import PICNN
from flax_picnn_3d_dual import ModelConfig
from flax.training import checkpoints
from flax.linen.activation import relu

# to debug
# jax.config.update("jax_disable_jit", True)

current_dir = os.path.dirname(__file__)

pp = configargparse.ArgumentParser()
pp.add_argument('--time', type=float, default=0.25,
                help='time-step to collect data')
opt = pp.parse_args()

# control bounds for the players
u_high = 12
d_high = 12

G = [utils_jax.GOAL_1_3d, utils_jax.GOAL_2_3d]  # goals for each type
R1 = np.array([[0.05, 0., 0.],[0., 0.05, 0.],
               [0., 0., 0.025]])

R2 = np.array([[0.05, 0., 0.],[0., 0.05, 0.],
               [0., 0., 0.1]])

dt = 0.25
t = opt.time
t_next = t - dt
ts = np.around(np.arange(dt, 1 + dt, dt), 2)
t_step = int(np.where(ts == t)[0][0] + 1)

EPSILON = 1e-6
p_hat_bound = 1.8

vel_bound_1 = (1 - t) * u_high
vel_bound_2 = (1 - t) * d_high

# for GDA steps
bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 9),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-p_hat_bound, p_hat_bound]] * 6),
    jnp.array([[-12., 12.]] * 9),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]


config = ModelConfig
model = PICNN(config)

if t != dt:
    load_dir = f'logs_3d_case_dual/t_{t_step - 1}/checkpoint_0/checkpoint'
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=None)
    state_dict = flax.core.FrozenDict(state_dict)

@jit
def compute_bounds(time_step, a_max):
    """
    Dynamically adjust velocity bounds for each time-step, assuming players always start at rest.
    ``
    eg. if t = 0.5, v_max = 0 + u_high * 0.5 = 6. This is the maximum possible velocity at t = 0.5
    ``
    params:
        time_step: current "forward" time. NOT BACKWARD, if using backward make sure to pass (1-t)

    return:
        bound b
    """
    return time_step * a_max

@jit
def dual_value_inter(params, x, p_hat):
    v1 = params[:3]
    v2 = params[3:6]
    v3 = params[6:9]
    l1 = params[9]
    l2 = params[10]
    p_hat_1 = params[11:13]
    p_hat_2 = params[13:15]
    p_hat_3 = params[15:17]
    u1 = params[34:37]
    u2 = params[37:40]
    u3 = params[40:43]

    l3 = 1 - l1 - l2 + EPSILON


    cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3)**2) + 1e-12)


    x_next_1 = x_next_3d(x, u1, v1, dt=dt)
    x_next_2 = x_next_3d(x, u2, v2, dt=dt)
    x_next_3 = x_next_3d(x, u3, v3, dt=dt)


    ins_cost_1 = dt * running_cost_3d(u1, v1).reshape(-1, )
    ins_cost_2 = dt * running_cost_3d(u2, v2).reshape(-1, )
    ins_cost_3 = dt * running_cost_3d(u3, v3).reshape(-1, )

    p_hat_1 = p_hat_1 - ins_cost_1
    p_hat_2 = p_hat_2 - ins_cost_2
    p_hat_3 = p_hat_3 - ins_cost_3


    input_1 = jnp.concat((x_next_1, p_hat_1))
    input_2 = jnp.concat((x_next_2, p_hat_2))
    input_3 = jnp.concat((x_next_3, p_hat_3))


    v_bound_next = compute_bounds(1 - t + dt, 12)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_3d(input_, v_bound_next, v_bound_next, v_bound_next, v_bound_next, v_bound_next, v_bound_next)
        return model.apply(state_dict, rescaled_input)

    val_1 = apply_to_model(input_1)
    val_2 = apply_to_model(input_2)
    val_3 = apply_to_model(input_3)


    lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
    p_hat_penalty = 8 * relu(cons_phat)
    objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + p_hat_penalty + lam_penalty

    return objective.reshape(())


@jit
def dual_value_final(params, x, p_hat):
    v1 = params[:3]
    v2 = params[3:6]
    v3 = params[6:9]
    l1 = params[9]
    l2 = params[10]
    p_hat_1 = params[11:13]
    p_hat_2 = params[13:15]
    p_hat_3 = params[15:17]
    u1 = params[34:37]
    u2 = params[37:40]
    u3 = params[40:43]

    l3 = 1 - l1 - l2 + EPSILON


    cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3)**2) + 1e-12)


    x_next_1 = x_next_3d(x, u1, v1, dt=dt)
    x_next_2 = x_next_3d(x, u2, v2, dt=dt)
    x_next_3 = x_next_3d(x, u3, v3, dt=dt)


    ins_cost_1 = dt * running_cost_3d(u1, v1).reshape(-1, )
    ins_cost_2 = dt * running_cost_3d(u2, v2).reshape(-1, )
    ins_cost_3 = dt * running_cost_3d(u3, v3).reshape(-1, )

    p_hat_1 = p_hat_1 - ins_cost_1
    p_hat_2 = p_hat_2 - ins_cost_2
    p_hat_3 = p_hat_3 - ins_cost_3

    val_1 = final_cost_function_3d_dual(x_next_1, p_hat_1)
    val_2 = final_cost_function_3d_dual(x_next_2, p_hat_2)
    val_3 = final_cost_function_3d_dual(x_next_3, p_hat_3)

    lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
    p_hat_penalty = 8 * relu(cons_phat)
    
    objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + p_hat_penalty + lam_penalty

    return objective.reshape(())


@partial(jit, static_argnums=(3, ))
def gradient_step(params, x, p, loss_fn):
    c = 5e-2 # 1e-2
    alpha = 1e-2
    beta = 0.4
    mu = 0.5
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(params, x, p)
    p2_params = params[:, :17]
    z = params[:, 17:34]
    p1_params = params[:, 34:43]
    v = params[:, 43:]
    new_p2_params = (p2_params - c * grad[:, :17]).clip(min_bounds[:17], max_bounds[:17])
    new_params = jnp.concatenate((new_p2_params, z, p1_params, v), axis=1)
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(new_params, x, p)
    new_p1_params = (p1_params + alpha * grad[:, 34:43]).clip(min_bounds[17:], max_bounds[17:])
    new_z = z + beta * (new_p2_params - z)
    new_v = v + mu * (new_p1_params - v)
    new_params = jnp.concatenate((new_p2_params, new_z, new_p1_params, new_v), axis=1)
    norm1 = jnp.linalg.norm(new_z - new_p2_params, ord=1)
    norm2 = jnp.linalg.norm(new_v - new_p1_params, ord=1)

    return new_params, norm1, norm2


def solve_minimax(params, x, p_hat, val_fn):
    curr_params = params
    iters = 200000
    iter = 0
    with tqdm(total=iters) as pbar:
        while iter <= iters:
            new_params, norm1, norm2 = gradient_step(curr_params, x, p_hat, val_fn)
            curr_params = new_params
            p2_params = curr_params[:, :17]
            p1_params = curr_params[:, 34:43]
            if not iter % 1000:
                print(norm1)
            pbar.update(1)
            iter += 1

    final_params = jnp.concatenate((p2_params, p1_params), axis=1)
    value = jax.vmap(val_fn)(curr_params, x, p_hat)

    return final_params, value


def solve_and_collect(pos_bound, num_points):
    """
    For given bounds, solve the optimization problem and collect data.

    """
    # sample uniform randomly
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
    pos = jax.random.uniform(key1, (num_points, 6), minval=-pos_bound, maxval=pos_bound)
    vel_1 = jax.random.uniform(key2, (num_points, 3), minval=-vel_bound_1, maxval=vel_bound_1)
    vel_2 = jax.random.uniform(key3, (num_points, 3), minval=-vel_bound_2, maxval=vel_bound_2)
    p_hat = jax.random.uniform(key4, (num_points, 2), minval=-p_hat_bound, maxval=p_hat_bound)

    states_ = jnp.concatenate((pos[:, :3], vel_1, pos[:, 3:6], vel_2), axis=1)

    params_d = jax.random.uniform(key5, (1, 9), minval=-1, maxval=1)
    params_d_lam = jnp.array([[EPSILON, 1-EPSILON]])
    params_d_phat = jax.random.uniform(key6, (1, 6), minval=-p_hat_bound, maxval=p_hat_bound)
    params_u = jax.random.uniform(key6, (1, 9), minval=-1, maxval=1)

    params = jnp.concatenate((params_d, params_d_lam, params_d_phat, params_u), axis=1)
    params = np.repeat(params, repeats=num_points, axis=0)

    params = jnp.repeat(params, repeats=2, axis=0).reshape(num_points, -1)  # repeat for dsgda


    if t == dt:
        final_params, results = solve_minimax(params, states_, p_hat, dual_value_final)
    else:
        final_params, results = solve_minimax(params, states_, p_hat, dual_value_inter)


    if t != 1:
        coords = jax.vmap(partial(normalize_to_max_final_3d, v1x_max=vel_bound_1, v1y_max=vel_bound_1, v1z_max=vel_bound_1, 
                                v2x_max=vel_bound_2, v2y_max=vel_bound_2, v2z_max=vel_bound_2))(states_)
    else:
        coords = states_

    coords = jnp.concatenate((coords, p_hat), axis=1)

    return coords, results


if __name__ == '__main__':
    save_root = f'train_data_3d_dual'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # check time
    start_time = time.time()

    num_points = 250000
    states, values = solve_and_collect(1, num_points)

    end_time = time.time()

    print(f'total time: {end_time - start_time}')

    gt = {'states': states,
          'values': values.reshape(-1, 1)}

    scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)