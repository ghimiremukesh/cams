"""
Script to collect data for training value network at a specified time.
Uses alternating descent-ascent algorithm to simultaneously compute optimal policy for the players.
"""
import os
import sys
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
from utils_jax import corner_final_cost, normalize_to_max_corner_1d, normalize_to_max_corner_final

from utils_jax import corner_dynamics, running_cost_corner
from functools import partial
from flax_picnn_corner import PICNN
from flax_picnn_corner import ModelConfig
from flax.training import checkpoints

# to debug
# jax.config.update("jax_disable_jit", True)

current_dir = os.path.dirname(__file__)

pp = configargparse.ArgumentParser()
pp.add_argument('--time', type=float, default=0.1,
                help='time-step to collect data')
opt = pp.parse_args()

# control bounds for the players
u_high = 12
d_high = 12

G = [utils_jax.GOAL_1, utils_jax.GOAL_2]  # goals for each type
R1 = np.array([[0.01, 0.],
               [0., 0.005]])

R2 = np.array([[0.01, 0],
               [0., 0.02]])

dt = 0.1
t = opt.time
t_next = t - dt

t_final = 0.4 
ts = np.around(np.arange(dt, t_final + dt, dt), 2) 
t_step = int(np.where(ts == t)[0][0] + 1)

EPSILON = 1e-6

vel_bound_1 = (0.4 - t) * u_high
vel_bound_2 = (0.4 - t) * d_high

# for GDA steps
bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 8),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-12., 12.]] * 4),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]

model_dir = os.path.join(current_dir, f'logs_corner/t_{t_step - 1}/checkpoint_0/checkpoint')
config = ModelConfig
model = PICNN(config)

if t != dt:
    load_dir = f'logs_corner/t_{t_step - 1}/checkpoint_0/checkpoint'
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
def value_inter(params, x, p):
    u1_1 = params[:2]
    u2_1 = params[2:4]
    u1_2 = params[4:6]
    u2_2 = params[6:8]
    a1 = params[8]
    a2 = params[9]
    v1 = params[20:22]
    v2 = params[22:24]


    p_u1 = a1 * p + a2 * (1 - p)
    p_u2 = 1 - p_u1

    # posteriors
    pos_1 = a1 * p / p_u1

    pos_2 = (1 - a1) * p / p_u2

    x_next_1 = corner_dynamics(x, u1_1, u2_1, v1, dt=dt)
    x_next_2 = corner_dynamics(x, u1_2, u2_2, v2, dt=dt)

    input_1 = jnp.concat((x_next_1, pos_1))
    input_2 = jnp.concat((x_next_2, pos_2))

    v_bound_next = compute_bounds(t_final - t + dt, 12)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_corner_1d(input_, v_bound_next)
        return model.apply(state_dict, rescaled_input)

    val_1 = apply_to_model(input_1)
    val_2 = apply_to_model(input_2)

    ins_cost_1 = dt * running_cost_corner(u1_1, u2_1, v1).reshape(-1, )
    ins_cost_2 = dt * running_cost_corner(u1_2, u2_2, v2).reshape(-1, )

    final_cost_1 = val_1 + ins_cost_1
    final_cost_2 = val_2 + ins_cost_2

    objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

    return objective.reshape(())

@partial(jit, static_argnums=(3, ))
def gradient_step(params, x, p, loss_fn):
    c = 1e-1
    alpha = 5e-2
    beta = 0.4
    mu = 0.5
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(params, x, p)
    p1_params = params[:, :10]
    z = params[:, 10:20]
    p2_params = params[:, 20:24]
    v = params[:, 24:]
    new_p1_params = (p1_params - c * grad[:, :10]).clip(min_bounds[:10], max_bounds[:10])
    new_params = jnp.concatenate((new_p1_params, z, p2_params, v), axis=1)
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(new_params, x, p)
    new_p2_params = (p2_params + alpha * grad[:, 20:24]).clip(min_bounds[10:], max_bounds[10:])
    new_z = z + beta * (new_p1_params - z)
    new_v = v + mu * (new_p2_params - v)
    new_params = jnp.concatenate((new_p1_params, new_z, new_p2_params, new_v), axis=1)
    norm1 = jnp.linalg.norm(new_z - new_p1_params, ord=1)
    norm2 = jnp.linalg.norm(new_v - new_p2_params, ord=1)

    return new_params, norm1, norm2


@jit
def value_final(params, x, p):
    u1_1 = params[:2]
    u2_1 = params[2:4]
    u1_2 = params[4:6]
    u2_2 = params[6:8]
    a1 = params[8]
    a2 = params[9]
    v1 = params[20:22]
    v2 = params[22:24]


    p_u1 = a1 * p + a2 * (1 - p)
    p_u2 = 1 - p_u1

    # posteriors
    pos_1 = a1 * p / p_u1

    pos_2 = (1 - a1) * p / p_u2

    x_next_1 = corner_dynamics(x, u1_1, u2_1, v1, dt=dt)
    x_next_2 = corner_dynamics(x, u1_2, u2_2, v2, dt=dt)

    val_1 = corner_final_cost(x_next_1, pos_1)
    val_2 = corner_final_cost(x_next_2, pos_2)
    
    #val_1 = corner_final_cost_soft_roles(x_next_1, pos_1)
    #val_2 = corner_final_cost_soft_roles(x_next_2, pos_2)

    ins_cost_1 = dt * running_cost_corner(u1_1, u2_1, v1).reshape(-1, )
    ins_cost_2 = dt * running_cost_corner(u1_2, u2_2, v2).reshape(-1, )

    objective = p_u1 * (val_1 + ins_cost_1) + p_u2 * (val_2 + ins_cost_2)

    return objective.reshape(())

def solve_minimax(params, x, p, val_fn):
    curr_params = params
    iters = 80000
    #p1 = []
    #p1.append(curr_params[:, :10])
    iter = 0
    # p2_params_list = []
    with tqdm(total=iters) as pbar:
        while iter <= iters:
            new_params, norm1, norm2 = gradient_step(curr_params, x, p, val_fn)
            curr_params = new_params
            p1_params = curr_params[:, :10]
            p2_params = curr_params[:, 20:24]
            if not iter % 1000:
                print(norm1)
            pbar.update(1)
            # p2_params_list.append(p2_params)
            iter += 1

    final_params = jnp.concatenate((p1_params, p2_params), axis=1)
    value = jax.vmap(val_fn)(curr_params, x, p)

    return final_params, value


def solve_and_collect(pos_bound, num_points):
    """
    For given bounds, solve the optimization problem and collect data.

    """
    # sample uniform randomly
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)
    pos = jax.random.uniform(key1, (num_points, 6), minval=-pos_bound, maxval=pos_bound)
    vel_1 = jax.random.uniform(key2, (num_points, 2), minval=-vel_bound_1, maxval=vel_bound_1)
    vel_2 = jax.random.uniform(key3, (num_points, 2), minval=-vel_bound_1, maxval=vel_bound_1)
    vel_3 = jax.random.uniform(key4, (num_points, 2), minval=-vel_bound_2, maxval=vel_bound_2)
    p = jax.random.uniform(key5, (num_points, 1), minval=EPSILON, maxval=1-EPSILON)

    states_ = jnp.concatenate((pos[:, :2], vel_1, pos[:, 2:4], vel_2, pos[:, 4:6], vel_3), axis=1)

    params_u = jax.random.uniform(key6, (1, 8), minval=-1, maxval=1)
    params_up = jnp.array([[EPSILON, 1-EPSILON]])
    params_d = jax.random.uniform(key7, (1, 4), minval=-1, maxval=1)

    params = jnp.concatenate((params_u, params_up, params_d), axis=1)
    params = np.repeat(params, repeats=num_points, axis=0)

    params = jnp.repeat(params, repeats=2, axis=0).reshape(num_points, -1)  # repeat for dsgda


    if t == dt:
        final_params, results = solve_minimax(params, states_, p, value_final)
    else:
        final_params, results = solve_minimax(params, states_, p, value_inter)

    if np.round(t, 2)!= t_final:
        coords = jax.vmap(partial(normalize_to_max_corner_final, v_max=vel_bound_1))(states_)
    else:
        coords = states_

    coords = jnp.concatenate((coords, p), axis=1)

    value = jnp.vstack(results)

    return coords, value


if __name__ == '__main__':
    save_root = f'train_data_corner/'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # check time
    start_time = time.time()

    num_points = 250000
    states, values = solve_and_collect(1, num_points)

    end_time = time.time()

    print(f'total time: {end_time - start_time}')

    gt = {'states': states,
          'values': values}

    scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)
