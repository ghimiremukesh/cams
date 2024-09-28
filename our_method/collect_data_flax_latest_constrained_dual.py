"""
Script to collect data for training value network at a specified time.
Uses alternating descent-ascent algorithm to simultaneously compute optimal policy for the players.
"""
import os

# os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=1"

#import pdb

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
from utils_jax import (final_cost_function, normalize_to_max_1d, normalize_to_max_final, normalize_to_max_1d_w_t,
                       final_cost_function_dual)

from utils_jax import x_next, running_cost
from functools import partial
from flax_picnn_dual import PICNN
from flax_picnn_dual import ModelConfig
from flax.training import checkpoints
from flax.linen import relu

# from BRT_Model.models import Siren
import torch
from BRT_Model import modules
from torch2jax import t2j


# load pytorch model
pytorch_model = modules.SingleBVPNet(in_features=9, out_features=1, type='sine', mode='mlp', num_hidden_layers=3, hidden_features=512)

pchk = torch.load('BRT_Model/pytorch_model/model_final.pth', weights_only=True)

pytorch_model.load_state_dict(pchk)

pytorch_model.eval()

ptoj_model = t2j(pytorch_model)
brt_params = {k: t2j(v) for k, v in pytorch_model.named_parameters()}

@jit
def apply_to_brt_model(x):
    return ptoj_model(x, state_dict=brt_params)


# to debugs
#jax.config.update("jax_disable_jit", True)

current_dir = os.path.dirname(__file__)

pp = configargparse.ArgumentParser()
pp.add_argument('--time', type=float, default=0.1,
                help='time-step to collect data')
opt = pp.parse_args()

# control bounds for the players
ux_high = 6
uy_high = 12
dx_high = 6
dy_high = 4

G = [utils_jax.GOAL_1, utils_jax.GOAL_2]  # goals for each type
R1 = np.array([[0.05, 0.],
               [0., 0.025]])

R2 = np.array([[0.05, 0],
               [0., 0.1]])

dt = 0.1
t = opt.time
t_next = t - dt
ts = np.around(np.arange(dt, 1 + dt, dt), 2)
t_step = int(np.where(ts == t)[0][0] + 1)

EPSILON = 1e-6

vel_bound_1_x = (1 - t) * ux_high
vel_bound_1_y = (1 - t) * uy_high

vel_bound_2_x = (1 - t) * dx_high
vel_bound_2_y = (1 - t) * dy_high

p_hat_bound = 1.4

# for GDA steps -- need to flip for p2 
bounds = jnp.concatenate((
    jnp.array([[-6., 6.]]),
    jnp.array([[-4., 4.]]),
    jnp.array([[-6., 6.]]),
    jnp.array([[-4., 4.]]),
    jnp.array([[-6., 6.]]),
    jnp.array([[-4., 4.]]),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-p_hat_bound, p_hat_bound]] * 6),
    jnp.array([[-6., 6.]]),
    jnp.array([[-12., 12.]]),
    jnp.array([[-6., 6.]]),
    jnp.array([[-12., 12.]]),
    jnp.array([[-6., 6.]]),
    jnp.array([[-12., 12.]]),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]


config = ModelConfig
model = PICNN(config)

if t != dt:
    load_dir = f'logs_cons_dual/t_{t_step - 1}/checkpoint_0/checkpoint'
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=load_dir, target=None)
    state_dict = flax.core.FrozenDict(state_dict)

@jit
def brt_value(t, x):
    """

    Args:
        t: time (backward)
        x: states

    Returns: value, if value <= 0 (infeasible, i.e. high penalty) otherwise feasible (0 penalty).
    """
    input_to_brt = jnp.concatenate((jnp.array([t]), x))
    scaled_input = normalize_to_max_1d_w_t(input_to_brt, 6, 12, 6, 4).reshape(-1, 9)  # bounds are for brt approximation # reshape for pytorch
    # pdb.set_trace()
    y = apply_to_brt_model(scaled_input)

    return y, y <= 0

@jit
def final_feasibility(x):
    r = 0.05
    x1 = x[:2]
    x2 = x[4:6]

    dist = jnp.linalg.norm(x1 - x2 + 1e-12)  # for numerical stability

    return dist - r




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


@partial(jit, static_argnums=(3, ))
def gradient_step(params, x, p, loss_fn):
    c = 5e-3 # 1e-2
    alpha = 4e-3
    beta = 0.4
    mu = 0.5
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(params, x, p)
    p2_params = params[:, :14]
    z = params[:, 14:28]
    p1_params = params[:, 28:34]
    v = params[:, 34:]
    #pdb.set_trace()
    new_p2_params = (p2_params - c * grad[:, :14]).clip(min_bounds[:14], max_bounds[:14])
    new_params = jnp.concatenate((new_p2_params, z, p1_params, v), axis=1)
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(new_params, x, p)
    new_p1_params = (p1_params + alpha * grad[:, 28:34]).clip(min_bounds[14:], max_bounds[14:])
    new_z = z + beta * (new_p2_params - z)
    new_v = v + mu * (new_p1_params - v)
    new_params = jnp.concatenate((new_p2_params, new_z, new_p1_params, new_v), axis=1)
    norm1 = jnp.linalg.norm(new_z - new_p2_params, ord=1)
    norm2 = jnp.linalg.norm(new_v - new_p1_params, ord=1)
    
    #pdb.set_trace()

    return new_params, norm1, norm2


@jit
def dual_value_inter(params, x, p_hat):
    r1 = 5
    r2 = 1
    v1 = params[:2]
    v2 = params[2:4]
    v3 = params[4:6]
    l1 = params[6]
    l2 = params[7]
    p_hat_1 = params[8:10]
    p_hat_2 = params[10:12]
    p_hat_3 = params[12:14]
    u1 = params[28:30]
    u2 = params[30:32]
    u3 = params[32:34]

    l3 = 1 - l1 - l2 + EPSILON

    # additional constraint
    cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3)**2) + 1e-12)


    x_next_1 = x_next(x, u1, v1)
    x_next_2 = x_next(x, u2, v2)
    x_next_3 = x_next(x, u3, v3)


    ins_cost_1 = dt * utils_jax.running_cost(u1, v1).reshape(-1, )
    ins_cost_2 = dt * utils_jax.running_cost(u2, v2).reshape(-1, )
    ins_cost_3 = dt * utils_jax.running_cost(u3, v3).reshape(-1, )


    input_1 = jnp.concat((x_next_1, p_hat_1 - ins_cost_1))
    input_2 = jnp.concat((x_next_2, p_hat_2 - ins_cost_2))
    input_3 = jnp.concat((x_next_3, p_hat_3 - ins_cost_3))


    v_bound_next_1_x = compute_bounds(1 - t + dt, 6)
    v_bound_next_1_y = compute_bounds(1 - t + dt, 12)

    v_bound_next_2_x = compute_bounds(1 - t + dt, 6)
    v_bound_next_2_y = compute_bounds(1 - t + dt, 4)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_1d(input_, v_bound_next_1_x, v_bound_next_1_y, v_bound_next_2_x, v_bound_next_2_y)
        return model.apply(state_dict, rescaled_input)

    val_1 = apply_to_model(input_1)
    val_2 = apply_to_model(input_2)
    val_3 = apply_to_model(input_3)


    lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
    p_hat_penalty = 8 * relu(cons_phat)

    #p2_params = params[:14]
    #z = params[14:28]
    #p1_params = params[28:34]
    #v = params[34:]

    #reg1 = (r1 / 2) * jnp.sum((p2_params - z) ** 2)
    #reg2 = (r2 / 2) * jnp.sum((p1_params - v) ** 2)

    from_brt_1, _ = brt_value(t_next, x_next_1)
    from_brt_2, _ = brt_value(t_next, x_next_2)

    objective = (l1 * val_1 + l2 * val_2 + l3 * val_3 + lam_penalty + p_hat_penalty -
                 1000 * relu(-from_brt_1) - 1000 * relu(-from_brt_2)) #+ reg1 - reg2

    return objective.reshape(())


@jit
def dual_value_final(params, x, p_hat):
    r1 = 5
    r2 = 1
    v1 = params[:2]
    v2 = params[2:4]
    v3 = params[4:6]
    l1 = params[6]
    l2 = params[7]
    p_hat_1 = params[8:10]
    p_hat_2 = params[10:12]
    p_hat_3 = params[12:14]
    u1 = params[28:30]
    u2 = params[30:32]
    u3 = params[32:34]

    l3 = 1 - l1 - l2 + EPSILON
    # use conditional
    # l3_true = lambda: 1 - l1 - l2
    # l3_false = lambda: 0.

    # ins_cost_1 = dt * utils_jax.running_cost(u1, v1).reshape(-1, )
    # ins_cost_2 = dt * utils_jax.running_cost(u2, v2).reshape(-1, )
    # ins_cost_3 = dt * utils_jax.running_cost(u3, v3).reshape(-1, )
    #
    # p_hat_1 = p_hat_1 - ins_cost_1
    # p_hat_2 = p_hat_2 - ins_cost_2
    # p_hat_3 = p_hat_3 - ins_cost_3

    # l3 = jax.lax.cond(l1 + l2 < 1, l3_true, l3_false)
    # p_hat_3 = (p_hat - l1 * p_hat_1 - l2 * p_hat_2)/l3
    cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3)**2) + 1e-12)


    x_next_1 = x_next(x, u1, v1)
    x_next_2 = x_next(x, u2, v2)
    x_next_3 = x_next(x, u3, v3)


    ins_cost_1 = dt * utils_jax.running_cost(u1, v1).reshape(-1, )
    ins_cost_2 = dt * utils_jax.running_cost(u2, v2).reshape(-1, )
    ins_cost_3 = dt * utils_jax.running_cost(u3, v3).reshape(-1, )

    p_hat_1 = p_hat_1 - ins_cost_1
    p_hat_2 = p_hat_2 - ins_cost_2
    p_hat_3 = p_hat_3 - ins_cost_3

    val_1 = final_cost_function_dual(x_next_1, p_hat_1)
    val_2 = final_cost_function_dual(x_next_2, p_hat_2)
    val_3 = final_cost_function_dual(x_next_3, p_hat_3)

    lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
    p_hat_penalty = 8 * relu(cons_phat)

    #p2_params = params[:14]
    #z = params[14:28]
    #p1_params = params[28:34]
    #v = params[34:]

    #reg1 = (r1/2) * jnp.sum((p2_params - z)**2)
    #reg2 = (r2/2) * jnp.sum((p1_params - v)**2)

    final_pen_1 = final_feasibility(x_next_1)
    final_pen_2 = final_feasibility(x_next_2)


    objective = (l1 * val_1 + l2 * val_2 + l3 * val_3 + p_hat_penalty +  lam_penalty -
                 1000 * relu(-final_pen_1) - 1000 * relu(-final_pen_2)) #+ reg1 - reg2
                 
    # debug
    #print(objective)

    return objective.reshape(())

def solve_minimax(params, x, p_hat, val_fn):
    curr_params = params
    iters = 200000
    iter = 0
    #p2_params_list = []
    #p1_params_ = []
    with tqdm(total=iters) as pbar:
        while iter <= iters:
            new_params, norm1, norm2 = gradient_step(curr_params, x, p_hat, val_fn)
            curr_params = new_params
            p2_params = curr_params[:, :14]
            p1_params = curr_params[:, 28:34]
            if not iter % 1000:
                print(norm1, norm2)
            pbar.update(1)
            iter += 1
            #p2_params_list.append(p2_params)
            #p1_params_.append(p1_params)

    final_params = jnp.concatenate((p2_params, p1_params), axis=1)
    value = jax.vmap(val_fn)(curr_params, x, p_hat)

    return final_params, value


def solve_and_collect(pos_bound, num_points):
    """
    For given bounds, solve the optimization problem and collect data.

    """
    # sample uniform randomly
    key = jax.random.PRNGKey(0)


    n = 2
    while True:
        key1, key2, key3, key4, key5, key6, key7, key8 = jax.random.split(key, 8)
        pos = jax.random.uniform(key1, (n * num_points, 4), minval=-pos_bound, maxval=pos_bound)
        vel_1_x = jax.random.uniform(key2, (n * num_points, 1), minval=-vel_bound_1_x, maxval=vel_bound_1_x)
        vel_1_y = jax.random.uniform(key3, (n * num_points, 1), minval=-vel_bound_1_y, maxval=vel_bound_1_y)
        vel_2_x = jax.random.uniform(key4, (n * num_points, 1), minval=-vel_bound_2_x, maxval=vel_bound_2_x)
        vel_2_y = jax.random.uniform(key5, (n * num_points, 1), minval=-vel_bound_2_y, maxval=vel_bound_2_y)

        states_ = jnp.concatenate((pos[:, :2], vel_1_x.reshape(-1, 1), vel_1_y.reshape(-1, 1), pos[:, 2:4], vel_2_x.reshape(-1, 1), 
                                   vel_2_y.reshape(-1, 1)), axis=1)

        # pdb.set_trace()
        _, infeasible = jax.vmap(brt_value)(t * jnp.ones_like(states_[:, 0]), states_)
        states = states_[~infeasible.squeeze()]
        # pdb.set_trace()
        if len(states) >= num_points:
            states = states[:num_points, :]
            break
        n += 1

    params_d = jax.random.uniform(key5, (1, 6), minval=-1, maxval=1)
    params_d_lam = jnp.array([[EPSILON, 1 - EPSILON]])
    params_d_phat = jax.random.uniform(key6, (1, 6), minval=-p_hat_bound, maxval=p_hat_bound)
    params_u = jax.random.uniform(key6, (1, 6), minval=-1, maxval=1)
    p_hat = jax.random.uniform(key4, (num_points, 2), minval=-p_hat_bound, maxval=p_hat_bound)  # p_hat_bound

    params = jnp.concatenate((params_d, params_d_lam, params_d_phat, params_u), axis=1)
    params = np.repeat(params, repeats=num_points, axis=0)

    params = jnp.repeat(params, repeats=2, axis=0).reshape(num_points, -1)  # repeat for dsgda
    
    #pdb.set_trace()

    if t == dt:
        final_params, results = solve_minimax(params, states, p_hat, dual_value_final)
    else:
        final_params, results = solve_minimax(params, states, p_hat, dual_value_inter)


    if t!= 1:
        coords = jax.vmap(partial(normalize_to_max_final, v1x_max=vel_bound_1_x, v1y_max=vel_bound_1_y,
                                  v2x_max=vel_bound_2_x, v2y_max=vel_bound_2_y))(states)
    else:
        coords = states

    coords = jnp.concatenate((coords, p_hat), axis=1)
    value = jnp.vstack(results)

    return coords, value


if __name__ == '__main__':
    save_root = f'train_data_cons_dual/'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # check time
    start_time = time.time()

    num_points = 100000
    states, values = solve_and_collect(1, num_points)

    end_time = time.time()

    print(f'total time: {end_time - start_time}')

    gt = {'states': states,
          'values': values}

    scio.savemat(os.path.join(save_root, f'train_data_t_{t:.2f}.mat'), gt)

