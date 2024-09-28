import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

import jax
import matplotlib.pyplot as plt
from flax.training import checkpoints
import numpy as np
import jax.numpy as jnp
from flax_picnn import PICNN, ModelConfig
import optax
import utils_jax
from jax import jit
from tqdm import tqdm
from functools import partial
import matplotlib
from utils_jax import (final_cost_function, running_cost, x_next, normalize_to_max_1d, compute_bounds,
                       normalize_to_max_1d_w_t)
import torch
from torch2jax import t2j
import scipy.io as scio
from BRT_Model import modules
import flax

matplotlib.use('TkAgg')

# jax.config.update("jax_disable_jit", True)

# load pytorch model
pytorch_model = modules.SingleBVPNet(in_features=9, out_features=1, type='sine', mode='mlp', num_hidden_layers=3, hidden_features=512)

pchk = torch.load('../BRT_Model/pytorch_model/model_final.pth', weights_only=True, map_location='cpu')

pytorch_model.load_state_dict(pchk)

pytorch_model.eval()

ptoj_model = t2j(pytorch_model)
brt_params = {k: t2j(v) for k, v in pytorch_model.named_parameters()}

@jit
def apply_to_brt_model(x):
    return ptoj_model(x, state_dict=brt_params)

config = ModelConfig
model = PICNN(config)
EPSILON = 1e-6


# for GDA steps
bounds = jnp.concatenate((
    jnp.array([[-6., 6.]]),
    jnp.array([[-12., 12.]]),
    jnp.array([[-6., 6.]]),
    jnp.array([[-12., 12.]]),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-6., 6.]]),
    jnp.array([[-4., 4.]]),
    jnp.array([[-6., 6.]]),
    jnp.array([[-4., 4.]]),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]


chkpnts = [f'../logs_cons/t_{i}/checkpoint_0/checkpoint' for i in range(1, 11)] 


total_steps = 10

dt = 0.1

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

    dist = jnp.linalg.norm(x1 - x2)

    return dist - r

@partial(jit, static_argnums=(3, 4))
def gradient_step(params, x, p, loss_fn, t):
    c = 5e-3
    alpha = 4e-3
    beta = 0.4
    mu = 0.5
    grad = jax.grad(loss_fn, argnums=0)(params, x, p, t)
    p1_params = params[:6]
    z = params[6:12]
    p2_params = params[12:16]
    v = params[16:]
    new_p1_params = (p1_params - c * grad[:6]).clip(min_bounds[:6], max_bounds[:6])
    new_params = jnp.concatenate((new_p1_params, z, p2_params, v))
    grad = jax.grad(loss_fn, argnums=0)(new_params, x, p, t)
    new_p2_params = (p2_params + alpha * grad[12:16]).clip(min_bounds[6:], max_bounds[6:])
    new_z = z + beta * (new_p1_params - z)
    new_v = v + mu * (new_p2_params - v)
    new_params = jnp.concatenate((new_p1_params, new_z, new_p2_params, new_v))
    norm1 = jnp.linalg.norm(new_z - new_p1_params, ord=1)
    norm2 = jnp.linalg.norm(new_v - new_p2_params, ord=1)

    return new_params, norm1, norm2


@partial(jit, static_argnums=(3,))
def dsgda_obj_final(params, x, p, t):
    r1 = 5
    r2 = 1
    u1 = params[:2]
    u2 = params[2:4]
    a1 = params[4]
    a2 = params[5]
    v1 = params[12:14]
    v2 = params[14:16]

    p_u1 = a1 * p + a2 * (1 - p)
    p_u2 = 1 - p_u1

    # posteriors
    pos_1 = a1 * p / p_u1

    pos_2 = (1 - a1) * p / p_u2

    ## no taylor expansion
    x_next_1 = x_next(x, u1, v1)
    x_next_2 = x_next(x, u2, v2)

    val_1 = final_cost_function(x_next_1, pos_1)
    val_2 = final_cost_function(x_next_2, pos_2)

    ins_1 = 0.1 * running_cost(u1, v1)

    ins_2 = 0.1 * running_cost(u2, v2)

    p1_params = params[:6]
    z = params[6:12]
    p2_params = params[12:16]
    v = params[16:]

    regularizer_1 = (r1 / 2) * jnp.sum((p1_params - z) ** 2)
    regularizer_2 = (r2 / 2) * jnp.sum((p2_params - v) ** 2)

    objective = p_u1 * (val_1 + ins_1) + p_u2 * (val_2 + ins_2) + regularizer_1 - regularizer_2


    final_pen_1 = final_feasibility(x_next_1)
    final_pen_2 = final_feasibility(x_next_2)

    objective = objective + 1000 * flax.linen.relu(-final_pen_1)  + 1000 * flax.linen.relu(-final_pen_2)

    return objective.reshape(())

@partial(jit, static_argnums=(3,))
def dsgda_obj_inter(params, x, p, t):
    chkpt_idx = int(10 * t) - 1  # current model
    weights_dir = chkpnts[chkpt_idx - 1]  # next model, we need next model in optimization
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=weights_dir, target=None)
    r1 = 5
    r2 = 1
    u1 = params[:2]
    u2 = params[2:4]
    a1 = params[4]
    a2 = params[5]
    v1 = params[12:14]
    v2 = params[14:16]

    p_u1 = a1 * p + a2 * (1 - p)
    p_u2 = 1 - p_u1

    # posteriors
    pos_1 = a1 * p / p_u1
    pos_2 = (1 - a1) * p / p_u2

    x_next_1 = x_next(x, u1, v1)
    x_next_2 = x_next(x, u2, v2)

    input_1 = jnp.concat((x_next_1, pos_1))
    input_2 = jnp.concat((x_next_2, pos_2))

    v_bound_next_1_x = compute_bounds(1 - t + dt, 6)
    v_bound_next_1_y = compute_bounds(1 - t + dt, 12)

    v_bound_next_2_x = compute_bounds(1 - t + dt, 6)
    v_bound_next_2_y = compute_bounds(1 - t + dt, 4)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_1d(input_, v_bound_next_1_x, v_bound_next_1_y, v_bound_next_2_x, v_bound_next_2_y)
        return model.apply(state_dict, rescaled_input)

    v_next_1 = apply_to_model(input_1)
    v_next_2 = apply_to_model(input_2)

    ins_cost_1 = dt * running_cost(u1, v1).reshape(-1, )
    ins_cost_2 = dt * running_cost(u2, v2).reshape(-1, )

    p1_params = params[:6]
    z = params[6:12]
    p2_params = params[12:16]
    v = params[16:]

    regularizer_1 = (r1 / 2) * jnp.sum((p1_params - z) ** 2)
    regularizer_2 = (r2 / 2) * jnp.sum((p2_params - v) ** 2)

    from_brt_1, _ = brt_value(jnp.round(t - dt, 3), x_next_1)
    from_brt_2, _ = brt_value(jnp.round(t - dt, 3), x_next_2)

    objective = p_u1 * (v_next_1 + ins_cost_1) + p_u2 * (v_next_2 + ins_cost_2) + regularizer_1 - regularizer_2

    objective = objective + 1000 * flax.linen.relu(-from_brt_1) + 1000 * flax.linen.relu(-from_brt_2)

    return objective.reshape(())

def value_inter(params, x, p, t):
    chkpt_idx = int(10 * t) - 1  # current model
    weights_dir = chkpnts[chkpt_idx - 1]  # next model, we need next model in optimization
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=weights_dir, target=None)
    r1 = 5
    r2 = 1
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

    v_bound_next_1_x = compute_bounds(1 - t + dt, 6)
    v_bound_next_1_y = compute_bounds(1 - t + dt, 12)

    v_bound_next_2_x = compute_bounds(1 - t + dt, 6)
    v_bound_next_2_y = compute_bounds(1 - t + dt, 4)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_1d(input_, v_bound_next_1_x, v_bound_next_1_y, v_bound_next_2_x, v_bound_next_2_y)
        return model.apply(state_dict, rescaled_input)

    v_next_1 = apply_to_model(input_1)
    v_next_2 = apply_to_model(input_2)

    ins_cost_1 = dt * running_cost(u1, v1).reshape(-1, )
    ins_cost_2 = dt * running_cost(u2, v2).reshape(-1, )


    from_brt_1, _ = brt_value(jnp.round(t - dt, 3), x_next_1)
    from_brt_2, _ = brt_value(jnp.round(t - dt, 3), x_next_2)

    objective = p_u1 * (v_next_1 + ins_cost_1) + p_u2 * (v_next_2 + ins_cost_2) #+ regularizer_1 - regularizer_2

    objective = objective + 1000 * flax.linen.relu(-from_brt_1) + 1000 * flax.linen.relu(-from_brt_2)

    return objective.reshape(())

def solve_minimax(params, x, p, t):
    if t == dt:
        obj_fn = dsgda_obj_final
    else:
        obj_fn = dsgda_obj_inter

    curr_params = params
    iters = 120000
    iter = 0

    # for plotting
    p1_params_list = []
    with tqdm(total=iters) as pbar:
        while iter <= iters:
            new_params, norm1, norm2 = gradient_step(curr_params, x, p, obj_fn, t)
            curr_params = new_params
            p1_params = curr_params[:6]
            p2_params = curr_params[12:16]
            # if not iter % 1000:
            #     print(norm1)
            pbar.update(1)
            iter += 1
            # p1_params_list.append(p2_params)  # plot one at at time and check
    #
    final_params = jnp.concatenate((p1_params, p2_params))


    return final_params


if __name__ == '__main__':
    R1 = jnp.array([[0.05, 0], [0, 0.025]])
    R2 = jnp.array([[0.05, 0], [0, 0.1]])
    fig, axs = plt.subplots(3, 1)
    Us = []
    Ds = []
    xs = [-0.5, 0.5]
    x2s = [-0.4, 0.4]
    for run in range(1):  # number of simulations to run
        type_map = {0: 'Goal 2', 1: 'Goal 1'}
        p1_type = 1
        print(f'P1 type is: {type_map[p1_type]}')

        p_t = 0.5

        curr_x1 = jnp.array([[xs[run], 0, 0, 0]])
        curr_x2 = jnp.array([[x2s[run], -0.1, 0, 0]])

        # curr_x1 = jnp.array([[-0.5, 0, 0, 0]])
        # curr_x2 = jnp.array([[0.5, 0, 0, 0]])

        states = []
        U = []
        D = []

        t = 1
        ts = np.arange(0, 1, dt)

        curr_state = jnp.hstack((curr_x1, curr_x2))

        # check if the initial state is feasible
        _, infeasible = brt_value(t, curr_state.reshape(-1, ))

        if infeasible:
            raise ValueError("Initial state is infeasible. Pick another one.")

        states.append(jnp.hstack((curr_state, jnp.array([[p_t]]))))

        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)

        # for the probability of the trajectory
        flag = None  # to check if the interaction was infeasible

        while t >= dt:
            params_u = jax.random.uniform(key1, (1, 4), minval=0, maxval=1)
            params_up = jnp.array([[EPSILON, 1 - EPSILON]])
            params_d = jax.random.uniform(key2, (1, 4), minval=0, maxval=1)
            params = jnp.concatenate((params_u, params_up, params_d), axis=1)

            params = jnp.repeat(params, repeats=2) # repeat for dsgda

            params = solve_minimax(params.reshape(-1, ), curr_state.reshape(-1, ), jnp.array([p_t]), t)

            #debug
            # test_val = value_inter(params.reshape(-1, ), curr_state.reshape(-1, ), jnp.array([p_t]), t)

            print(f'Parameters are: {params}')
            a1 = params.reshape(-1, )[4]
            a2 = params.reshape(-1, )[5]

            p_u1 = a1 * p_t + a2 * (1 - p_t)
            p_u2 = 1 - p_u1

            # posteriors
            pos_1 = a1 * p_t / p_u1
            pos_2 = (1 - a1) * p_t / p_u2

            u1 = params[:2]
            u2 = params[2:4]
            v1 = params[6:8]
            v2 = params[8:10]

            if p1_type == 1:
                u_1_prob = a1
                u_2_prob = 1 - a1
            else:
                u_1_prob = a2
                u_2_prob = 1 - a2

            print(f'At t = {1 - t:.2f}, P1 with type:\"{type_map[p1_type]}\" has the following choices: \n')
            print(f'P1 could take action {u1} with probability {u_1_prob:.2f} and update belief to {pos_1:.2f}')
            print(f'P1 could take action {u2} with probability {u_2_prob:.2f} and update belief to {pos_2:.2f}')

            dist = [u_1_prob, u_2_prob]
            a_idx = [0, 1]
            action_idx = random.choices(a_idx, dist)[0]

            if action_idx == 0:
                p1_action = u1
                p2_action = v1
                p_t = pos_1
            else:
                p1_action = u2
                p2_action = v2
                p_t = pos_2

            print(f'P1 chooses: {p1_action}, and P2 chooses: {p2_action}. The belief is now {p_t:.2f}')

            U.append(p1_action)
            D.append(p2_action)


            curr_state = x_next(curr_state.reshape(-1, ), p1_action, p2_action)

            # clip the states just to be safe
            curr_state = jnp.concatenate((curr_state[:2].clip(-1, 1), curr_state[2:4], curr_state[4:6].clip(-1, 1),
                                         curr_state[6:]))

            t = np.round(t - dt, 3)

            _, infeasible = brt_value(t, curr_state)

            if infeasible and t!= 0:
                # raise ValueError('New state is infeasible! Game Ends!')
                flag = jnp.ones(10)
                print('Infeasibe at: ', t)
            else:
                print(f'current state is {curr_state}')

            states.append(jnp.hstack((curr_state, jnp.array([p_t]))))



        print(f'final state is: {curr_state}')
        times = np.arange(0, 1.1, 0.1)
        g1 = np.array([0, 1])
        g2 = np.array([0, -1])

        states = np.vstack(states)
        x1 = states[:, 0]
        y1 = states[:, 1]
        x2 = states[:, 4]
        y2 = states[:, 5]
        p_t = states[:, -1]

        U = np.vstack(U)
        D = np.vstack(D)

        reach_probs = np.vstack(reach_probs).reshape(-1, )

        axs[2].plot(np.linspace(0, 1 - dt, total_steps), U[:, 0], label='$u_x$')
        axs[2].plot(np.linspace(0, 1 - dt, total_steps), U[:, 1], label='$u_y$')
        axs[2].plot(np.linspace(0, 1 - dt, total_steps), D[:, 0], '-.', label='$d_x$')
        axs[2].plot(np.linspace(0, 1 - dt, total_steps), D[:, 1], '--', label='$d_y$')
        # axs[2].set_xlim([-0.05, 1])
        axs[2].legend()
        axs[0].set_title(f"Goal Selected: {type_map[p1_type]} ")
        if p1_type == 0:
            axs[0].scatter(g1[0], g1[1], marker='o', facecolor='none', edgecolor='magenta')
            axs[0].scatter(g2[0], g2[1], marker='o', facecolor='magenta', edgecolor='magenta')
        else:
            axs[0].scatter(g1[0], g1[1], marker='o', facecolor='magenta', edgecolor='magenta')
            axs[0].scatter(g2[0], g2[1], marker='o', facecolor='none', edgecolor='magenta')
        axs[0].annotate("1", (g1[0] + 0.01, g1[1]))
        axs[0].annotate("2", (g2[0] + 0.01, g2[1]))
        axs[0].scatter(x1[0], y1[0], marker='o', color='red')
        axs[0].scatter(x2[0], y2[0], marker='o', color='blue')
        axs[0].scatter(x1[-1], y1[-1], marker='*', color='red')
        axs[0].scatter(x2[-1], y2[-1], marker='*', color='blue')
        axs[0].plot(x1, y1, color='red', label='A', marker='o', markersize=2)
        axs[0].plot(x2, y2, color='blue', label='D', marker='o', markersize=2)
        axs[0].set_xlim([-1, 1])
        axs[0].set_ylim([-1, 1])
        axs[0].legend()
        axs[1].plot(np.linspace(0, 1, total_steps + 1), p_t, linewidth=2)
        axs[1].set_xlabel('time (t)')
        axs[1].set_ylabel('belief (p_t)')
        axs[1].set_ylim([-0.1, 1])
        # plt.show()
    # plt.show()

        import pandas as pd

        df1 = pd.DataFrame(data={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'p': p_t, 'f': flag})
        # df1.to_csv(f'cons_multiple_trajs/cons_type_{p1_type}_{run}.csv')

    plt.show()