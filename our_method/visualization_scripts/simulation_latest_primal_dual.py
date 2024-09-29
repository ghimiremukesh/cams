import os
import sys
# quick fix for import issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import random

import flax.core
import jax
import matplotlib.pyplot as plt
from flax.training import checkpoints
import numpy as np
import jax.numpy as jnp
from flax_picnn_dual import PICNN as dual , ModelConfig as dual_config
from flax_picnn import PICNN as primal, ModelConfig as primal_config
import optax
import utils_jax
from jax import jit
from tqdm import tqdm
from functools import partial
import matplotlib
from utils_jax import (final_cost_function_dual, running_cost, x_next, normalize_to_max_1d,
                       compute_bounds, final_cost_function)
from flax.core.nn import relu

matplotlib.use('TkAgg')

# jax.config.update("jax_disable_jit", True)

dual_model = dual(dual_config)
EPSILON = 1e-6
p_hat_bound = 1.4
# for GDA steps
primal_bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 4),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-12., 12.]] * 4),
))
min_bounds_primal = primal_bounds[:, 0]
max_bounds_primal = primal_bounds[:, 1]

dual_bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 6),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-p_hat_bound, p_hat_bound]] * 6),
    jnp.array([[-12., 12.]] * 6),
))
min_bounds_dual = dual_bounds[:, 0]
max_bounds_dual = dual_bounds[:, 1]



# more data
chkpnts = [f'../logs_dual_value/t_{i}/checkpoint_0/checkpoint' for i in range(1, 11)]

total_steps = 10


primal_chkpnts = [f'../logs_uncons/t_{i}/checkpoint_0/checkpoint' for i in range(1, 11)]



dt = 0.1

# primal game functions
@partial(jit, static_argnums=(3,))
def value_final(params, x, p, t):
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

    val_1 = final_cost_function(x_next_1, pos_1)
    val_2 = final_cost_function(x_next_2, pos_2)

    ins_1 = 0.1 * running_cost(u1, v1)

    ins_2 = 0.1 * running_cost(u2, v2)

    objective = p_u1 * (val_1 + ins_1) + p_u2 * (val_2 + ins_2)

    return objective.reshape(())


@partial(jit, static_argnums=(3,))
def value_inter(params, x, p, t):
    chkpt_idx = int(10 * t) - 1  # current model
    weights_dir = primal_chkpnts[chkpt_idx - 1]  # next model, we need next model in optimization
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=weights_dir, target=None)
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

    v_bound_next = compute_bounds(1 - t + dt, 12)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_1d(input_, v_bound_next, v_bound_next, v_bound_next, v_bound_next)
        return primal_model.apply(state_dict, rescaled_input)

    val_1 = apply_to_model(input_1)
    val_2 = apply_to_model(input_2)

    ins_cost_1 = dt * utils_jax.running_cost(u1, v1).reshape(-1, )
    ins_cost_2 = dt * utils_jax.running_cost(u2, v2).reshape(-1, )

    final_cost_1 = val_1 + ins_cost_1
    final_cost_2 = val_2 + ins_cost_2

    objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

    return objective.reshape(())


@partial(jit, static_argnums=(3, 4))
def gradient_step_primal(params, x, p, loss_fn, t):
    c = 1e-1
    alpha = 5e-2
    grad = jax.grad(loss_fn, argnums=0)(params, x, p, t)
    p1_params = params[:6]
    p2_params = params[6:]

    new_p1_params = (p1_params - c * grad[:6]).clip(min_bounds_primal[:6], max_bounds_primal[:6])
    new_params = jnp.concatenate((new_p1_params, p2_params))
    grad = jax.grad(loss_fn, argnums=0)(new_params, x, p, t)
    new_p2_params = (p2_params + alpha * grad[6:]).clip(min_bounds_primal[6:], max_bounds_primal[6:])
    new_params = jnp.concatenate((new_p1_params, new_p2_params))

    return new_params

def solve_minimax_primal(params, x, p, t):
    if t == dt:
        val_fn = value_final
    else:
        val_fn = value_inter

    curr_params = params
    iters = 30000
    iter = 0
    with tqdm(total=iters) as pbar:
        while iter <= iters:
            new_params = gradient_step_primal(curr_params, x, p, val_fn, t)
            curr_params = new_params
            p1_params = curr_params[:6]
            p2_params = curr_params[6:]
            pbar.update(1)
            iter += 1

    final_params = jnp.concatenate((p1_params, p2_params))

    return final_params


# dual value funcs
@partial(jit, static_argnums=(3, ))
def dual_value_final(params, x, p_hat, t):
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

    # regularization
    p2_params = params[:14]
    z = params[14:28]
    p1_params = params[28:34]
    v = params[34:]

    reg1 = (r1 / 2) * jnp.sum((p2_params - z) ** 2)
    reg2 = (r2 / 2) * jnp.sum((p1_params - v) ** 2)

    lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
    p_hat_penalty = 8 * relu(cons_phat)
    objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + p_hat_penalty + lam_penalty # + reg1 - reg2


    return objective.reshape(())


@partial(jit, static_argnums=(3, ))
def dual_value_inter(params, x, p_hat, t):
    chkpt_idx = int(10 * t) - 1  # current model
    weights_dir = chkpnts[chkpt_idx - 1]  # next model, we need next model in optimization
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=weights_dir, target=None)
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

    # l3 = jax.lax.cond(l1 + l2 < 1, l3_true, l3_false)
    # p_hat_3 = (p_hat - l1 * p_hat_1 - l2 * p_hat_2)/l3
    cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3)**2) + 1e-12)

    # l3 = 1 - l1 - l2 + EPSILON
    # p_hat_3 = (p_hat - l1 * p_hat_1 - l2 * p_hat_2)/l3
    # p_hat_3 = p_hat_3.clip(-p_hat_bound, p_hat_bound) # we should not clip


    x_next_1 = x_next(x, u1, v1)
    x_next_2 = x_next(x, u2, v2)
    x_next_3 = x_next(x, u3, v3)


    ins_cost_1 = dt * utils_jax.running_cost(u1, v1).reshape(-1, )
    ins_cost_2 = dt * utils_jax.running_cost(u2, v2).reshape(-1, )
    ins_cost_3 = dt * utils_jax.running_cost(u3, v3).reshape(-1, )


    input_1 = jnp.concat((x_next_1, p_hat_1 - ins_cost_1))
    input_2 = jnp.concat((x_next_2, p_hat_2 - ins_cost_2))
    input_3 = jnp.concat((x_next_3, p_hat_3 - ins_cost_3))


    v_bound_next = compute_bounds(1 - t + dt, 12)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_1d(input_, v_bound_next, v_bound_next, v_bound_next, v_bound_next)
        return dual_model.apply(state_dict, rescaled_input)

    val_1 = apply_to_model(input_1)
    val_2 = apply_to_model(input_2)
    val_3 = apply_to_model(input_3)

    # regularization
    p2_params = params[:14]
    z = params[14:28]
    p1_params = params[28:34]
    v = params[34:]

    reg1 = (r1 / 2) * jnp.sum((p2_params - z) ** 2)
    reg2 = (r2 / 2) * jnp.sum((p1_params - v) ** 2)


    lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
    p_hat_penalty = 8 * relu(cons_phat)
    objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + p_hat_penalty + lam_penalty #+ reg1 - reg2

    return objective.reshape(())

@partial(jit, static_argnums=(3, 4))
def gradient_step_dual(params, x, p, loss_fn, t):
    c = 5e-2 #1e-2
    alpha = 1e-2
    beta = 0.4
    mu = 0.5
    grad = jax.grad(loss_fn, argnums=0)(params, x, p, t)
    p2_params = params[:14]
    z = params[14:28]
    p1_params = params[28:34]
    v = params[34:]
    new_p2_params = (p2_params - c * grad[:14]).clip(min_bounds_dual[:14], max_bounds_dual[:14])
    new_params = jnp.concatenate((new_p2_params, z, p1_params, v))
    grad = jax.grad(loss_fn, argnums=0)(new_params, x, p, t)
    new_p1_params = (p1_params + alpha * grad[28:34]).clip(min_bounds_dual[14:], max_bounds_dual[14:])
    new_z = z + beta * (new_p2_params - z)
    new_v = v + mu * (new_p1_params - v)
    new_params = jnp.concatenate((new_p2_params, new_z, new_p1_params, new_v))
    norm1 = jnp.linalg.norm(new_z - new_p2_params, ord=1)
    norm2 = jnp.linalg.norm(new_v - new_p1_params, ord=1)

    return new_params, norm1, norm2


def solve_minimax_dual(params, x, p_hat, t):
    if t == dt:
        val_fn = dual_value_final
    else:
        val_fn = dual_value_inter

    curr_params = params
    iters = 200000
    iter = 0
    with tqdm(total=iters) as pbar:
        while iter <= iters:
            new_params, norm1, norm2 = gradient_step_dual(curr_params, x, p_hat, val_fn, t)
            curr_params = new_params
            p2_params = curr_params[:14]
            p1_params = curr_params[28:34]
            # if not iter % 1000:
            #     print(norm1)
            pbar.update(1)
            iter += 1

    final_params = jnp.concatenate((p2_params, p1_params))
    # value = jax.vmap(val_fn)(curr_params, x, p_hat)

    return final_params


if __name__ == '__main__':
    R1 = jnp.array([[0.05, 0], [0, 0.025]])
    R2 = jnp.array([[0.05, 0], [0, 0.1]])
    fig, axs = plt.subplots(3, 1)
    ys = [-0.5, -0.25, 0.25, 0.5]
    for run in range(len(ys)):  # number of simulations to run
        type_map = {0: 'Goal 2', 1: 'Goal 1'}
        p1_type = 1
        print(f'P1 type is: {type_map[p1_type]}')

        p_t = 0.5

        curr_x1 = jnp.array([[-0.5, ys[run], 0, 0]])
        curr_x2 = jnp.array([[0.5, ys[run], 0, 0]])
        states = []
        U = []
        D = []

        t = 1
        ts = np.arange(0, 1, dt)

        curr_state = jnp.hstack((curr_x1, curr_x2))


        # input to primal value network
        input_to_primal = jnp.hstack((curr_state, jnp.array([[p_t]]))).squeeze()

        val_fn = lambda x: primal_model.apply(primal_state, x)

        primal_value = val_fn(input_to_primal)
        primal_value_grad = jax.jacfwd(val_fn)(input_to_primal)

        dvdp = primal_value_grad[:, -1]

        # p_hat_2 = primal_value - p_t * dvdp
        # p_hat_1 = p_hat_2 + dvdp
        p_hat_1 = -0.02466
        p_hat_2 = -0.02466

        p_hat_t = jnp.array([p_hat_1, p_hat_2]).T.squeeze()

        # states.append(jnp.hstack((curr_state, p_hat_t.reshape(1, -1))))
        states.append(jnp.hstack((curr_state, jnp.array([[p_t]]))))

        print("belief is:", p_t)
        print("p_hat is:", p_hat_t)

        # a_max = 12  # maximum control bound for both players
        key = jax.random.PRNGKey(0)
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        while t >= dt:
            # primal optimization first
            params_u = jax.random.uniform(key1, (1, 4), minval=0, maxval=1)
            params_up = jnp.array([[EPSILON, 1 - EPSILON]])
            params_d = jax.random.uniform(key2, (1, 4), minval=0, maxval=1)
            params = jnp.concatenate((params_u, params_up, params_d), axis=1)

            params = solve_minimax_primal(params.reshape(-1, ), curr_state.reshape(-1, ), jnp.array([p_t]), t)
            a1 = params.reshape(-1, )[4]
            a2 = params.reshape(-1, )[5]

            p_u1 = a1 * p_t + a2 * (1 - p_t)
            p_u2 = 1 - p_u1

            # posteriors
            pos_1 = a1 * p_t / p_u1
            pos_2 = (1 - a1) * p_t / p_u2

            u1 = params[:2]
            u2 = params[2:4]

            if p1_type == 1:
                u_1_prob = a1
                u_2_prob = 1 - a1
            else:
                u_1_prob = a2
                u_2_prob = 1 - a2

            dist = [u_1_prob, u_2_prob]
            a_idx = [0, 1]
            action_idx = random.choices(a_idx, dist)[0]

            if action_idx == 0:
                p1_action = u1
                p_t = pos_1
            else:
                p1_action = u2
                p_t = pos_2


            # dual optimization second
            params_d = jax.random.uniform(key5, (1, 6), minval=-1, maxval=1)
            params_d_lam = jnp.array([[EPSILON, 1 - EPSILON]])
            params_d_phat = jax.random.uniform(key6, (1, 6), minval=-p_hat_bound, maxval=p_hat_bound)
            params_u = jax.random.uniform(key6, (1, 6), minval=-1, maxval=1)

            params = jnp.concatenate((params_d, params_d_lam, params_d_phat, params_u), axis=1)
            params = np.repeat(params, repeats=1, axis=0)

            params = jnp.repeat(params, repeats=2, axis=0).reshape(1, -1)  # repeat for dsgda

            params = solve_minimax_dual(params.reshape(-1, ), curr_state.reshape(-1, ), p_hat_t, t)
            # print(f'Parameters are: {params}')
            params = params.reshape(-1, )
            p2_params = params[:14]
            p1_params = params[14:]

            v1 = p2_params[:2]
            v2 = p2_params[2:4]
            v3 = p2_params[4:6]

            u1_dual = p1_params[0:2]
            u2_dual = p1_params[2:4]
            u3_dual = p1_params[4:6]

            # compute instantaneous losses


            l1 = p2_params[6]
            l2 = p2_params[7]
            if l1 + l2 < 1:
                l3 = 1 - l1 - l2
            else:
                l3 = 0.

            ins_cost_1 = dt * utils_jax.running_cost(u1_dual, v1).reshape(-1, )
            ins_cost_2 = dt * utils_jax.running_cost(u2_dual, v2).reshape(-1, )
            ins_cost_3 = dt * utils_jax.running_cost(u3_dual, v3).reshape(-1, )

            p_hat_1 = p2_params[8:10] - ins_cost_1
            p_hat_2 = p2_params[10:12] - ins_cost_2
            p_hat_3 = p2_params[12:14] - ins_cost_3


            print(f'At t = {1 - t:.2f}, P1 with type:\"{type_map[p1_type]}\" has the following choices: \n')
            print(f'P1 could take action {u1} with probability {u_1_prob:.2f} and update belief to {pos_1:.2f}')
            print(f'P1 could take action {u2} with probability {u_2_prob:.2f} and update belief to {pos_2:.2f}')

            print(f'At t = {1 - t:.2f}, P2 has the following choices: \n')
            print(f'P2 could take action {v1} with probability {l1:.2f} and update p_hat to {p_hat_1}')
            print(f'P2 could take action {v2} with probability {l2:.2f} and update p_hat to {p_hat_2}')
            print(f'P2 could take action {v3} with probability {l3:.2f} and update p_hat to {p_hat_3}')


            dist = jnp.array([l1, l2, l3])

            dist = dist/(dist.sum())  # tms

            a_idx = [0, 1, 2]

            action_idx = random.choices(a_idx, dist)[0]

            if action_idx == 0:
                p2_action = v1
                p_hat_t = p_hat_1
            elif action_idx == 1:
                p2_action = v2
                p_hat_t = p_hat_2
            else:
                p2_action = v3
                p_hat_t = p_hat_3

            print(f'P1 chooses: {p1_action}, and P2 chooses: {p2_action}. The belief is now {p_t:.2f}.'
                  f' The p_hat is now {p_hat_t}')

            U.append(p1_action)
            D.append(p2_action)

            curr_state = x_next(curr_state.reshape(-1, ), p1_action, p2_action)

            curr_state = curr_state.at[:2].set(curr_state[:2].clip(-1, 1))
            curr_state = curr_state.at[4:6].set(curr_state[4:6].clip(-1, 1))

            print(f'current state is {curr_state}')

            t = np.round(t - dt, 3)
            states.append(jnp.hstack((curr_state, p_t)))

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

        import pandas as pd
        df1 = pd.DataFrame(data={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'p': p_t})
        df1.to_csv(f'primal_dual_trajs/uncons_primal_dual_type_{p1_type}_{run}.csv')

    plt.show()
