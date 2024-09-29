import os
import sys
# quick fix for import issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
from utils_jax import final_cost_function, running_cost, x_next, normalize_to_max_1d, compute_bounds

import scipy.io as scio

matplotlib.use('TkAgg')

# jax.config.update("jax_disable_jit", True)

config = ModelConfig
model = PICNN(config)
EPSILON = 1e-6

# for GDA steps
bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 4),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-12., 12.]] * 4),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]

chkpnts = [f'../logs_for_cfr/t_{i}/checkpoint_0/checkpoint' for i in range(1, 5)]
total_steps = 4

dt = 0.25


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

    x_next_1 = x_next(x, u1, v1, dt=dt)
    x_next_2 = x_next(x, u2, v2, dt=dt)

    val_1 = final_cost_function(x_next_1, pos_1)
    val_2 = final_cost_function(x_next_2, pos_2)

    ins_1 = dt * running_cost(u1, v1)

    ins_2 = dt * running_cost(u2, v2)

    objective = p_u1 * (val_1 + ins_1) + p_u2 * (val_2 + ins_2)

    return objective.reshape(())


@partial(jit, static_argnums=(3,))
def value_inter(params, x, p, t):
    chkpt_idx = int(4 * t) - 1  # current model
    weights_dir = chkpnts[chkpt_idx - 1]  # next model, we need next model in optimization
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

    x_next_1 = x_next(x, u1, v1, dt=dt)
    x_next_2 = x_next(x, u2, v2, dt=dt)

    input_1 = jnp.concat((x_next_1, pos_1))
    input_2 = jnp.concat((x_next_2, pos_2))

    v_bound_next = compute_bounds(1 - t + dt, 12)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_1d(input_, v_bound_next, v_bound_next, v_bound_next, v_bound_next)
        return model.apply(state_dict, rescaled_input)

    val_1 = apply_to_model(input_1)
    val_2 = apply_to_model(input_2)

    ins_cost_1 = dt * utils_jax.running_cost(u1, v1).reshape(-1, )
    ins_cost_2 = dt * utils_jax.running_cost(u2, v2).reshape(-1, )

    final_cost_1 = val_1 + ins_cost_1
    final_cost_2 = val_2 + ins_cost_2

    objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

    return objective.reshape(())


@partial(jit, static_argnums=(3, 4))
def gradient_step(params, x, p, loss_fn, t):
    c = 1e-1
    alpha = 5e-2
    grad = jax.grad(loss_fn, argnums=0)(params, x, p, t)
    p1_params = params[:6]
    p2_params = params[6:]

    new_p1_params = (p1_params - c * grad[:6]).clip(min_bounds[:6], max_bounds[:6])
    new_params = jnp.concatenate((new_p1_params, p2_params))
    grad = jax.grad(loss_fn, argnums=0)(new_params, x, p, t)
    new_p2_params = (p2_params + alpha * grad[6:]).clip(min_bounds[6:], max_bounds[6:])
    new_params = jnp.concatenate((new_p1_params, new_p2_params))

    return new_params


def solve_minimax(params, x, p, t):
    if t == dt:
        val_fn = value_final
    else:
        val_fn = value_inter

    curr_params = params
    iters = 30000
    iter = 0
    while iter <= iters:
        new_params = gradient_step(curr_params, x, p, val_fn, t)
        curr_params = new_params
        p1_params = curr_params[:6]
        p2_params = curr_params[6:]
        # pbar.update(1)
        iter += 1

    final_params = jnp.concatenate((p1_params, p2_params))

    return final_params


if __name__ == '__main__':
    R1 = jnp.array([[0.05, 0], [0, 0.025]])
    R2 = jnp.array([[0.05, 0], [0, 0.1]])
    fig, axs = plt.subplots(3, 1)
    Us = []
    Ds = []
    U_GTs = []
    D_GTs = []
    for _ in tqdm(range(100)):  # number of simulations to run
        type_map = {0: 'Goal 2', 1: 'Goal 1'}
        p1_type = 1
        # print(f'P1 type is: {type_map[p1_type]}')

        p_t = 0.5

        x1 = np.random.uniform(-1, 1, 1).item()
        y1 = np.random.uniform(-1, 1, 1).item()
        x2 = np.random.uniform(-1, 1, 1).item()
        y2 = np.random.uniform(-1, 1, 1).item()
        # x1 = -0.5
        # y1 = 0
        # x2 = 0.5
        # y2 = 0
        curr_x1 = jnp.array([[x1, y1, 0, 0]])
        curr_x2 = jnp.array([[x2, y2, 0, 0]])
        states = []
        U = []
        D = []

        l2_norm = []

        t = 1
        ts = np.arange(0, 1, dt)

        curr_state = jnp.hstack((curr_x1, curr_x2))
        states.append(jnp.hstack((curr_state, jnp.array([[p_t]]))))

        # a_max = 12  # maximum control bound for both players
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        while t >= dt:
            params_u = jax.random.uniform(key1, (1, 4), minval=0, maxval=1)
            params_up = jnp.array([[EPSILON, 1 - EPSILON]])
            params_d = jax.random.uniform(key2, (1, 4), minval=0, maxval=1)
            params = jnp.concatenate((params_u, params_up, params_d), axis=1)

            params = solve_minimax(params.reshape(-1, ), curr_state.reshape(-1, ), jnp.array([p_t]), t)
            # print(f'Parameters are: {params}')
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

            # print(f'At t = {1 - t:.2f}, P1 with type:\"{type_map[p1_type]}\" has the following choices: \n')
            # print(f'P1 could take action {u1} with probability {u_1_prob:.2f} and update belief to {pos_1:.2f}')
            # print(f'P1 could take action {u2} with probability {u_2_prob:.2f} and update belief to {pos_2:.2f}')

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

            # print(f'P1 chooses: {p1_action}, and P2 chooses: {p2_action}. The belief is now {p_t:.2f}')

            # U.append(p1_action)
            # D.append(p2_action)
            # append action with probs
            U.append(np.concatenate((u1, u2, u_1_prob.reshape(-1, ), u_2_prob.reshape(-1, ))))

            curr_state = x_next(curr_state.reshape(-1, ), p1_action, p2_action, dt=dt)

            # print(f'current state is {curr_state}')

            t = np.round(t - dt, 3)
            states.append(jnp.hstack((curr_state, jnp.array([p_t]))))

        # print(f'final state is: {curr_state}')
        times = np.arange(0, 1.1, dt)
        g1 = np.array([0, 1])
        g2 = np.array([0, -1])

        states = np.vstack(states)
        x1 = states[:, 0]
        y1 = states[:, 1]
        x2 = states[:, 4]
        y2 = states[:, 5]
        p_t = states[:, -1]

        U = np.vstack(U)
        # D = np.vstack(D)

        import pandas as pd
        data = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        df = pd.DataFrame(data)
        df.to_csv(f'trajectory_type_{p1_type}.csv')

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

        #### for comparison
        Us.append(U)
        # Ds.append(D)

        # gt solution
        tau = dt
        A = jnp.eye(4) + jnp.array([[0, 0, tau, 0], [0, 0, 0, tau], [0, 0, 0, 0], [0, 0, 0, 0]])
        B = jnp.array([[0.5 * tau ** 2, 0], [0, 0.5 * tau ** 2], [tau, 0], [0, tau]])
        Qf = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        Q = jnp.zeros((4, 4))
        dt = dt
        N = 4
        R1 = jnp.array([[0.05, 0], [0, 0.025]]) * tau
        R2 = jnp.array([[0.05, 0], [0, 0.1]]) * tau

        K1 = utils_jax.discrete_lqr(A, B, Q, R1, Qf, N)
        K2 = utils_jax.discrete_lqr(A, B, Q, R2, Qf, N)

        # states_ = jnp.array([-0.5, 0, 0, 0, 0.5, 0, 0, 0])
        p = 0.5
        # change index for different time steps

        target = p1_type

        trajs = []
        U_GT = []
        D_GT = []
        # trajs.append(states_)
        N = 4
        for i in range(total_steps, 0, -1):
            states_ = states[-i-1]
            if i <= 2:
                p = 0 if target == 0 else 1
            x1 = states_[:4]
            x2 = states_[4:8]
            goal = jnp.array([0, 2 * p - 1, 0, 0]) * jnp.ones_like(p)
            u = -K1[-i] @ (x1 - goal).T
            v = -K2[-i] @ (x2 - goal).T
            # states_ = utils_jax.x_next(states_, u, v)
            U_GT.append(u)
            D_GT.append(v)
            # trajs.append(states_)

        states = jnp.vstack(trajs)
        x1 = states[:, 0]
        y1 = states[:, 1]
        x2 = states[:, 4]
        y2 = states[:, 5]
        
        axs[0].plot(x1, y1, '--', color='orange', marker='o', markersize=2)
        axs[0].plot(x2, y2, '--', color='teal', marker='o', markersize=2)
        U_GTs.append(jnp.vstack(U_GT))
        D_GTs.append(jnp.vstack(D_GT))

    data = {'U': Us, 'D': Ds, 'U_GT': jnp.vstack(U_GTs), 'D_GT': jnp.vstack(D_GTs)}
    scio.savemat(f'distance_to_gt_type{p1_type}_100.mat', data)


    plt.show()

