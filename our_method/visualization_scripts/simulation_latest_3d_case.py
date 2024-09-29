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
from flax_picnn_3d import PICNN, ModelConfig
import optax
import utils_jax
from jax import jit
from tqdm import tqdm
from functools import partial
import matplotlib
from utils_jax import final_cost_function_3d, running_cost_3d, x_next_3d, normalize_to_max_3d, compute_bounds

import scipy.io as scio

matplotlib.use('TkAgg')


config = ModelConfig
model = PICNN(config)
EPSILON = 1e-6

# for GDA steps
bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 6),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-12., 12.]] * 6),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]

chkpnts = [f'../logs_3d_case/t_{i}/checkpoint_0/checkpoint' for i in range(1, 5)]
total_steps = 4

dt = 0.25


@partial(jit, static_argnums=(3,))
def value_final(params, x, p, t):
    u1 = params[:3]
    u2 = params[3:6]
    a1 = params[6]
    a2 = params[7]
    v1 = params[8:11]
    v2 = params[11:14]

    p_u1 = a1 * p + a2 * (1 - p)
    p_u2 = 1 - p_u1

    # posteriors
    pos_1 = a1 * p / p_u1

    pos_2 = (1 - a1) * p / p_u2

    x_next_1 = x_next_3d(x, u1, v1, dt=dt)
    x_next_2 = x_next_3d(x, u2, v2, dt=dt)

    val_1 = final_cost_function_3d(x_next_1, pos_1)
    val_2 = final_cost_function_3d(x_next_2, pos_2)

    ins_1 = dt * running_cost_3d(u1, v1)

    ins_2 = dt * running_cost_3d(u2, v2)

    objective = p_u1 * (val_1 + ins_1) + p_u2 * (val_2 + ins_2)

    return objective.reshape(())


@partial(jit, static_argnums=(3,))
def value_inter(params, x, p, t):
    chkpt_idx = int(4 * t) - 1  # current model
    weights_dir = chkpnts[chkpt_idx - 1]  # next model, we need next model in optimization
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=weights_dir, target=None)
    u1 = params[:3]
    u2 = params[3:6]
    a1 = params[6]
    a2 = params[7]
    v1 = params[8:11]
    v2 = params[11:14]

    p_u1 = a1 * p + a2 * (1 - p)
    p_u2 = 1 - p_u1

    # posteriors
    pos_1 = a1 * p / p_u1

    pos_2 = (1 - a1) * p / p_u2

    x_next_1 = x_next_3d(x, u1, v1, dt=dt)
    x_next_2 = x_next_3d(x, u2, v2, dt=dt)

    input_1 = jnp.concat((x_next_1, pos_1))
    input_2 = jnp.concat((x_next_2, pos_2))

    v_bound_next = compute_bounds(1 - t + dt, 12)

    # apply normalized input to the model and keep track for gradient computation
    @jit
    def apply_to_model(input_):
        rescaled_input = normalize_to_max_3d(input_, v_bound_next, v_bound_next, v_bound_next, v_bound_next, v_bound_next, v_bound_next)
        return model.apply(state_dict, rescaled_input)

    val_1 = apply_to_model(input_1)
    val_2 = apply_to_model(input_2)

    ins_cost_1 = dt * utils_jax.running_cost_3d(u1, v1).reshape(-1, )
    ins_cost_2 = dt * utils_jax.running_cost_3d(u2, v2).reshape(-1, )

    final_cost_1 = val_1 + ins_cost_1
    final_cost_2 = val_2 + ins_cost_2

    objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

    return objective.reshape(())


@partial(jit, static_argnums=(3, 4))
def gradient_step(params, x, p, loss_fn, t):
    c = 1e-1
    alpha = 5e-2
    grad = jax.grad(loss_fn, argnums=0)(params, x, p, t)
    p1_params = params[:8]
    p2_params = params[8:]

    new_p1_params = (p1_params - c * grad[:8]).clip(min_bounds[:8], max_bounds[:8])
    new_params = jnp.concatenate((new_p1_params, p2_params))
    grad = jax.grad(loss_fn, argnums=0)(new_params, x, p, t)
    new_p2_params = (p2_params + alpha * grad[8:]).clip(min_bounds[8:], max_bounds[8:])
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
    with tqdm(total=iters) as pbar:
        while iter <= iters:
            new_params = gradient_step(curr_params, x, p, val_fn, t)
            curr_params = new_params
            p1_params = curr_params[:8]
            p2_params = curr_params[8:]
            pbar.update(1)
            iter += 1

    final_params = jnp.concatenate((p1_params, p2_params))

    return final_params


if __name__ == '__main__':
    R1 = jnp.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.025]])
    R2 = jnp.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.1]])
    # fig, axs = plt.subplots(3, 1)
    ax = plt.axes(projection='3d')
    Us = []
    Ds = []
    U_GTs = []
    D_GTs = []
    x1s = [-0.5, -0.5, -0.5, -0.5, -0.5]
    y1s = [-0.5, -0.25, 0, 0.25, 0.5]
    z1s = [-0.5, -0.25, 0, 0.25, 0.5]
    x2s = [0.5, 0.5, 0.5, 0.5, 0.5]
    y2s = [-0.5, -0.25, 0, 0.25, 0.5]
    z2s = [-0.5, -0.25, 0, 0.25, 0.5]
    for run in range(len(x1s)):  # number of simulations to run
        type_map = {0: 'Goal 2', 1: 'Goal 1'}
        p1_type = 0
        # print(f'P1 type is: {type_map[p1_type]}')

        p_t = 0.5

        # x1 = np.random.uniform(-1, 1, 1).item()
        # y1 = np.random.uniform(-1, 1, 1).item()
        # z1 = np.random.uniform(-1, 1, 1).item()
        # x2 = np.random.uniform(-1, 1, 1).item()
        # y2 = np.random.uniform(-1, 1, 1).item()
        # z2 = np.random.uniform(-1, 1, 1).item()

        x1 = x1s[run]
        y1 = y1s[run]
        z1 = z1s[run]
        x2 = x2s[run]
        y2 = y2s[run]
        z2 = z2s[run]
        curr_x1 = jnp.array([[x1, y1, z1, 0, 0, 0]])
        curr_x2 = jnp.array([[x2, y2, z2, 0, 0, 0]])
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
            params_u = jax.random.uniform(key1, (1, 6), minval=0, maxval=1)
            params_up = jnp.array([[EPSILON, 1 - EPSILON]])
            params_d = jax.random.uniform(key2, (1, 6), minval=0, maxval=1)
            params = jnp.concatenate((params_u, params_up, params_d), axis=1)

            params = solve_minimax(params.reshape(-1, ), curr_state.reshape(-1, ), jnp.array([p_t]), t)
            # print(f'Parameters are: {params}')
            a1 = params.reshape(-1, )[6]
            a2 = params.reshape(-1, )[7]

            p_u1 = a1 * p_t + a2 * (1 - p_t)
            p_u2 = 1 - p_u1

            # posteriors
            pos_1 = a1 * p_t / p_u1
            pos_2 = (1 - a1) * p_t / p_u2

            u1 = params[:3]
            u2 = params[3:6]
            v1 = params[8:11]
            v2 = params[11:14]

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

            # print(f'P1 chooses: {p1_action}, and P2 chooses: {p2_action}. The belief is now {p_t:.2f}')

            # U.append(p1_action)
            # D.append(p2_action)
            # append action with probs
            U.append(np.concatenate((u1, u2, u_1_prob.reshape(-1, ), u_2_prob.reshape(-1, ))))

            curr_state = x_next_3d(curr_state.reshape(-1, ), p1_action, p2_action, dt=dt)

            curr_state = curr_state.at[:3].set(curr_state[:3].clip(-1, 1))
            curr_state = curr_state.at[6:9].set(curr_state[6:9].clip(-1, 1))

            # print(f'current state is {curr_state}')

            t = np.round(t - dt, 3)
            states.append(jnp.hstack((curr_state, jnp.array([p_t]))))

        # print(f'final state is: {curr_state}')
        times = np.arange(0, 1.1, dt)
        g1 = np.array([0, 0, 1])
        g2 = np.array([0, 0, -1])

        states = np.vstack(states)
        x1 = states[:, 0]
        y1 = states[:, 1]
        z1 = states[:, 2]
        x2 = states[:, 6]
        y2 = states[:, 7]
        z2 = states[:, 8]
        p_t = states[:, -1]

        U = np.vstack(U)
        # D = np.vstack(D)

        import pandas as pd
        data = {'x1': x1, 'y1': y1, 'z1': z1, 'x2': x2, 'y2': y2, 'z2': z2, 'p': p_t}
        df = pd.DataFrame(data)
        df.to_csv(f'3d_trajs_type_{p1_type}_{run}.csv')

        ax.plot3D(x1, y1, z1)
        ax.plot3D(x2, y2, z2)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        if p1_type == 0:
            ax.scatter3D(g1[0], g1[1], g1[2], marker='o', facecolor='none', edgecolor='magenta')
            ax.scatter3D(g2[0], g2[1], g2[2], marker='o', facecolor='magenta', edgecolor='magenta')
        else:
            ax.scatter3D(g1[0], g1[1], g1[2], marker='o', facecolor='magenta', edgecolor='magenta')
            ax.scatter3D(g2[0], g2[1], g2[2], marker='o', facecolor='none', edgecolor='magenta')


        ax.scatter3D(x1[0], y1[0], z1[0], marker='o', color='red')
        ax.scatter3D(x2[0], y2[0], z2[0], marker='o', color='blue')
        ax.scatter3D(x1[-1], y1[-1], z1[-1], marker='*', color='red')
        ax.scatter3D(x2[-1], y2[-1], z2[-1], marker='*', color='blue')

    plt.show()


