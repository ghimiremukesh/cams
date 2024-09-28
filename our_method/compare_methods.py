import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import product
import jax
import jax.numpy as jnp
from jax import jit
from utils_jax import running_cost, x_next, final_cost_function, point_dyn, make_pairs, final_cost_function_batch
from utils_jax import inst_cost_enum, cav_vex
from functools import partial
from tqdm import tqdm
import utils_jax
import time

# jax.config.update("jax_disable_jit", True)

EPSILON = 1e-6

bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 4),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-12., 12.]] * 4),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]

dt = 1

R1 = jnp.array([[0.05, 0], [0, 0.025]])
R2 = jnp.array([[0.05, 0], [0, 0.1]])


@jit
def optimal_utilities(x, p):
    # states and goals
    a1 = 1
    a2 = 0

    p_u1 = a1 * p + a2 * (1 - p)
    p_u2 = 1 - p_u1

    # posteriors
    pos_1 = a1 * p / p_u1

    pos_2 = (1 - a1) * p / p_u2

    # analytical solution to goal1
    u1, v1 = utils_jax.analytical_sol(x, 1, dt)
    u2, v2 = utils_jax.analytical_sol(x, 0, dt)

    x_next_1 = x_next(x, u1, v1, dt)
    x_next_2 = x_next(x, u2, v2, dt)

    # dist1 = jnp.linalg.norm(x_next_1[:2] - g1) ** 2 # p1 distance to goal 1
    # dist2 = jnp.linalg.norm(x_next_2[:2] - g2) ** 2 # p1 distance to goal2
    #
    # dist1_p2 = jnp.linalg.norm(x_next_1[4:6] - g1) ** 2 # p2 distance to goal 1
    # dist2_p2 = jnp.linalg.norm(x_next_2[4:6] - g2) ** 2 # p2 distance to goal 2

    final_cost_p1_1 = utils_jax.final_cost_function_p1(x_next_1, pos_1)
    final_cost_p1_2 = utils_jax.final_cost_function_p1(x_next_2, pos_2)

    final_cost_p2_1 = utils_jax.final_cost_function_p2(x_next_1, pos_1)
    final_cost_p2_2 = utils_jax.final_cost_function_p2(x_next_2, pos_2)

    expected_p1_utility = (p_u1 * (final_cost_p1_1 + dt * (0.05 * u1[0] ** 2 + 0.025 * u1[1] ** 2)) +
                           (p_u2) * (final_cost_p1_2 + dt * (0.05 * u2[0] ** 2 + 0.025 * u2[1] ** 2)))

    expected_p2_utility = (p_u1 * (final_cost_p2_1 + dt * (0.05 * v1[0] ** 2 + 0.1 * v1[1] ** 2)) +
                           p_u2 * (final_cost_p2_2 + dt * (0.05 * v2[0] ** 2 + 0.1 * v2[1] ** 2)))

    # what if we sum the utilities
    # p1 wants to minimize, so maximize -ve utility


    return expected_p1_utility, expected_p2_utility

@jit
def current_utility(params, x, p):
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

    x_next_1 = x_next(x, u1, v1, dt)
    x_next_2 = x_next(x, u2, v2, dt)

    final_cost_p1_1 = utils_jax.final_cost_function_p1(x_next_1, pos_1)
    final_cost_p1_2 = utils_jax.final_cost_function_p1(x_next_2, pos_2)

    final_cost_p2_1 = utils_jax.final_cost_function_p2(x_next_1, pos_1)
    final_cost_p2_2 = utils_jax.final_cost_function_p2(x_next_2, pos_2)


    expected_p1_utility = (p_u1 * (final_cost_p1_1 + dt * (0.05 * u1[0] ** 2 + 0.025 * u1[1] ** 2)) +
                           p_u2 * (final_cost_p1_2 + dt * (0.05 * u2[0] ** 2 + 0.025 * u2[1] ** 2)))

    expected_p2_utility = (p_u1 * (final_cost_p2_1 + dt * (0.05 * v1[0] ** 2 + 0.1 * v1[1] ** 2)) +
                           p_u2 * (final_cost_p2_2 + dt * (0.05 * v2[0] ** 2 + 0.1 * v2[1] ** 2)))


    return expected_p1_utility, expected_p2_utility
    # return (expected_p1_utility - expected_p2_utility), (expected_p2_utility - expected_p1_utility)

@jit
def dsgda_obj_final(params, x, p):
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

    x_next_1 = x_next(x, u1, v1, dt)
    x_next_2 = x_next(x, u2, v2, dt)

    val_1 = final_cost_function(x_next_1, pos_1)
    val_2 = final_cost_function(x_next_2, pos_2)

    ins_1 = dt * running_cost(u1, v1)

    ins_2 = dt * running_cost(u2, v2)

    p1_params = params[:6]
    z = params[6:12]
    p2_params = params[12:16]
    v = params[16:]

    regularizer_1 = (r1 / 2) * jnp.linalg.norm(p1_params - z) ** 2
    regularizer_2 = (r2 / 2) * jnp.linalg.norm(p2_params - v) ** 2

    objective = p_u1 * (val_1 + ins_1) + p_u2 * (val_2 + ins_2)  # + regularizer_1 - regularizer_2

    return objective.reshape(())


@jit
def value_final(params, x, p):
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

    x_next_1 = x_next(x, u1, v1, dt)
    x_next_2 = x_next(x, u2, v2, dt)

    val_1 = final_cost_function(x_next_1, pos_1)
    val_2 = final_cost_function(x_next_2, pos_2)

    ins_1 = dt * running_cost(u1, v1)

    ins_2 = dt * running_cost(u2, v2)

    objective = p_u1 * (val_1 + ins_1) + p_u2 * (val_2 + ins_2)

    return objective.reshape(())


# optim_method
@partial(jit, static_argnums=(3,))
def gradient_step(params, x, p, loss_fn):
    c = 1e-1
    alpha = 5e-2
    beta = 0.4
    mu = 0.5
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(params, x, p.reshape(-1, 1))
    p1_params = params[:, :6]
    z = params[:, 6:12]
    p2_params = params[:, 12:16]
    v = params[:, 16:]
    new_p1_params = (p1_params - c * grad[:, :6]).clip(min_bounds[:6], max_bounds[:6])
    new_params = jnp.concatenate((new_p1_params, z, p2_params, v), axis=1)
    grad = jax.vmap(jax.grad(loss_fn, argnums=0))(new_params, x, p.reshape(-1, 1))
    new_p2_params = (p2_params + alpha * grad[:, 12:16]).clip(min_bounds[6:], max_bounds[6:])
    new_z = z + beta * (new_p1_params - z)
    new_v = v + mu * (new_p2_params - v)
    new_params = jnp.concatenate((new_p1_params, new_z, new_p2_params, new_v), axis=1)
    norm1 = jnp.linalg.norm(new_z - new_p1_params, ord=1)
    norm2 = jnp.linalg.norm(new_v - new_p2_params, ord=1)

    return new_params, norm1, norm2


def solve_minimax(params, x, p, obj_fn, val_fn):
    curr_params = params
    iter = 0
    exp = 1
    # iters = 30000
    p1_params_list=[]
    with tqdm() as pbar:
        while  exp > 1e-5:
            new_params, norm1, norm2 = gradient_step(curr_params, x, p, obj_fn)
            curr_params = new_params
            p1_params = curr_params[:, :6]
            p2_params = curr_params[:, 12:16]
            p1_params_list.append(p1_params)
            # compute "exploitability"
            curr_util_1, curr_util_2 = current_utility(curr_params.reshape(-1, ), x.reshape(-1, ), p.reshape(-1, ))
            op_util_1, op_util_2 = optimal_utilities(x.reshape(-1, ), p.reshape(-1, ))
            # exp = (curr_util_1 - op_util_1 + curr_util_2 - op_util_2)/2
            if not iter % 20:
                exp = curr_util_1 - op_util_1#(curr_util_1 - op_util_1 + curr_util_2 - op_util_2)/2
                print(exp)
            pbar.update(1)
            iter += 1

    final_params = jnp.concatenate((p1_params, p2_params), axis=1)
    value = jax.vmap(val_fn)(final_params, x, p.reshape(-1, 1))

    return final_params, value, p1_params_list


def optim_method(states_, p):
    key = jax.random.PRNGKey(0)
    key5, key6 = jax.random.split(key, 2)
    params_u = jax.random.uniform(key5, (1, 4), minval=-1, maxval=1)
    params_up = jnp.array([[EPSILON, 1 - EPSILON]])
    params_d = jax.random.uniform(key6, (1, 4), minval=-1, maxval=1)

    params = jnp.concatenate((params_u, params_up, params_d), axis=1)

    params = jnp.repeat(params, repeats=2, axis=0).reshape(1, -1)

    final_params, value, p1_params_all = solve_minimax(params, states_, p, dsgda_obj_final, value_final)

    return final_params, value, p1_params_all


def enumeration_method(states_, p, n):
    NUM_PS = 101
    ps = jnp.linspace(0, 1, NUM_PS)
    index_at = 50  # hardcode the index of p to evaluate the v at.
    vs = []
    Us = []
    Ds = []
    ins_cost = inst_cost_enum(12, 12, 12, 12, R1, R2, n=n).reshape(-1, n * n, n * n)
    for p_each in tqdm(ps):
        next_states = point_dyn(states_, 12, 12, 12, 12, dt=dt, n=n)
        next_states_all = make_pairs(next_states[:, :4], next_states[:, 4:8], n * n)
        val_next = final_cost_function_batch(next_states_all, p_each).reshape(-1, n * n, n * n) + \
                   dt * ins_cost

        minmax_val = jnp.min(jnp.max(val_next, 2), 1)
        u = jnp.argmin(jnp.max(val_next, 2))
        d = jnp.argmax(val_next, 2).reshape(-1, )[u]
        Us.append(u)
        Ds.append(d)
        vs.append(minmax_val)

    val = cav_vex(vs, type='vex', num_ps=NUM_PS)

    return val.reshape(-1, )[index_at], Us[0].item(), Us[-1].item(), Ds[0].item(), Ds[-1].item()


class compute_value_and_actions():
    def __init__(self, method, disp=False, n=50, pos=None):
        self.method = method
        key = jax.random.PRNGKey(0)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        if pos is None:
            pos = jax.random.uniform(key1, (1, 4), minval=-0.5, maxval=0.5)
        else:
            pos = jnp.array(pos).reshape(1, -1)
        vel_1 = jax.random.uniform(key2, (1, 2), minval=0, maxval=0)
        vel_2 = jax.random.uniform(key3, (1, 2), minval=0, maxval=0)
        p = jnp.array([0.5])

        states_ = jnp.concatenate((pos[:, :2], vel_1, pos[:, 2:4], vel_2), axis=1)

        if self.method == 'optim':
            self.optim_param, self.optim_value, self.all_p1_params = optim_method(states_, p)
            if disp:
                print(f'Results from optimization')
                print(self.optim_param, self.optim_value)
        else:
            val_enum, u1_idx, u2_idx, d1_idx, d2_idx = enumeration_method(states_, p, n)
            uxs = jnp.linspace(-12, 12, n)
            uys = jnp.linspace(-12, 12, n)
            dxs = jnp.linspace(-12, 12, n)
            dys = jnp.linspace(-12, 12, n)
            us = list(product(uxs, uys))
            ds = list(product(dxs, dys))
            umap = {k: v for (k, v) in enumerate(us)}
            dmap = {k: v for (k, v) in enumerate(ds)}
            # self.sol = jnp.array(umap[u_idx].reshape(-1, ))
            if disp:
                print(f'Results from enumeration: value at p=0.5, and actions for lam = 0 and lam = 1 \n')
                print(val_enum)
                print(jnp.array(umap[u1_idx]).reshape(-1, ), jnp.array(umap[u2_idx]).reshape(-1, ))
                print(jnp.array(dmap[d1_idx]).reshape(-1, ), jnp.array(dmap[d2_idx]).reshape(-1, ))

    def get_params(self):
        return self.optim_param, self.all_p1_params

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    pos = jax.random.uniform(key1, (1, 4), minval=-0.5, maxval=0.5)
    vel_1 = jax.random.uniform(key2, (1, 2), minval=0, maxval=0)
    vel_2 = jax.random.uniform(key3, (1, 2), minval=0, maxval=0)
    p = jnp.array([0.5])

    states_ = jnp.concatenate((pos[:, :2], vel_1, pos[:, 2:4], vel_2), axis=1)

    time_optim_0 = time.time()
    optim_param, optim_value = optim_method(states_, p)

    print(optim_param, optim_value)
    time_optim_1 = time.time()
    print(f'Time taken: {time_optim_1 - time_optim_0:.2f}')
