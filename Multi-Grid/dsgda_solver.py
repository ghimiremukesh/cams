import copy
from functools import partial

import jax
from utils import util_funcs
import jax.numpy as jnp
from tqdm import tqdm
import pdb

EPSILON = 1e-6

bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 4),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-12., 12.]] * 4),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]


def dsgda_solver(val_model, model_params, coords, t, tau, init_policies=None, iters=120000):
    # value function
    @jax.jit
    def value_fn(params, X):
        u1 = params[:2]
        u2 = params[2:4]
        a1 = params[4]
        a2 = params[5]
        v1 = params[6:8]
        v2 = params[8:10]

        x = X[:8]
        p = X[8:]  # belief

        p_u1 = a1 * p + a2 * (1 - p)
        p_u2 = 1 - p_u1

        # posteriors
        pos_1 = a1 * p / p_u1

        pos_2 = (1 - a1) * p / p_u2

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=tau)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=tau)

        input_1 = jnp.concat((x_next_1, pos_1.reshape(-1, )))
        input_2 = jnp.concat((x_next_2, pos_2.reshape(-1, )))

        v_bound_next = util_funcs.compute_bounds(t + tau, 12)
        # apply normalized input to the model and keep track for gradient computation
        @jax.jit
        def apply_to_model(input_):
            rescaled_input = util_funcs.normalize_to_max_1d(input_, v_bound_next, v_bound_next, v_bound_next, v_bound_next)
            return val_model.apply(model_params, rescaled_input)

        val_1 = apply_to_model(input_1)
        val_2 = apply_to_model(input_2)

        ins_cost_1 = tau * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = tau * util_funcs.running_cost(u2, v2).reshape(-1, )

        final_cost_1 = val_1 + ins_cost_1
        final_cost_2 = val_2 + ins_cost_2

        objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

        return objective.reshape(())

    @jax.jit
    def gradient_step(params, x):
        c = 1e-1
        alpha = 5e-2
        grad = jax.vmap(jax.grad(value_fn, argnums=0))(params, x)
        p1_params = params[:, :6]
        p2_params = params[:, 6:10]
        new_p1_params = (p1_params - c * grad[:, :6]).clip(min_bounds[:6], max_bounds[:6])
        new_params = jnp.concatenate((new_p1_params, p2_params), axis=1)
        grad = jax.vmap(jax.grad(value_fn, argnums=0))(new_params, x)
        new_p2_params = (p2_params + alpha * grad[:, 6:]).clip(min_bounds[6:], max_bounds[6:])
        new_params = jnp.concatenate((new_p1_params, new_p2_params), axis=1)

        return new_params

    key = jax.random.PRNGKey(1)
    key5, key6 = jax.random.split(key, 2)
    num_points = coords.shape[0]

    if init_policies is not None:
        curr_params = jnp.array(init_policies)
    else:
        params_u = jax.random.uniform(key5, (num_points, 4), minval=-1, maxval=1)
        params_up = jnp.array([[EPSILON, 1 - EPSILON]])
        params_d = jax.random.uniform(key6, (num_points, 4), minval=-1, maxval=1)

        params_up = jnp.repeat(params_up, num_points, axis=0)

        params = jnp.concatenate((params_u, params_up, params_d), axis=1)
        curr_params = params

    iter = 0
    # p1_params = []
    # with tqdm(total=iters, position=0) as pbar1:
    while iter <= iters:
        new_params = gradient_step(curr_params, coords)
        curr_params = new_params
        # p1_params.append(curr_params)
        # pbar1.update(1)
        iter += 1

    value = jax.vmap(value_fn)(curr_params, coords)
    policies = curr_params

    return value, policies


def dsgda_solver_final(coords, tau, init_policies=None, iters=120000):
    @jax.jit
    def value_fn(params, X):
        u1 = params[:2]
        u2 = params[2:4]
        a1 = params[4]
        a2 = params[5]
        v1 = params[6:8]
        v2 = params[8:10]

        x = X[:8]
        p = X[8:]  # belief

        p_u1 = a1 * p + a2 * (1 - p)
        p_u2 = 1 - p_u1

        # posteriors
        pos_1 = a1 * p / p_u1
        pos_2 = (1 - a1) * p / p_u2

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=tau)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=tau)


        val_1 = util_funcs.final_cost_function(x_next_1, pos_1)
        val_2 = util_funcs.final_cost_function(x_next_2, pos_2)

        ins_cost_1 = tau * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = tau * util_funcs.running_cost(u2, v2).reshape(-1, )

        final_cost_1 = val_1 + ins_cost_1
        final_cost_2 = val_2 + ins_cost_2

        objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

        return objective.reshape(())

    @jax.jit
    def gradient_step(params, x):
        c = 1e-1
        alpha = 5e-2
        grad = jax.vmap(jax.grad(value_fn, argnums=0))(params, x)
        p1_params = params[:, :6]
        p2_params = params[:, 6:10]
        new_p1_params = (p1_params - c * grad[:, :6]).clip(min_bounds[:6], max_bounds[:6])
        new_params = jnp.concatenate((new_p1_params, p2_params), axis=1)
        grad = jax.vmap(jax.grad(value_fn, argnums=0))(new_params, x)
        new_p2_params = (p2_params + alpha * grad[:, 6:]).clip(min_bounds[6:], max_bounds[6:])
        new_params = jnp.concatenate((new_p1_params, new_p2_params), axis=1)

        return new_params

    key = jax.random.PRNGKey(1)
    key5, key6 = jax.random.split(key, 2)
    num_points = coords.shape[0]

    if init_policies is not None:
        curr_params = jnp.array(init_policies)
    else:
        params_u = jax.random.uniform(key5, (num_points, 4), minval=-1, maxval=1)
        params_up = jnp.array([[EPSILON, 1 - EPSILON]])
        params_d = jax.random.uniform(key6, (num_points, 4), minval=-1, maxval=1)

        params_up = jnp.repeat(params_up, num_points, axis=0)

        params = jnp.concatenate((params_u, params_up, params_d), axis=1)
        curr_params = params


    iter = 0
    # p1_params = []
    # with tqdm(total=iters, position=0) as pbar1:
    while iter <= iters:
        new_params = gradient_step(curr_params, coords)
        curr_params = new_params
        # p1_params.append(curr_params)
        # pbar1.update(1)
        iter += 1

    value = jax.vmap(value_fn)(curr_params, coords)
    policies = curr_params

    return value, policies

def coarse_solvers(aug_val_fn, coarse_val_fn, coords, t, H, h, iters=50000):
    # first solve the O_H(V_H + e_H)_t+1 and then O_H(V_H)_t+1
    @jax.jit
    def value_fn_aug(params, X):
        u1 = params[:2]
        u2 = params[2:4]
        a1 = params[4]
        a2 = params[5]
        v1 = params[6:8]
        v2 = params[8:10]

        x = X[:8]
        p = X[8:]  # belief

        p_u1 = a1 * p + a2 * (1 - p)
        p_u2 = 1 - p_u1

        # posteriors
        pos_1 = a1 * p / p_u1

        pos_2 = (1 - a1) * p / p_u2

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=H)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=H)

        input_1 = jnp.concat((x_next_1, pos_1.reshape(-1, )))
        input_2 = jnp.concat((x_next_2, pos_2.reshape(-1, )))

        v_bound_next_1 = util_funcs.compute_bounds(t + H, 12)
        v_bound_next_2 = util_funcs.compute_bounds(t + H + h, 12)
        # apply normalized input to the model and keep track for gradient computation
        @jax.jit
        def apply_to_model(input_):
            rescaled_input_1 = util_funcs.normalize_to_max_1d(input_, v_bound_next_1, v_bound_next_1,
                                                              v_bound_next_1, v_bound_next_1)
            rescaled_input_2 = util_funcs.normalize_to_max_1d(input_, v_bound_next_2, v_bound_next_2,
                                                              v_bound_next_2, v_bound_next_2)
            return aug_val_fn(rescaled_input_1, rescaled_input_2)

        val_1 = apply_to_model(input_1)
        val_2 = apply_to_model(input_2)

        ins_cost_1 = H * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = H * util_funcs.running_cost(u2, v2).reshape(-1, )

        final_cost_1 = val_1 + ins_cost_1
        final_cost_2 = val_2 + ins_cost_2

        objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

        return objective.reshape(())

    @jax.jit
    def value_fn_coarse(params, X):
        u1 = params[:2]
        u2 = params[2:4]
        a1 = params[4]
        a2 = params[5]
        v1 = params[6:8]
        v2 = params[8:10]

        x = X[:8]
        p = X[8:]  # belief

        p_u1 = a1 * p + a2 * (1 - p)
        p_u2 = 1 - p_u1

        # posteriors
        pos_1 = a1 * p / p_u1

        pos_2 = (1 - a1) * p / p_u2

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=H)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=H)

        input_1 = jnp.concat((x_next_1, pos_1.reshape(-1, )))
        input_2 = jnp.concat((x_next_2, pos_2.reshape(-1, )))

        v_bound_next_1 = util_funcs.compute_bounds(t + H, 12)
        v_bound_next_2 = util_funcs.compute_bounds(t + H + h, 12)
        # apply normalized input to the model and keep track for gradient computation
        @jax.jit
        def apply_to_model(input_):
            rescaled_input_1 = util_funcs.normalize_to_max_1d(input_, v_bound_next_1, v_bound_next_1,
                                                              v_bound_next_1, v_bound_next_1)
            rescaled_input_2 = util_funcs.normalize_to_max_1d(input_, v_bound_next_2, v_bound_next_2,
                                                              v_bound_next_2, v_bound_next_2)
            return coarse_val_fn(rescaled_input_1, rescaled_input_2)

        val_1 = apply_to_model(input_1)
        val_2 = apply_to_model(input_2)

        ins_cost_1 = H * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = H * util_funcs.running_cost(u2, v2).reshape(-1, )

        final_cost_1 = val_1 + ins_cost_1
        final_cost_2 = val_2 + ins_cost_2

        objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

        return objective.reshape(())


    @partial(jax.jit, static_argnums=(2, ))
    def gradient_step(params, x, val_fn):
        c = 1e-1
        alpha = 5e-2
        grad = jax.vmap(jax.grad(val_fn, argnums=0))(params, x)
        p1_params = params[:, :6]
        p2_params = params[:, 6:10]
        new_p1_params = (p1_params - c * grad[:, :6]).clip(min_bounds[:6], max_bounds[:6])
        new_params = jnp.concatenate((new_p1_params, p2_params), axis=1)
        grad = jax.vmap(jax.grad(val_fn, argnums=0))(new_params, x)
        new_p2_params = (p2_params + alpha * grad[:, 6:]).clip(min_bounds[6:], max_bounds[6:])
        new_params = jnp.concatenate((new_p1_params, new_p2_params), axis=1)

        return new_params

    key = jax.random.PRNGKey(1)
    key5, key6 = jax.random.split(key, 2)
    num_points = coords.shape[0]
    params_u = jax.random.uniform(key5, (num_points, 4), minval=-1, maxval=1)
    params_up = jnp.array([[EPSILON, 1 - EPSILON]])
    params_d = jax.random.uniform(key6, (num_points, 4), minval=-1, maxval=1)

    params_up = jnp.repeat(params_up, num_points, axis=0)

    params = jnp.concatenate((params_u, params_up, params_d), axis=1)
    params_copy = copy.deepcopy(params)

    # run dsgda for first value
    curr_params = params
    iter = 0
    # p1_params = []
    with tqdm(total=iters, position=0) as pbar1:
        while iter <= iters:
            new_params = gradient_step(curr_params, coords, value_fn_aug)
            curr_params = new_params
            # p1_params.append(curr_params)
            pbar1.update(1)
            iter += 1

    value_1 = jax.vmap(value_fn_aug)(curr_params, coords)

    # run dsgda for second value
    curr_params = params_copy
    iter = 0
    # p1_params = []
    with tqdm(total=iters, position=0) as pbar1:
        while iter <= iters:
            new_params = gradient_step(curr_params, coords, value_fn_coarse)
            curr_params = new_params
            # p1_params.append(curr_params)
            pbar1.update(1)
            iter += 1

    value_2 = jax.vmap(value_fn_coarse)(curr_params, coords)


    return value_1 - value_2

def coarse_solvers_adaptive(aug_val_fn, coarse_val_fn, coords, t, H, h, init_policies=None, iters=50000):
    # first solve the O_H(V_H + e_H)_t+1 and then O_H(V_H)_t+1
    @jax.jit
    def value_fn_aug(params, X):
        u1 = params[:2]
        u2 = params[2:4]
        a1 = params[4]
        a2 = params[5]
        v1 = params[6:8]
        v2 = params[8:10]

        x = X[:8]
        p = X[8:]  # belief

        p_u1 = a1 * p + a2 * (1 - p)
        p_u2 = 1 - p_u1

        # posteriors
        pos_1 = a1 * p / p_u1

        pos_2 = (1 - a1) * p / p_u2

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=H)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=H)

        input_1 = jnp.concat((x_next_1, pos_1.reshape(-1, )))
        input_2 = jnp.concat((x_next_2, pos_2.reshape(-1, )))

        v_bound_next_1 = util_funcs.compute_bounds(t + H, 12)
        v_bound_next_2 = util_funcs.compute_bounds(t + H + h, 12)
        # apply normalized input to the model and keep track for gradient computation
        @jax.jit
        def apply_to_model(input_):
            rescaled_input_1 = util_funcs.normalize_to_max_1d(input_, v_bound_next_1, v_bound_next_1,
                                                              v_bound_next_1, v_bound_next_1)
            rescaled_input_2 = util_funcs.normalize_to_max_1d(input_, v_bound_next_2, v_bound_next_2,
                                                              v_bound_next_2, v_bound_next_2)
            return aug_val_fn(rescaled_input_1, rescaled_input_2)

        val_1 = apply_to_model(input_1)
        val_2 = apply_to_model(input_2)

        ins_cost_1 = H * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = H * util_funcs.running_cost(u2, v2).reshape(-1, )

        final_cost_1 = val_1 + ins_cost_1
        final_cost_2 = val_2 + ins_cost_2

        objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

        return objective.reshape(())

    @jax.jit
    def value_fn_coarse(params, X):
        u1 = params[:2]
        u2 = params[2:4]
        a1 = params[4]
        a2 = params[5]
        v1 = params[6:8]
        v2 = params[8:10]

        x = X[:8]
        p = X[8:]  # belief

        p_u1 = a1 * p + a2 * (1 - p)
        p_u2 = 1 - p_u1

        # posteriors
        pos_1 = a1 * p / p_u1

        pos_2 = (1 - a1) * p / p_u2

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=H)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=H)

        input_1 = jnp.concat((x_next_1, pos_1.reshape(-1, )))
        input_2 = jnp.concat((x_next_2, pos_2.reshape(-1, )))

        v_bound_next_1 = util_funcs.compute_bounds(t + H, 12)
        v_bound_next_2 = util_funcs.compute_bounds(t + H + h, 12)
        # apply normalized input to the model and keep track for gradient computation
        @jax.jit
        def apply_to_model(input_):
            rescaled_input_1 = util_funcs.normalize_to_max_1d(input_, v_bound_next_1, v_bound_next_1,
                                                              v_bound_next_1, v_bound_next_1)
            rescaled_input_2 = util_funcs.normalize_to_max_1d(input_, v_bound_next_2, v_bound_next_2,
                                                              v_bound_next_2, v_bound_next_2)
            return coarse_val_fn(rescaled_input_1, rescaled_input_2)

        val_1 = apply_to_model(input_1)
        val_2 = apply_to_model(input_2)

        ins_cost_1 = H * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = H * util_funcs.running_cost(u2, v2).reshape(-1, )

        final_cost_1 = val_1 + ins_cost_1
        final_cost_2 = val_2 + ins_cost_2

        objective = p_u1 * final_cost_1 + p_u2 * final_cost_2

        return objective.reshape(())


    @partial(jax.jit, static_argnums=(2, ))
    def gradient_step(params, x, val_fn):
        c = 1e-1
        alpha = 5e-2
        grad = jax.vmap(jax.grad(val_fn, argnums=0))(params, x)
        p1_params = params[:, :6]
        p2_params = params[:, 6:10]
        new_p1_params = (p1_params - c * grad[:, :6]).clip(min_bounds[:6], max_bounds[:6])
        new_params = jnp.concatenate((new_p1_params, p2_params), axis=1)
        grad = jax.vmap(jax.grad(val_fn, argnums=0))(new_params, x)
        new_p2_params = (p2_params + alpha * grad[:, 6:]).clip(min_bounds[6:], max_bounds[6:])
        new_params = jnp.concatenate((new_p1_params, new_p2_params), axis=1)

        return new_params

    key = jax.random.PRNGKey(1)
    key5, key6 = jax.random.split(key, 2)
    num_points = coords.shape[0]

    if init_policies is not None:
        params = jnp.array(init_policies)
    else:
        params_u = jax.random.uniform(key5, (num_points, 4), minval=-1, maxval=1)
        params_up = jnp.array([[EPSILON, 1 - EPSILON]])
        params_d = jax.random.uniform(key6, (num_points, 4), minval=-1, maxval=1)

        params_up = jnp.repeat(params_up, num_points, axis=0)

        params = jnp.concatenate((params_u, params_up, params_d), axis=1)

    params_copy = copy.deepcopy(params)

    # run dsgda for first value
    curr_params = params
    iter = 0
    # p1_params = []
    norm = float('inf')
    count = 0
    with tqdm(total=iters, position=0) as pbar1:
        while norm > 1e-3 and count < 10 and iter <= iters:
            new_params = gradient_step(curr_params, coords, value_fn_aug)
            norm = jnp.linalg.norm(curr_params - new_params, ord=1)/len(curr_params)
            if norm <= 1e-3:
                count += 1
            else:
                count = 0
            # if not (iter % 500):
            #     print(f'Norm: {norm}')
            # pdb.set_trace()
            curr_params = new_params
            # p1_params.append(curr_params)
            pbar1.update(1)
            iter += 1

    # pdb.set_trace()
    value_1 = jax.vmap(value_fn_aug)(curr_params, coords)

    # run dsgda for second value
    curr_params = new_params
    iter = 0
    count = 0
    # p1_params = []
    norm = float('inf')
    with tqdm(total=iters, position=0) as pbar1:
        while norm > 1e-3 and count < 10 and iter <= iters:
            new_params = gradient_step(curr_params, coords, value_fn_coarse)
            norm = jnp.linalg.norm(curr_params - new_params, ord=1)/len(curr_params)
            if norm <= 1e-3:
                count += 1
            else:
                count = 0
            curr_params = new_params
            # p1_params.append(curr_params)
            pbar1.update(1)
            iter += 1

    value_2 = jax.vmap(value_fn_coarse)(curr_params, coords)


    return value_1 - value_2

