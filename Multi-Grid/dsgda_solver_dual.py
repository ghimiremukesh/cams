import copy
from functools import partial

import jax
from utils import util_funcs
import jax.numpy as jnp
from tqdm import tqdm
from flax.linen import relu
# import pdb

EPSILON = 1e-6
p_hat_bound = 1.4

bounds = jnp.concatenate((
    jnp.array([[-12., 12.]] * 6),
    jnp.array([[EPSILON, 1 - EPSILON]] * 2),
    jnp.array([[-p_hat_bound, p_hat_bound]] * 6),
    jnp.array([[-12., 12.]] * 6),
))
min_bounds = bounds[:, 0]
max_bounds = bounds[:, 1]


def dsgda_solver(val_model, model_params, coords, t, tau, init_policies=None, iters=120000):
    # value function
    @jax.jit
    def value_fn(params, X):
        v1 = params[:2]
        v2 = params[2:4]
        v3 = params[4:6]
        l1 = params[6]
        l2 = params[7]
        p_hat_1 = params[8:10]
        p_hat_2 = params[10:12]
        p_hat_3 = params[12:14]
        u1 = params[14:16]
        u2 = params[16:18]
        u3 = params[18:20]

        l3 = 1 - l1 - l2 + EPSILON

        x = X[:-2]
        p_hat = x[-2:]

        # additional constraint
        cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3) ** 2) + 1e-12)

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=tau)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=tau)
        x_next_3 = util_funcs.x_next(x, u3, v3, dt=tau)

        ins_cost_1 = tau * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = tau * util_funcs.running_cost(u2, v2).reshape(-1, )
        ins_cost_3 = tau * util_funcs.running_cost(u3, v3).reshape(-1, )

        input_1 = jnp.concat((x_next_1, p_hat_1 - ins_cost_1))
        input_2 = jnp.concat((x_next_2, p_hat_2 - ins_cost_2))
        input_3 = jnp.concat((x_next_3, p_hat_3 - ins_cost_3))

        v_bound_next = util_funcs.compute_bounds(1 - t + tau, 12)

        # apply normalized input to the model and keep track for gradient computation
        @jax.jit
        def apply_to_model(input_):
            rescaled_input = util_funcs.normalize_to_max_1d(input_, v_bound_next, v_bound_next, v_bound_next, v_bound_next)
            return val_model.apply(model_params, rescaled_input)

        val_1 = apply_to_model(input_1)
        val_2 = apply_to_model(input_2)
        val_3 = apply_to_model(input_3)

        lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
        p_hat_penalty = 8 * relu(cons_phat)

        objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + lam_penalty + p_hat_penalty

        return objective.reshape(())

    @jax.jit
    def gradient_step(params, x):
        c = 5e-2  # 1e-1
        alpha = 1e-2
        grad = jax.vmap(jax.grad(value_fn, argnums=0))(params, x)
        p2_params = params[:, :14]
        p1_params = params[:, 14:]
        new_p2_params = (p2_params - c * grad[:, :14]).clip(min_bounds[:14], max_bounds[:14])
        new_params = jnp.concatenate((new_p2_params, p1_params), axis=1)
        grad = jax.vmap(jax.grad(value_fn, argnums=0))(new_params, x)
        new_p1_params = (p1_params + alpha * grad[:, 14:]).clip(min_bounds[14:], max_bounds[14:])
        new_params = jnp.concatenate((new_p2_params, new_p1_params), axis=1)

        return new_params

    key = jax.random.PRNGKey(1)
    key5, key6 = jax.random.split(key, 2)
    num_points = coords.shape[0]

    if init_policies is not None:
        curr_params = jnp.array(init_policies)
    else:
        params_d = jax.random.uniform(key5, (num_points, 6), minval=-1, maxval=-0.5)
        params_d_lam = jnp.array([[EPSILON, 1 - EPSILON]])
        params_d_phat = jax.random.uniform(key6, (num_points, 6), minval=-p_hat_bound, maxval=p_hat_bound)
        params_u = jax.random.uniform(key6, (num_points, 6), minval=-1, maxval=1)

        params_d_lam = jnp.repeat(params_d_lam, num_points, axis=0)

        params = jnp.concatenate((params_d, params_d_lam, params_d_phat, params_u), axis=1)
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
        v1 = params[:2]
        v2 = params[2:4]
        v3 = params[4:6]
        l1 = params[6]
        l2 = params[7]
        p_hat_1 = params[8:10]
        p_hat_2 = params[10:12]
        p_hat_3 = params[12:14]
        u1 = params[14:16]
        u2 = params[16:18]
        u3 = params[18:20]

        l3 = 1 - l1 - l2 + EPSILON

        x = X[:-2]
        p_hat = x[-2:]

        # additional constraint
        cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3) ** 2) + 1e-12)

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=tau)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=tau)
        x_next_3 = util_funcs.x_next(x, u3, v3, dt=tau)

        ins_cost_1 = tau * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = tau * util_funcs.running_cost(u2, v2).reshape(-1, )
        ins_cost_3 = tau * util_funcs.running_cost(u3, v3).reshape(-1, )

        p_hat_1 = p_hat_1 - ins_cost_1
        p_hat_2 = p_hat_2 - ins_cost_2
        p_hat_3 = p_hat_3 - ins_cost_3

        val_1 = util_funcs.final_cost_function_dual(x_next_1, p_hat_1)
        val_2 = util_funcs.final_cost_function_dual(x_next_2, p_hat_2)
        val_3 = util_funcs.final_cost_function_dual(x_next_3, p_hat_3)

        lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
        p_hat_penalty = 8 * relu(cons_phat)

        objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + p_hat_penalty + lam_penalty
        return objective.reshape(())

    @jax.jit
    def gradient_step(params, x):
        c = 5e-2  # 1e-1
        alpha = 1e-2
        grad = jax.vmap(jax.grad(value_fn, argnums=0))(params, x)
        p2_params = params[:, :14]
        p1_params = params[:, 14:]
        new_p2_params = (p2_params - c * grad[:, :14]).clip(min_bounds[:14], max_bounds[:14])
        new_params = jnp.concatenate((new_p2_params, p1_params), axis=1)
        grad = jax.vmap(jax.grad(value_fn, argnums=0))(new_params, x)
        new_p1_params = (p1_params + alpha * grad[:, 14:]).clip(min_bounds[14:], max_bounds[14:])
        new_params = jnp.concatenate((new_p2_params, new_p1_params), axis=1)

        return new_params

    key = jax.random.PRNGKey(1)
    key5, key6 = jax.random.split(key, 2)
    num_points = coords.shape[0]

    if init_policies is not None:
        curr_params = jnp.array(init_policies)
    else:
        params_d = jax.random.uniform(key5, (num_points, 6), minval=-1, maxval=-0.5)
        params_d_lam = jnp.array([[EPSILON, 1 - EPSILON]])
        params_d_phat = jax.random.uniform(key6, (num_points, 6), minval=-p_hat_bound, maxval=p_hat_bound)
        params_u = jax.random.uniform(key6, (num_points, 6), minval=-1, maxval=1)

        params_d_lam = jnp.repeat(params_d_lam, num_points, axis=0)

        params = jnp.concatenate((params_d, params_d_lam, params_d_phat, params_u), axis=1)
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
        v1 = params[:2]
        v2 = params[2:4]
        v3 = params[4:6]
        l1 = params[6]
        l2 = params[7]
        p_hat_1 = params[8:10]
        p_hat_2 = params[10:12]
        p_hat_3 = params[12:14]
        u1 = params[14:16]
        u2 = params[16:18]
        u3 = params[18:20]

        l3 = 1 - l1 - l2 + EPSILON

        x = X[:-2]
        p_hat = x[-2:]

        # additional constraint
        cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3) ** 2) + 1e-12)

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=H)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=H)
        x_next_3 = util_funcs.x_next(x, u3, v3, dt=H)

        ins_cost_1 = H * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = H * util_funcs.running_cost(u2, v2).reshape(-1, )
        ins_cost_3 = H * util_funcs.running_cost(u3, v3).reshape(-1, )

        input_1 = jnp.concat((x_next_1, p_hat_1 - ins_cost_1))
        input_2 = jnp.concat((x_next_2, p_hat_2 - ins_cost_2))
        input_3 = jnp.concat((x_next_3, p_hat_3 - ins_cost_3))

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
        val_3 = apply_to_model(input_3)

        lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
        p_hat_penalty = 8 * relu(cons_phat)

        objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + lam_penalty + p_hat_penalty

        return objective.reshape(())

    @jax.jit
    def value_fn_coarse(params, X):
        v1 = params[:2]
        v2 = params[2:4]
        v3 = params[4:6]
        l1 = params[6]
        l2 = params[7]
        p_hat_1 = params[8:10]
        p_hat_2 = params[10:12]
        p_hat_3 = params[12:14]
        u1 = params[14:16]
        u2 = params[16:18]
        u3 = params[18:20]

        l3 = 1 - l1 - l2 + EPSILON

        x = X[:-2]
        p_hat = x[-2:]

        # additional constraint
        cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3) ** 2) + 1e-12)

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=H)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=H)
        x_next_3 = util_funcs.x_next(x, u3, v3, dt=H)

        ins_cost_1 = H * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = H * util_funcs.running_cost(u2, v2).reshape(-1, )
        ins_cost_3 = H * util_funcs.running_cost(u3, v3).reshape(-1, )

        input_1 = jnp.concat((x_next_1, p_hat_1 - ins_cost_1))
        input_2 = jnp.concat((x_next_2, p_hat_2 - ins_cost_2))
        input_3 = jnp.concat((x_next_3, p_hat_3 - ins_cost_3))

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
        val_3 = apply_to_model(input_3)

        lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
        p_hat_penalty = 8 * relu(cons_phat)

        objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + lam_penalty + p_hat_penalty

        return objective.reshape(())


    @partial(jax.jit, static_argnums=(2, ))
    def gradient_step(params, x, val_fn):
        c = 5e-2  # 1e-1
        alpha = 1e-2
        grad = jax.vmap(jax.grad(val_fn, argnums=0))(params, x)
        p2_params = params[:, :14]
        p1_params = params[:, 14:]
        new_p2_params = (p2_params - c * grad[:, :14]).clip(min_bounds[:14], max_bounds[:14])
        new_params = jnp.concatenate((new_p2_params, p1_params), axis=1)
        grad = jax.vmap(jax.grad(val_fn, argnums=0))(new_params, x)
        new_p1_params = (p1_params + alpha * grad[:, 14:]).clip(min_bounds[14:], max_bounds[14:])
        new_params = jnp.concatenate((new_p2_params, new_p1_params), axis=1)

        return new_params

    key = jax.random.PRNGKey(1)
    key5, key6 = jax.random.split(key, 2)
    num_points = coords.shape[0]
    params_d = jax.random.uniform(key5, (num_points, 6), minval=-1, maxval=-0.5)
    params_d_lam = jnp.array([[EPSILON, 1 - EPSILON]])
    params_d_phat = jax.random.uniform(key6, (num_points, 6), minval=-p_hat_bound, maxval=p_hat_bound)
    params_u = jax.random.uniform(key6, (num_points, 6), minval=-1, maxval=1)

    params_d_lam = jnp.repeat(params_d_lam, num_points, axis=0)

    params = jnp.concatenate((params_d, params_d_lam, params_d_phat, params_u), axis=1)
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

def coarse_solvers_adaptive(aug_val_fn, coarse_val_fn, coords, t, H, h, init_policies=None, iters=100000):
    # first solve the O_H(V_H + e_H)_t+1 and then O_H(V_H)_t+1
    @jax.jit
    def value_fn_aug(params, X):
        v1 = params[:2]
        v2 = params[2:4]
        v3 = params[4:6]
        l1 = params[6]
        l2 = params[7]
        p_hat_1 = params[8:10]
        p_hat_2 = params[10:12]
        p_hat_3 = params[12:14]
        u1 = params[14:16]
        u2 = params[16:18]
        u3 = params[18:20]

        l3 = 1 - l1 - l2 + EPSILON

        x = X[:-2]
        p_hat = x[-2:]

        # additional constraint
        cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3) ** 2) + 1e-12)

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=H)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=H)
        x_next_3 = util_funcs.x_next(x, u3, v3, dt=H)

        ins_cost_1 = H * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = H * util_funcs.running_cost(u2, v2).reshape(-1, )
        ins_cost_3 = H * util_funcs.running_cost(u3, v3).reshape(-1, )

        input_1 = jnp.concat((x_next_1, p_hat_1 - ins_cost_1))
        input_2 = jnp.concat((x_next_2, p_hat_2 - ins_cost_2))
        input_3 = jnp.concat((x_next_3, p_hat_3 - ins_cost_3))

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
        val_3 = apply_to_model(input_3)

        lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
        p_hat_penalty = 8 * relu(cons_phat)

        objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + lam_penalty + p_hat_penalty

        return objective.reshape(())

    @jax.jit
    def value_fn_coarse(params, X):
        v1 = params[:2]
        v2 = params[2:4]
        v3 = params[4:6]
        l1 = params[6]
        l2 = params[7]
        p_hat_1 = params[8:10]
        p_hat_2 = params[10:12]
        p_hat_3 = params[12:14]
        u1 = params[14:16]
        u2 = params[16:18]
        u3 = params[18:20]

        l3 = 1 - l1 - l2 + EPSILON

        x = X[:-2]
        p_hat = x[-2:]

        # additional constraint
        cons_phat = jnp.sqrt(jnp.sum((p_hat - l1 * p_hat_1 - l2 * p_hat_2 - l3 * p_hat_3) ** 2) + 1e-12)

        x_next_1 = util_funcs.x_next(x, u1, v1, dt=H)
        x_next_2 = util_funcs.x_next(x, u2, v2, dt=H)
        x_next_3 = util_funcs.x_next(x, u3, v3, dt=H)

        ins_cost_1 = H * util_funcs.running_cost(u1, v1).reshape(-1, )
        ins_cost_2 = H * util_funcs.running_cost(u2, v2).reshape(-1, )
        ins_cost_3 = H * util_funcs.running_cost(u3, v3).reshape(-1, )

        input_1 = jnp.concat((x_next_1, p_hat_1 - ins_cost_1))
        input_2 = jnp.concat((x_next_2, p_hat_2 - ins_cost_2))
        input_3 = jnp.concat((x_next_3, p_hat_3 - ins_cost_3))

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
        val_3 = apply_to_model(input_3)

        lam_penalty = 8 * relu(-l3)  # if lambda_3 is negative penalty is high
        p_hat_penalty = 8 * relu(cons_phat)

        objective = l1 * val_1 + l2 * val_2 + l3 * val_3 + lam_penalty + p_hat_penalty

        return objective.reshape(())


    @partial(jax.jit, static_argnums=(2, ))
    def gradient_step(params, x, val_fn):
        c = 5e-2  # 1e-1
        alpha = 1e-2
        grad = jax.vmap(jax.grad(val_fn, argnums=0))(params, x)
        p2_params = params[:, :14]
        p1_params = params[:, 14:]
        new_p2_params = (p2_params - c * grad[:, :14]).clip(min_bounds[:14], max_bounds[:14])
        new_params = jnp.concatenate((new_p2_params, p1_params), axis=1)
        grad = jax.vmap(jax.grad(val_fn, argnums=0))(new_params, x)
        new_p1_params = (p1_params + alpha * grad[:, 14:]).clip(min_bounds[14:], max_bounds[14:])
        new_params = jnp.concatenate((new_p2_params, new_p1_params), axis=1)

        return new_params

    key = jax.random.PRNGKey(1)
    key5, key6 = jax.random.split(key, 2)
    num_points = coords.shape[0]

    if init_policies is not None:
        params = jnp.array(init_policies)
    else:
        params_d = jax.random.uniform(key5, (num_points, 6), minval=-1, maxval=-0.5)
        params_d_lam = jnp.array([[EPSILON, 1 - EPSILON]])
        params_d_phat = jax.random.uniform(key6, (num_points, 6), minval=-p_hat_bound, maxval=p_hat_bound)
        params_u = jax.random.uniform(key6, (num_points, 6), minval=-1, maxval=1)

        params_d_lam = jnp.repeat(params_d_lam, num_points, axis=0)

        params = jnp.concatenate((params_d, params_d_lam, params_d_phat, params_u), axis=1)

    # params_copy = copy.deepcopy(params)

    # run dsgda for first value
    curr_params = params
    iter = 0
    # p1_params = []
    norm = float('inf')
    count = 0
    with tqdm(total=iters, position=0) as pbar1:
        while norm > 5e-3 and count < 10 and iter <= iters:
            new_params = gradient_step(curr_params, coords, value_fn_aug)
            norm = jnp.linalg.norm(curr_params - new_params, ord=1)/len(curr_params)
            if norm <= 5e-3:
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
        while norm > 5e-3 and count < 10 and iter <= iters:
            new_params = gradient_step(curr_params, coords, value_fn_coarse)
            norm = jnp.linalg.norm(curr_params - new_params, ord=1)/len(curr_params)
            if norm <= 5e-3:
                count += 1
            else:
                count = 0
            curr_params = new_params
            # p1_params.append(curr_params)
            pbar1.update(1)
            iter += 1

    value_2 = jax.vmap(value_fn_coarse)(curr_params, coords)


    return value_1 - value_2

