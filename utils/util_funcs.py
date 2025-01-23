"""
This file contains utility functions used in the project
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d


def final_cost_function_batch(x, p):
    """
    Given a batch of input (x, p) compute the final cost function. `final_cost_function` implements the same with
    single input. To be used when loading data from torch
    :param x: state of size (n, x_dim)
    :param p: belief of size (n, 1)
    :return: array of size (n, 1) of final values
    """
    x1 = x[:, :2]
    x2 = x[:, 4:6]
    g1 = np.array((0, 1))
    g2 = np.array((0, -1))

    dist1 = np.linalg.norm(x1 - g1, axis=1).reshape(-1, 1) ** 2
    dist2 = np.linalg.norm(x1 - g2, axis=1).reshape(-1, 1) ** 2

    dist1_p2 = np.linalg.norm(x2 - g1, axis=1).reshape(-1, 1) ** 2
    dist2_p2 = np.linalg.norm(x2 - g2, axis=1).reshape(-1, 1) ** 2

    return np.multiply(p, dist1) + np.multiply((1 - p), dist2) - \
           (np.multiply(p, dist1_p2) + np.multiply((1 - p), dist2_p2))

@jax.jit
def final_cost_function(x, p):
    x1 = x[:2]
    x2 = x[4:6]
    g1 = jnp.array((0, 1))
    g2 = jnp.array((0, -1))

    dist1 = jnp.linalg.norm(x1 - g1) ** 2
    dist2 = jnp.linalg.norm(x1 - g2) ** 2

    dist1_p2 = jnp.linalg.norm(x2 - g1) ** 2
    dist2_p2 = jnp.linalg.norm(x2 - g2) ** 2

    return p * dist1 + (1 - p) * dist2 - (p * dist1_p2 + (1 - p) * dist2_p2)

@jax.jit
def final_cost_function_dual(x, p_hat):
    x1 = x[:2]
    x2 = x[4:6]
    g1 = jnp.array((0, 1))
    g2 = jnp.array((0, -1))

    p_hat_1 = p_hat[0]
    p_hat_2 = p_hat[1]

    dist1 = jnp.linalg.norm(x1 - g1) ** 2
    dist2 = jnp.linalg.norm(x1 - g2) ** 2

    dist1_p2 = jnp.linalg.norm(x2 - g1) ** 2
    dist2_p2 = jnp.linalg.norm(x2 - g2) ** 2

    final_cost_1 = p_hat_1 - (dist1 - dist1_p2)
    final_cost_2 = p_hat_2 - (dist2 - dist2_p2)

    return jnp.maximum(final_cost_1, final_cost_2)


@jax.jit
def normalize_inputs(X, v1x_max, v1y_max, v2x_max, v2y_max):
    """
    Normalize input (x) to -1 and 1 based on the max values of velocities
    :param X: states containing position and velocities in x and y direction for two players and p (or p_hat)
    :param v1x_max: max x-velocity for P1
    :param v1y_max: max y-velocity for P1
    :param v2x_max: max x-velocity for P2
    :param v2y_max: max y-velocity for P2
    :return: normalized X
    """

    t = X[0]
    x = X[1:]
    x1 = jnp.clip(x[:2], -1, 1)
    x2 = jnp.clip(x[4:6], -1, 1)

    v1_x = x[2]
    v1_y = x[3]

    v2_x = x[6]
    v2_y = x[7]
    p = x[8:]

    a = -1
    b = 1

    v1_x_b = -1 + (b - a) * (v1_x + v1x_max) / (v1x_max + v1x_max)
    v1_y_b = -1 + (b - a) * (v1_y + v1y_max) / (v1y_max + v1y_max)

    v2_x_b = -1 + (b - a) * (v2_x + v2x_max) / (v2x_max + v2x_max)
    v2_y_b = -1 + (b - a) * (v2_y + v2y_max) / (v2y_max + v2y_max)

    x_norm = jnp.concatenate((t.reshape(-1, ), x1, v1_x_b.reshape(-1, ), v1_y_b.reshape(-1, ),
                              x2, v2_x_b.reshape(-1, ), v2_y_b.reshape(-1, ), p.reshape(-1, )))

    return x_norm


@jax.jit
def normalize_to_max_1d(x, v1x_max, v1y_max, v2x_max, v2y_max):
    x1 = jnp.clip(x[:2], -1, 1)
    x2 = jnp.clip(x[4:6], -1, 1)

    v1_x = x[2]
    v1_y = x[3]

    v2_x = x[6]
    v2_y = x[7]
    p = x[8:]

    a = -1
    b = 1

    v1_x_b = -1 + (b - a) * (v1_x + v1x_max) / (v1x_max + v1x_max)
    v1_y_b = -1 + (b - a) * (v1_y + v1y_max) / (v1y_max + v1y_max)

    v2_x_b = -1 + (b - a) * (v2_x + v2x_max) / (v2x_max + v2x_max)
    v2_y_b = -1 + (b - a) * (v2_y + v2y_max) / (v2y_max + v2y_max)

    x_norm = jnp.concatenate((x1, v1_x_b.reshape(-1, ), v1_y_b.reshape(-1, ),
                              x2, v2_x_b.reshape(-1, ), v2_y_b.reshape(-1, ), p.reshape(-1, )))

    return x_norm

# @jax.jit
# def normalize_to_max_1d_w_t(X, v1x_max, v1y_max, v2x_max, v2y_max):
#     t = X[0]
#     x = X[1:]
#
#     x1 = jnp.clip(x[:2], -1, 1)
#     x2 = jnp.clip(x[4:6], -1, 1)
#
#     v1_x = x[2]
#     v1_y = x[3]
#
#     v2_x = x[6]
#     v2_y = x[7]
#     p = x[8:]
#
#     a = -1
#     b = 1
#
#     v1_x_b = -1 + (b - a) * (v1_x + v1x_max) / (v1x_max + v1x_max)
#     v1_y_b = -1 + (b - a) * (v1_y + v1y_max) / (v1y_max + v1y_max)
#
#     v2_x_b = -1 + (b - a) * (v2_x + v2x_max) / (v2x_max + v2x_max)
#     v2_y_b = -1 + (b - a) * (v2_y + v2y_max) / (v2y_max + v2y_max)
#
#     x_norm = jnp.concatenate((t.reshape(-1, ), x1, v1_x_b.reshape(-1, ), v1_y_b.reshape(-1, ),
#                               x2, v2_x_b.reshape(-1, ), v2_y_b.reshape(-1, ), p.reshape(-1, )))
#
#     return x_norm


@jax.jit
def x_next(x, u, v, dt=0.1):
    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)

    x1 = x[:4]
    x2 = x[4:]

    x1_n = x1[:2] + x1[2:] * dt + 0.5 * u * dt ** 2
    vx1_n = x1[2:] + u * dt

    x2_n = x2[:2] + x2[2:] * dt + 0.5 * v * dt ** 2
    vx2_n = x2[2:] + v * dt

    x_n = jnp.concatenate((x1_n.reshape(-1, ), vx1_n.reshape(-1, ), x2_n.reshape(-1, ), vx2_n.reshape(-1, )))

    return x_n

@jax.jit
def running_cost(u, v):
    R1 = jnp.array([[0.05, 0.],
                    [0., 0.025]])

    R2 = jnp.array([[0.05, 0],
                    [0., 0.1]])

    loss1 = jnp.sum(jnp.multiply(jnp.diag(R1), u ** 2), axis=-1)
    loss2 = jnp.sum(jnp.multiply(jnp.diag(R2), v ** 2), axis=-1)

    return loss1 - loss2

@jax.jit
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

@jax.jit
def normalize_to_max_final(x, v1x_max, v1y_max, v2x_max, v2y_max):
    x1 = jnp.clip(x[:2], -1, 1)
    x2 = jnp.clip(x[4:6], -1, 1)

    v1_x = x[2]
    v1_y = x[3]

    v2_x = x[6]
    v2_y = x[7]

    a = -1
    b = 1

    v1_x_b = -1 + (b - a) * (v1_x + v1x_max) / (v1x_max + v1x_max)
    v1_y_b = -1 + (b - a) * (v1_y + v1y_max) / (v1y_max + v1y_max)

    v2_x_b = -1 + (b - a) * (v2_x + v2x_max) / (v2x_max + v2x_max)
    v2_y_b = -1 + (b - a) * (v2_y + v2y_max) / (v2y_max + v2y_max)

    x_norm = jnp.concatenate((x1, v1_x_b.reshape(-1, ), v1_y_b.reshape(-1, ),
                              x2, v2_x_b.reshape(-1, ), v2_y_b.reshape(-1, )))

    return x_norm

@jax.jit
def unnormalize_states(X, v1x_max, v1y_max, v2x_max, v2y_max):
    x = X[:, :8]
    p = X[:, 8:]

    x1 = x[:, :2]
    x2 = x[:, 4:6]

    v1_xn = x[:, 2]
    v1_yn = x[:, 3]

    v2_xn = x[:, 6]
    v2_yn = x[:, 7]

    v1_x = v1_xn * v1x_max
    v1_y = v1_yn * v1y_max

    v2_x = v2_xn * v2x_max
    v2_y = v2_yn * v2y_max

    x_unnorm = jnp.concatenate((x1, v1_x.reshape(-1, 1), v1_y.reshape(-1, 1),
                                x2, v2_x.reshape(-1, 1), v2_y.reshape(-1, 1), p.reshape(-1, 1)), axis=1)

    return x_unnorm

@jax.jit
def unnormalize_states_dual(X, v1x_max, v1y_max, v2x_max, v2y_max):
    x = X[:, :8]
    p = X[:, 8:]

    x1 = x[:, :2]
    x2 = x[:, 4:6]

    v1_xn = x[:, 2]
    v1_yn = x[:, 3]

    v2_xn = x[:, 6]
    v2_yn = x[:, 7]

    v1_x = v1_xn * v1x_max
    v1_y = v1_yn * v1y_max

    v2_x = v2_xn * v2x_max
    v2_y = v2_yn * v2y_max

    x_unnorm = jnp.concatenate((x1, v1_x.reshape(-1, 1), v1_y.reshape(-1, 1),
                                x2, v2_x.reshape(-1, 1), v2_y.reshape(-1, 1), p.reshape(-1, 2)), axis=1)

    return x_unnorm









def discrete_lqr(Ad, Bd, Q, R, Qf, N):
    """
    Compute K_k matrices for a finite horizon discrete-time LQR problem.

    Parameters:
    A (np.ndarray): Discrete-time system matrix.
    B (np.ndarray): Discrete-time input matrix.
    Q (np.ndarray): Discrete-time State cost matrix.
    R (np.ndarray): Discrete-time Input cost matrix.
    Qf (np.ndarray): Final state cost matrix.
    dt (float): Sampling time.
    N (int): Number of time steps.

    Returns:
    list: A list of K_k matrices.
    """
    # Discretize the cost matrices
    Qd = Q
    Rd = R

    # Initialize the list for K_k matrices
    K_matrices = []

    # Initialize P_N
    Pk = Qf

    # Backward recursion to compute P_k and K_k
    for k in range(N, 0, -1):
        Fk = jnp.linalg.inv(Rd + Bd.T @ Pk @ Bd) @ Bd.T @ Pk @ Ad
        Pk = Fk.T @ Rd @ Fk + (Ad - Bd @ Fk).T @ Pk @ (Ad - Bd @ Fk)
        K_matrices.insert(0, Fk)

    return K_matrices


def continuous_importance_sampling(t_values, errors, n_samples):
    # Ensure errors are non-negative
    errors = np.maximum(errors, 0)

    # Normalize errors to create a probability distribution
    probabilities = errors / np.sum(errors)

    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(probabilities)

    # Ensure the CDF starts at 0 and ends at 1
    cdf = np.insert(cdf, 0, 0)
    t_values_extended = np.insert(t_values, 0, 0)

    # Create an interpolation function for the inverse CDF
    inverse_cdf = interp1d(cdf, t_values_extended, bounds_error=False, fill_value=(0, 1))

    # Generate uniform random numbers
    u = np.random.uniform(0, 1, n_samples)

    # Use inverse transform sampling
    samples = inverse_cdf(u)

    return samples