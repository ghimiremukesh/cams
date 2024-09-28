import numpy as np
from jax import jacfwd, jit
from jax import numpy as jnp
import os


GOAL_1 = (0, 1)
GOAL_2 = (0, -1)


GOAL_1_3d = (0, 0, 1)
GOAL_2_3d = (0, 0, -1)


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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


@jit
def analytical_sol(x, p, tau):
    x1 = x[:4]
    x2 = x[4:8]
    r1x = 0.05
    r1y = 0.025
    r2x = 0.05
    r2y = 0.1

    tt = 2 * p - 1

    ux = -3 * (x1[0] + x1[2] * tau) * tau / (3 * r1x + tau ** 3)
    uy = -3 * (x1[1] + x1[3] * tau - tt) * tau / (3 * r1y + tau ** 3)

    vx = -3 * (x2[0] + x2[2] * tau) * tau / (3 * r2x + tau ** 3)
    vy = -3 * (x2[1] + x2[3] * tau - tt) * tau / (3 * r2y + tau ** 3)

    # u = jnp.array([[ux, uy]])
    # v = jnp.array([[vx, vy]])
    u = jnp.concatenate((ux.reshape(-1, ), uy.reshape(-1, )))
    v = jnp.concatenate((vx.reshape(-1, ), vy.reshape(-1, )))

    return u, v


@jit
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
    
@jit
def final_cost_function_3d(x, p):
    x1 = x[:3]
    x2 = x[6:9]
    g1 = jnp.array((0, 0, 1))
    g2 = jnp.array((0, 0, -1))

    dist1 = jnp.linalg.norm(x1 - g1) ** 2
    dist2 = jnp.linalg.norm(x1 - g2) ** 2

    dist1_p2 = jnp.linalg.norm(x2 - g1) ** 2
    dist2_p2 = jnp.linalg.norm(x2 - g2) ** 2

    return p * dist1 + (1 - p) * dist2 - (p * dist1_p2 + (1 - p) * dist2_p2)
    
    
    
@jit
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
    
@jit
def final_cost_function_3d_dual(x, p_hat):
    x1 = x[:3]
    x2 = x[6:9]
    g1 = jnp.array((0, 0, 1))
    g2 = jnp.array((0, 0, -1))

    p_hat_1 = p_hat[0]
    p_hat_2 = p_hat[1]

    dist1 = jnp.linalg.norm(x1 - g1) ** 2
    dist2 = jnp.linalg.norm(x1 - g2) ** 2

    dist1_p2 = jnp.linalg.norm(x2 - g1) ** 2
    dist2_p2 = jnp.linalg.norm(x2 - g2) ** 2

    final_cost_1 = p_hat_1 - (dist1 - dist1_p2)
    final_cost_2 = p_hat_2 - (dist2 - dist2_p2)

    return jnp.maximum(final_cost_1, final_cost_2)


@jit
def final_cost_function_batch(x, p):
    x1 = x[:, :2]
    x2 = x[:, 4:6]
    g1 = jnp.array((0, 1))
    g2 = jnp.array((0, -1))

    dist1 = jnp.linalg.norm(x1 - g1, axis=1).reshape(-1, 1) ** 2
    dist2 = jnp.linalg.norm(x1 - g2, axis=1).reshape(-1, 1) ** 2

    dist1_p2 = jnp.linalg.norm(x2 - g1, axis=1).reshape(-1, 1) ** 2
    dist2_p2 = jnp.linalg.norm(x2 - g2, axis=1).reshape(-1, 1) ** 2

    return jnp.multiply(p, dist1) + jnp.multiply((1 - p), dist2) - \
           (jnp.multiply(p, dist1_p2) + jnp.multiply((1 - p), dist2_p2))


@jit
def xdot(x, u, v):
    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)

    A = jnp.zeros((8, 8))
    A = A.at[0, 2].set(1.)
    A = A.at[1, 3].set(1.)
    A = A.at[4, 6].set(1.)
    A = A.at[5, 7].set(1.)

    B = jnp.zeros((8, 2))
    B = B.at[2, 0].set(1.)
    B = B.at[3, 1].set(1.)

    C = jnp.zeros((8, 2))
    C = C.at[6, 0].set(1.)
    C = C.at[7, 1].set(1.)

    return (A @ x + B @ u + C @ v)


@jit
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

    x_n = jnp.concat((x1_n.reshape(-1, ), vx1_n.reshape(-1, ), x2_n.reshape(-1, ), vx2_n.reshape(-1, )))

    return x_n
    
@jit
def x_next_3d(x, u, v, dt=0.1):
    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)

    x1 = x[:6]
    x2 = x[6:]

    x1_n = x1[:3] + x1[3:] * dt + 0.5 * u * dt ** 2
    vx1_n = x1[3:] + u * dt

    x2_n = x2[:3] + x2[3:] * dt + 0.5 * v * dt ** 2
    vx2_n = x2[3:] + v * dt

    x_n = jnp.concat((x1_n.reshape(-1, ), vx1_n.reshape(-1, ), x2_n.reshape(-1, ), vx2_n.reshape(-1, )))

    return x_n
    
    

# @jit
def x_next_corr(x, u, v, tau):
    x = x.reshape(-1, 1)
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)

    x1 = x[:4]
    x2 = x[4:]

    dt = 0.1


    x1_n = x1[:2] + x1[2:] * u * (0.5 * dt ** 2 - (1/(6 * tau)) * dt ** 3)
    vx1_n = x1[2:] + u * (dt - (1/(2 * tau)) * dt ** 2)

    x2_n = x2[:2] + x2[2:] * v * (0.5 * dt ** 2 - (1/(6 * tau)) * dt ** 3)
    vx2_n = x2[2:] + v * (dt - (1/(2 * tau)) * dt ** 2)

    x_n = jnp.concat((x1_n.reshape(-1, ), vx1_n.reshape(-1, ), x2_n.reshape(-1, ), vx2_n.reshape(-1, )))

    return x_n

@jit
def go_forward(x, U, D, dt=0.1, a=0.5):
    ux = U[:, 0].reshape(-1, 1)
    uy = U[:, 1].reshape(-1, 1)

    dx = D[:, 0].reshape(-1, 1)
    dy = D[:, 1].reshape(-1, 1)

    X1 = x[:, :4]
    X2 = x[:, 4:8]

    # for p1
    x1 = X1[:, 0].reshape(-1, 1)
    y1 = X1[:, 1].reshape(-1, 1)
    vx1 = X1[:, 2].reshape(-1, 1)
    vy1 = X1[:, 3].reshape(-1, 1)

    x1dot = vx1
    y1dot = vy1
    vx1dot = ux
    vy1dot = uy

    x1_new = x1 + x1dot * dt + a * ux * dt ** 2
    y1_new = y1 + y1dot * dt + a * uy * dt ** 2
    vx1_new = vx1 + vx1dot * dt
    vy1_new = vy1 + vy1dot * dt

    # for p2
    x2 = X2[:, 0].reshape(-1, 1)
    y2 = X2[:, 1].reshape(-1, 1)
    vx2 = X2[:, 2].reshape(-1, 1)
    vy2 = X2[:, 3].reshape(-1, 1)

    x2dot = vx2
    y2dot = vy2
    vx2dot = dx
    vy2dot = dy

    x2_new = x2 + x2dot * dt + a * dx * dt ** 2
    y2_new = y2 + y2dot * dt + a * dy * dt ** 2
    vx2_new = vx2 + vx2dot * dt
    vy2_new = vy2 + vy2dot * dt

    return jnp.concat((x1_new, y1_new, vx1_new, vy1_new, x2_new, y2_new, vx2_new, vy2_new), axis=1)


@jit
def running_cost(u, v):
    R1 = jnp.array([[0.05, 0.],
                    [0., 0.025]])

    R2 = jnp.array([[0.05, 0],
                    [0., 0.1]])

    loss1 = jnp.sum(jnp.multiply(jnp.diag(R1), u ** 2), axis=-1)
    loss2 = jnp.sum(jnp.multiply(jnp.diag(R2), v ** 2), axis=-1)

    return loss1 - loss2
    
@jit
def running_cost_3d(u, v):
    R1 = np.array([[0.05, 0., 0.],[0., 0.05, 0.],
                   [0., 0., 0.025]])
    
    R2 = np.array([[0.05, 0., 0.],[0., 0.05, 0.],
                   [0., 0., 0.1]])

    loss1 = jnp.sum(jnp.multiply(jnp.diag(R1), u ** 2), axis=-1)
    loss2 = jnp.sum(jnp.multiply(jnp.diag(R2), v ** 2), axis=-1)

    return loss1 - loss2

@jit
def running_cost_corr(u, v, tau):
    R1 = jnp.array([[0.05, 0.],
                   [0., 0.025]])

    R2 = jnp.array([[0.05, 0],
                   [0., 0.1]])

    dt = 0.1

    factor = (1 - (dt / tau) + (dt ** 2) / (3 * tau ** 2))

    loss1 = jnp.sum(jnp.multiply(jnp.diag(R1), u ** 2), axis=-1)
    loss2 = jnp.sum(jnp.multiply(jnp.diag(R2), v ** 2), axis=-1)

    return (loss1 - loss2) * factor

# def running_cost_batch(u, v):
#     R1 = jnp.array([[0.05, 0.],
#                     [0., 0.025]])
#
#     R2 = jnp.array([[0.05, 0],
#                     [0., 0.1]])
#
#     loss1 = jnp.sum(jnp.multiply(jnp.diag(R1), u ** 2), axis=-1)
#     loss2 = jnp.sum(jnp.multiply(jnp.diag(R2), v ** 2), axis=-1)
#
#     return loss1 - loss2
@jit
def normalize_to_max(x, v1x_max, v1y_max, v2x_max, v2y_max):
    x1 = jnp.clip(x[:, :2], -1, 1)
    x2 = jnp.clip(x[:, 4:6], -1, 1)
    p = x[:, -1]

    v1_x = x[:, 2]
    v1_y = x[:, 3]

    v2_x = x[:, 6]
    v2_y = x[:, 7]

    a = -1
    b = 1

    v1_x_b = -1 + (b - a) * (v1_x + v1x_max) / (v1x_max + v1x_max)
    v1_y_b = -1 + (b - a) * (v1_y + v1y_max) / (v1y_max + v1y_max)

    v2_x_b = -1 + (b - a) * (v2_x + v2x_max) / (v2x_max + v2x_max)
    v2_y_b = -1 + (b - a) * (v2_y + v2y_max) / (v2y_max + v2y_max)

    # x_norm = deepcopy(x)
    # x_norm.at[:, :2].set(x1)
    # x_norm.at[:, 2].set(v1_x_b)
    # x_norm.at[:, 3].set(v1_y_b)
    # x_norm.at[:, 4:6].set(x2)
    # x_norm.at[:, 6].set(v2_x_b)
    # x_norm.at[:, 7].set(v2_y_b)

    x_norm = jnp.concatenate((x1, v1_x_b.reshape(-1, 1), v1_y_b.reshape(-1, 1),
                              x2, v2_x_b.reshape(-1, 1), v2_y_b.reshape(-1, 1), p.reshape(-1, 1)))
    return x_norm


@jit
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

    # x_norm = deepcopy(x)
    # x_norm.at[:2].set(x1)
    # x_norm.at[2].set(v1_x_b)
    # x_norm.at[3].set(v1_y_b)
    # x_norm.at[4:6].set(x2)
    # x_norm.at[6].set(v2_x_b)
    # x_norm.at[7].set(v2_y_b)
    x_norm = jnp.concatenate((x1, v1_x_b.reshape(-1, ), v1_y_b.reshape(-1, ),
                              x2, v2_x_b.reshape(-1, ), v2_y_b.reshape(-1, ), p.reshape(-1, )))

    return x_norm
    
@jit
def normalize_to_max_3d(x, v1x_max, v1y_max, v1z_max, v2x_max, v2y_max, v2z_max):
    x1 = jnp.clip(x[:3], -1, 1)
    x2 = jnp.clip(x[6:9], -1, 1)

    v1_x = x[3]
    v1_y = x[4]
    v1_z = x[5]

    v2_x = x[9]
    v2_y = x[10]
    v2_z = x[11]
    p = x[12:]

    a = -1
    b = 1

    v1_x_b = -1 + (b - a) * (v1_x + v1x_max) / (v1x_max + v1x_max)
    v1_y_b = -1 + (b - a) * (v1_y + v1y_max) / (v1y_max + v1y_max)
    v1_z_b = -1 + (b - a) * (v1_z + v1z_max) / (v1z_max + v1z_max)

    v2_x_b = -1 + (b - a) * (v2_x + v2x_max) / (v2x_max + v2x_max)
    v2_y_b = -1 + (b - a) * (v2_y + v2y_max) / (v2y_max + v2y_max)
    v2_z_b = -1 + (b - a) * (v2_z + v2z_max) / (v2z_max + v2z_max)

    # x_norm = deepcopy(x)
    # x_norm.at[:2].set(x1)
    # x_norm.at[2].set(v1_x_b)
    # x_norm.at[3].set(v1_y_b)
    # x_norm.at[4:6].set(x2)
    # x_norm.at[6].set(v2_x_b)
    # x_norm.at[7].set(v2_y_b)
    x_norm = jnp.concatenate((x1, v1_x_b.reshape(-1, ), v1_y_b.reshape(-1, ), v1_z_b.reshape(-1, ),
                              x2, v2_x_b.reshape(-1, ), v2_y_b.reshape(-1, ), v2_z_b.reshape(-1, ), p.reshape(-1, )))

    return x_norm
    

@jit
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
    
@jit
def normalize_to_max_final_3d(x, v1x_max, v1y_max, v1z_max, v2x_max, v2y_max, v2z_max):
    x1 = jnp.clip(x[:3], -1, 1)
    x2 = jnp.clip(x[6:9], -1, 1)

    v1_x = x[3]
    v1_y = x[4]
    v1_z = x[5]

    v2_x = x[9]
    v2_y = x[10]
    v2_z = x[11]

    a = -1
    b = 1

    v1_x_b = -1 + (b - a) * (v1_x + v1x_max) / (v1x_max + v1x_max)
    v1_y_b = -1 + (b - a) * (v1_y + v1y_max) / (v1y_max + v1y_max)
    v1_z_b = -1 + (b - a) * (v1_z + v1z_max) / (v1z_max + v1z_max)

    v2_x_b = -1 + (b - a) * (v2_x + v2x_max) / (v2x_max + v2x_max)
    v2_y_b = -1 + (b - a) * (v2_y + v2y_max) / (v2y_max + v2y_max)
    v2_z_b = -1 + (b - a) * (v2_z + v2z_max) / (v2z_max + v2z_max)

    x_norm = jnp.concatenate((x1, v1_x_b.reshape(-1, ), v1_y_b.reshape(-1, ), v1_z_b.reshape(-1, ),
                              x2, v2_x_b.reshape(-1, ), v2_y_b.reshape(-1, ), v2_z_b.reshape(-1, )))

    return x_norm
    
@jit
def normalize_to_max_1d_w_t(X, v1x_max, v1y_max, v2x_max, v2y_max):
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

from jax.tree_util import tree_map
from torch.utils import data

def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))



# for references
# @jit
# def objective_1d(params, x, p):
#     u1 = params[:2]
#     u2 = params[2:4]
#     a1 = params[4]
#     a2 = params[5]
#     v1 = params[6:8]
#     v2 = params[8:]
#
#
#     p_u1 = a1 * p + a2 * (1 - p)
#     p_u2 = 1 - p_u1
#
#     # posteriors
#     pos_1 = a1 * p / p_u1
#
#     pos_2 = (1 - a1) * p / p_u2
#
#     input_to_grad_1 = jnp.concat((x, pos_1))
#     input_to_grad_2 = jnp.concat((x, pos_2))
#
#     v_bound_next = compute_bounds(1 - t + dt, 12)
#     input_1 = normalize_to_max_1d(input_to_grad_1,
#                                v_bound_next, v_bound_next, v_bound_next, v_bound_next)
#     input_2 = normalize_to_max_1d(input_to_grad_2,
#                                v_bound_next, v_bound_next, v_bound_next, v_bound_next)
#
#     val_1 = model.apply(state_dict, input_1)
#     val_2 = model.apply(state_dict, input_2)
#
#     v_grad = lambda model_in: model.apply(state_dict, model_in)
#
#     jacs_v1 = jacfwd(v_grad)(input_1)
#     # grad_v1 = jnp.array([jacs_v1[i, 0, i, :-1] for i in range(input_1.shape[0])])
#     grad_v1 = jacs_v1[:, :-1].reshape(-1, 1)
#     jacs_v2 = jacfwd(v_grad)(input_2)
#     # grad_v2 = jnp.array([jacs_v2[i, 0, i, :-1] for i in range(input_2.shape[0])])
#     grad_v2 = jacs_v2[:, :-1].reshape(-1, 1)
#
#     # rescale grad
#     grad_v1 = rescale_grad(grad_v1, v_bound_next)
#     grad_v2 = rescale_grad(grad_v2, v_bound_next)
#
#     inner_prod_1 = dt * jnp.sum(grad_v1 * xdot(input_1[:-1], u1, v1)).reshape(-1, )
#     inner_prod_2 = dt * jnp.sum(grad_v2 * xdot(input_2[:-1], u2, v2)).reshape(-1, )
#
#     ins_cost_1 = dt * running_cost(u1, v1).reshape(-1, )
#     ins_cost_2 = dt * running_cost(u2, v2).reshape(-1, )
#
#     final_cost_1 = val_1 + ins_cost_1 + inner_prod_1
#     final_cost_2 = val_2 + ins_cost_2 + inner_prod_2
#
#     objective = p_u1 * final_cost_1 + p_u2 * final_cost_2
#
#     # objective = return_objective(input_1, input_2, val_1, val_2, p_u1, p_u2, grad_v1, grad_v2, u1, v1, u2, v2)
#
#     return objective.reshape(())
from odeintw import odeintw
from functools import partial

@jit
def dPdt(P, t, A, B, Q, R, S, ):
    n = A.shape[0]
    m = B.shape[1]

    if S is None:
        S = jnp.zeros((n, m))

    return -(A.T @ P + P @ A - (P @ B + S) @ jnp.linalg.inv(R) @ (B.T @ P + S.T) + Q)

@jit
def dPhi(Phi, t, A):
    return jnp.dot(A, Phi)


class analytical_solver():
    def __init__(self, t):
        total_steps = 10

        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

        Q = np.zeros((4, 4))

        R1 = np.array([[0.05, 0], [0, 0.025]])

        PT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        tspan = np.linspace(0, 1, total_steps + 1)
        tspan = np.flip(tspan)
        K1 = odeintw(dPdt, PT, tspan, args=(A, B, Q, R1, None,))

        self.K1 = np.flip(K1, axis=0)

        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        t_span = np.linspace(0, 1, total_steps + 1)
        t_span = np.flip(t_span)
        PhiT = np.eye(4)

        Phi_sol = odeintw(dPhi, PhiT, t_span, args=(A,))
        self.Phi_sol = np.flip(Phi_sol, axis=0)

        self.z = np.array([[0], [1], [0], [0]])
        # d1 = utils.d(Phi_sol, K1, B, R1, z)
        B2 = B

        R2 = np.array([[0.05, 0], [0, 0.1]])
        K2 = odeintw(dPdt, PT, tspan, args=(A, B2, Q, R2, None,))
        self.K2 = np.flip(K2, axis=0)
        self.index = int(np.round(1 - t, 2) * 10)

    @partial(jit, static_argnums=(0, ))
    def get_analytical_u(self, x, p, t):
        total_steps = 10
        R1 = jnp.array([[0.05, 0], [0, 0.025]])

        R2 = jnp.array([[0.05, 0], [0, 0.1]])


        ztheta = jnp.array([0, 1, 0, 0]) * (2 * p.reshape(-1, 1) - 1)

        # p1
        x1 = x[:4]
        x2 = x[4:8]
        K = jnp.array(self.K1[self.index, :, :])
        Phi = jnp.array(self.Phi_sol[self.index, :, :])

        B = jnp.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        u = -jnp.linalg.inv(R1) @ B.T @ K @ x1 + jnp.linalg.inv(R1) @ B.T @ K @ Phi @ (ztheta).reshape(-1, 1)

        # p2
        K = jnp.array(self.K2[self.index, :, :])
        B = jnp.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        v = -jnp.linalg.inv(R2) @ B.T @ K @ x2 + jnp.linalg.inv(R2) @ B.T @ K @ Phi @ (ztheta).reshape(-1, 1)

        return u.reshape(-1, ), v.reshape(-1, )
        

@jit
def corner_dynamics(x, u1, u2, v, dt=0.25):

    # for repulsive forces
    k = 10
    # 2 attackers 1 defender
    a1 = x[:4]
    a2 = x[4:8]
    d = x[8:]

    # x
    x_a1 = a1[:2]
    x_a2 = a2[:2]
    x_d = d[:2]
    # xdot
    x_a1_dot = a1[2:]
    x_a2_dot = a2[2:]
    x_d_dot = d[2:]

    eps = 1e-12

    # forces
    #phi = lambda x1, x2: jnp.exp(-(jnp.linalg.norm(x1 - x2 + eps)/0.05)**2)

    # print(phi(x_a1, x_a2))
    #F_a1 = k * (phi(x_a1, x_a2) * (x_a2 - x_a1) + phi(x_a1, x_d) * (x_d - x_a1))
    #F_a2 = k * (phi(x_a2, x_a1) * (x_a1 - x_a2) + phi(x_a2, x_d) * (x_d - x_a2))
    #F_d = k * (phi(x_d, x_a1) * (x_a1 - x_d) + phi(x_d, x_a2) * (x_a2 - x_d))
    
    # forces
    phi = lambda x1, x2: jnp.exp(-(jnp.linalg.norm(x1[:2] + dt * x1[2:]  - x2[:2] - dt * x2[2:] + eps)/0.05)**2)

    F_a1 = k * (phi(a1, a2) * (x_a2_dot - x_a1_dot) + phi(a1, d) * (x_d_dot - x_a1_dot))
    F_a2 = k * (phi(a2, a1) * (x_a1_dot - x_a2_dot) + phi(a2, d) * (x_d_dot - x_a2_dot))
    F_d = k * (phi(d, a1) * (x_a1_dot - x_d_dot) + phi(d, a2) * (x_a2_dot - x_d_dot))


    v_a1_dot = F_a1 + u1
    v_a2_dot = F_a2 + u2
    v_d_dot =  F_d + v 

    # next states
    x_a1_next = x_a1 + x_a1_dot * dt + (1/2) * v_a1_dot * dt ** 2
    x_a2_next = x_a2 + x_a2_dot * dt + (1/2) * v_a2_dot * dt ** 2
    x_d_next = x_d + x_d_dot * dt + (1/2) * v_d_dot * dt ** 2

    v_a1_next = x_a1_dot + v_a1_dot * dt
    v_a2_next = x_a2_dot + v_a2_dot * dt
    v_d_next = x_d_dot + v_d_dot * dt

    # return jnp.vstack((x_a1_dot, v_a1_dot, x_a2_dot, v_a2_dot, x_d_dot, v_d_dot)).reshape(-1, )

    return jnp.concatenate((x_a1_next, v_a1_next, x_a2_next, v_a2_next, x_d_next, v_d_next))
    
@jit
def smooth_min(a):
    alpha = 1
    return jnp.sum(a * jnp.exp(-alpha * a))/jnp.sum(jnp.exp(-alpha * a))

@jit
def corner_final_cost(x, p):
    x1 = x[:2] # attacker 1
    x2 = x[4:6] # attacker 2
    d = x[8:10] # defender
    g1 = jnp.array((0.75, 0.5))
    g2 = jnp.array((0.75, -0.5))

    dist1 = smooth_min(jnp.array([jnp.linalg.norm(x1 - g1) ** 2, jnp.linalg.norm(x2 - g1) ** 2]))
    dist2 = smooth_min(jnp.array([jnp.linalg.norm(x1 - g2) ** 2, jnp.linalg.norm(x2 - g2) ** 2]))
    
    #dist1 = jnp.minimum(jnp.linalg.norm(x1 - g1) ** 2, jnp.linalg.norm(x2 - g1) ** 2)
    #dist2 = jnp.minimum(jnp.linalg.norm(x1 - g2) ** 2, jnp.linalg.norm(x2 - g2) ** 2)

    dist1_p2 = jnp.linalg.norm(d - g1) ** 2
    dist2_p2 = jnp.linalg.norm(d - g2) ** 2

    return p * dist1 + (1 - p) * dist2 - (p * dist1_p2 + (1 - p) * dist2_p2)

@jit
def running_cost_corner(u1, u2, v):
    R1 = jnp.array([[0.01, 0.],
                    [0., 0.005]])

    R2 = jnp.array([[0.01, 0],
                    [0., 0.02]])

    loss_a1 = jnp.sum(jnp.multiply(jnp.diag(R1), u1 ** 2), axis=-1)
    loss_a2 = jnp.sum(jnp.multiply(jnp.diag(R1), u2 ** 2), axis=-1)
    loss_d = jnp.sum(jnp.multiply(jnp.diag(R2), v ** 2), axis=-1)

    return loss_a1 + loss_a2 - loss_d
    
@jit
def normalize_to_max_corner_1d(x, v_max):
    # assume same vmax
    a1 = jnp.clip(x[:2], -1, 1)
    a2 = jnp.clip(x[4:6], -1, 1)
    d = jnp.clip(x[8:10], -1, 1)

    p = x[12:]


    va1_x = x[2]
    va1_y = x[3]

    va2_x = x[6]
    va2_y = x[7]

    vd_x = x[10]
    vd_y = x[11]


    a = -1
    b = 1

    va1_x_b = -1 + (b - a) * (va1_x + v_max) / (v_max + v_max)
    va1_y_b = -1 + (b - a) * (va1_y + v_max) / (v_max + v_max)

    va2_x_b = -1 + (b - a) * (va2_x + v_max) / (v_max + v_max)
    va2_y_b = -1 + (b - a) * (va2_y + v_max) / (v_max + v_max)

    vd_x_b = -1 + (b - a) * (vd_x + v_max) / (v_max + v_max)
    vd_y_b = -1 + (b - a) * (vd_y + v_max) / (v_max + v_max)

    x_norm = jnp.concatenate((a1, va1_x_b.reshape(-1, ), va1_y_b.reshape(-1, ),
                              a2, va2_x_b.reshape(-1, ), va2_y_b.reshape(-1, ),
                              d, vd_x_b.reshape(-1, ), vd_y_b.reshape(-1, ),
                              p.reshape(-1, )))

    return x_norm

@jit
def normalize_to_max_corner_final(x, v_max):
    # assume same vmax

    a1 = jnp.clip(x[:2], -1, 1)
    a2 = jnp.clip(x[4:6], -1, 1)
    d = jnp.clip(x[8:10], -1, 1)


    va1_x = x[2]
    va1_y = x[3]

    va2_x = x[6]
    va2_y = x[7]

    vd_x = x[10]
    vd_y = x[11]


    a = -1
    b = 1

    va1_x_b = -1 + (b - a) * (va1_x + v_max) / (v_max + v_max)
    va1_y_b = -1 + (b - a) * (va1_y + v_max) / (v_max + v_max)

    va2_x_b = -1 + (b - a) * (va2_x + v_max) / (v_max + v_max)
    va2_y_b = -1 + (b - a) * (va2_y + v_max) / (v_max + v_max)

    vd_x_b = -1 + (b - a) * (vd_x + v_max) / (v_max + v_max)
    vd_y_b = -1 + (b - a) * (vd_y + v_max) / (v_max + v_max)

    x_norm = jnp.concatenate((a1, va1_x_b.reshape(-1, ), va1_y_b.reshape(-1, ),
                              a2, va2_x_b.reshape(-1, ), va2_y_b.reshape(-1, ),
                              d, vd_x_b.reshape(-1, ), vd_y_b.reshape(-1, )))

    return x_norm

@jit
def corner_final_cost_soft_roles(x, p, alpha=10.0):
    import flax
    """
     Compute the cost with soft dynamic role assignment based on proximity.

     Parameters:
     - x: Position vector [x1, y1, unused, unused, x2, y2, unused, unused, d_x, d_y]
     - p: Probability that the actual goal is g1
     - alpha: Sharpness parameter for soft assignment

     Returns:
     - cost: Scalar cost value
     """
    x1 = x[:2]
    x2 = x[4:6]
    d = x[8:10]

    g1 = jnp.array([0.75, 0.5])
    g2 = jnp.array([0.75, -0.5])

    # Compute distances
    dist_a1_def = jnp.linalg.norm(x1 - d)**2
    dist_a2_def = jnp.linalg.norm(x2 - d)**2

    # Softmax for role assignment
    logits_blocker = jnp.array([dist_a2_def - dist_a1_def]) * alpha  # Higher alpha sharpens the assignment
    weight_a1_blocker = flax.linen.sigmoid(logits_blocker)
    weight_a2_blocker = 1 - weight_a1_blocker

    # Compute blocker distance to defender
    blocker_dist_defender = weight_a1_blocker * dist_a1_def + weight_a2_blocker * dist_a2_def

    # Compute attacker distances to goals
    attacker_g1_dist = weight_a1_blocker * jnp.linalg.norm(x2 - g1)**2 + weight_a2_blocker * jnp.linalg.norm(x1 - g1)**2
    attacker_g2_dist = weight_a1_blocker * jnp.linalg.norm(x2 - g2)**2 + weight_a2_blocker * jnp.linalg.norm(x1 - g2)**2

    # Compute cost
    cost = (
            p * (attacker_g1_dist - jnp.linalg.norm(d - g1)**2) +
            (1 - p) * (attacker_g2_dist - jnp.linalg.norm(d - g2)**2) +
            blocker_dist_defender
    )

    return cost