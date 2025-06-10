import jax.numpy as jnp



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



def get_GT(states, p, target, n, tau=0.1):
    """
    Given an state, belief, target, current time, and time-discretization, return the analytical solution at that time-step 
    
    states: (8, ) array of current state for both P1 and P2 (x1, y1, vx1, vy1, x2, y2, vx2, vy2)
    p: scalar belief 
    target: 1 for type-1 goal (0, 1) or 0 for type-2 goal (0, -1)
    t: current time-step (backward time)
    tau: time-discretization (default is 0.1)
    
    Returns: u (2, ) array of control input for P1 at time t
             v (2, ) array of control input for P2 at time t
    """

    N = int(1 / tau)  # Number of time steps

    A = jnp.eye(4) + jnp.array([[0, 0, tau, 0], [0, 0, 0, tau], [0, 0, 0, 0], [0, 0, 0, 0]])
    B = jnp.array([[0.5 * tau ** 2, 0], [0, 0.5 * tau ** 2], [tau, 0], [0, tau]])
    Qf = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    Q = jnp.zeros((4, 4))
    
    R1 = jnp.array([[0.05, 0], [0, 0.025]]) * tau
    R2 = jnp.array([[0.05, 0], [0, 0.1]]) * tau

    K1 = discrete_lqr(A, B, Q, R1, Qf, N)
    K2 = discrete_lqr(A, B, Q, R2, Qf, N)


    # assumes t_r = 0.5
    if n <= 5:
        p = 0 if target == 0 else 1
    
    x1 = states[:4]
    x2 = states[4:8]
    goal = jnp.array([0, 2 * p - 1, 0, 0]) * jnp.ones_like(p)
    u = -K1[-n] @ (x1 - goal).T
    v = -K2[-n] @ (x2 - goal).T

    return u, v
    

# use this to get the critical time index
# def get_tr(A, B, P1, P2, R1, R2, N=10):
#     """
#     Return the index of the critical time. 
#     """
#     def compute_d(A, B, P, R):
#         z = jnp.array([[0, 1, 0, 0]]).reshape(-1, 1)
#         d = z.T @ A @ P @ B @ jnp.linalg.inv(R) @ B.T @ P @ A @ z
        
#         return d

#     d1s = jnp.vstack([compute_d(A, B, P1[i], R1) for i in range(N+1)])
#     d2s = jnp.vstack([compute_d(A, B, P2[i], R2) for i in range(N+1)])

#     f_n = d1s - d2s

#     summation = jnp.array([sum(f_n.reshape(-1, )[:i]) for i in range(N+1)])
    
#     return jnp.argmin(summation)
    


if __name__ == "__main__":
    # Example usage
    states = jnp.array([-0.5, 0, 0, 0, 0.5, 0, 0, 0])
    p = 0.5
    target = 1
    n = 10  # initial time-step 
    tau = 0.1

    u, v = get_GT(states, p, target, n, tau)
    print("Control input for P1:", u)
    print("Control input for P2:", v)







