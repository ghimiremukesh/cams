import os
import pdb
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from utils import util_funcs
from dsgda_solver import dsgda_solver, dsgda_solver_final, coarse_solvers_adaptive
import jax_dataloader as jdl
import training
from tqdm import tqdm

EPSILON = 1e-6
A_MAX = 12

def sample_states(t, num_points):
    # vel_bound = util_funcs.compute_bounds(t, 12)
    x = np.random.uniform(-1, 1, size=((num_points, 8)))

    if t == 0:
        x[:, 2:4] = 0
        x[:, 6:8] = 0

    p = np.random.uniform(EPSILON, 1-EPSILON, size=((num_points, 1)))

    X = np.hstack((x, p))

    # unnormalize
    # X = util_funcs.unnormalize_states(X, vel_bound, vel_bound, vel_bound, vel_bound)

    return jnp.array(X)


def restrict_residuals(coarse_steps, residuals):
    fine_steps = len(residuals)
    coarse_residuals = defaultdict()
    for t in range(coarse_steps):
        res_H = jnp.add(residuals[2*t], residuals[2*t+1])
        res_H = res_H/2
        coarse_residuals[t] = res_H

    return coarse_residuals

def restrict_value_fn(val_model, coarse_steps, fine_params):
    v_Hs = defaultdict()
    for t in range(coarse_steps):
        fun = lambda X1, X2: (val_model.apply(fine_params[2*t], X1) + val_model.apply(fine_params[2*t+1], X2))/2
        v_Hs[t] = fun

    return v_Hs

def augment_correction_fn(val_model, restricted_val_fn, corr_param):
    ve_H = lambda X1, X2: restricted_val_fn(X1, X2) + val_model.apply(corr_param, X1)

    return ve_H


def get_smoothing_data(coords, policies, val_model, model_params, t, tau, fine_iters):
    """
    Implement post smoothing by collecting data with few dsgda iterations

    :param policies: initial solution for dsgda solver
    :param val_model: value function nn
    :param model_params: value function parameters if the next time-step is not final time
    :param t: current time-step
    :param tau: time discretization
    :param fine_iters: number of dsgda iterations

    :return: dataset (x, v)
    """
    # sample states
    X = coords
    vel_bound = util_funcs.compute_bounds(t, A_MAX)
    X_unnorm = util_funcs.unnormalize_states(X, vel_bound, vel_bound, vel_bound, vel_bound)

    if model_params is None:
        val, policy = dsgda_solver_final(X_unnorm, tau, init_policies=policies, iters=fine_iters)
    else:
        val, policy = dsgda_solver(val_model, model_params, X_unnorm, t, tau, init_policies=policies,
                                   iters=fine_iters)

    dataset = jdl.ArrayDataset(X, val.reshape(-1, 1))

    return dataset, policy



class MultigridFAS:
    def __init__(self, val_model, optim, kmax=4, kmin=1,seed=0, batch_size=10000, fine_iters=1000):
        self.kmax = kmax
        self.kmin = kmin
        self.hs = [2 ** (-k) for k in range(kmax, kmin, -1)]
        self.H = 2 ** (-kmin)
        self.val_model = val_model
        self.batch_size = batch_size
        self.t_H = np.arange(0, 1, self.H)
        self.seed = seed
        self.fine_iters = fine_iters
        self.optim = optim
        # define models for value functions for fine grid and correction functions for coarse grid
        key = jax.random.PRNGKey(seed)
        dummy_input = jnp.zeros((batch_size, val_model.config.in_features))
        self.Vh_params = defaultdict(list)
        # init value nn params for all grid sizes
        for h in self.hs:
            self.Vh_params[h] = [val_model.init(key, dummy_input) for _ in range(int(1/h))]

        # init coarse network params for the coarsest grid
        self.epsilon_H_params = [val_model.init(key, dummy_input) for _ in range(int(1/self.H))]
        self.policies = defaultdict(lambda: defaultdict(list))
        self.residuals = defaultdict(lambda: defaultdict(list))
        self.coarse_corr = defaultdict()
        self.fine_corrs = defaultdict(lambda: defaultdict(list))
        self.X = None
        self.sum_residuals = 0

    def cycle(self):
        """
        Implement a V-cycle with Full Approximation Scheme (FAS)

        Updates the value functions at fine grid
        """
        # sample states and belief for each time-step
        self.X = [sample_states(t, self.batch_size) for t in range(int(2**self.kmax))]
        # down cycle
        # compute target and residual for value function at each time step at each grid level
        for i in range(len(self.hs)):
            h = self.hs[i]
            val_fn_params = self.Vh_params[h]
            t_h = np.arange(0, 1, h)
            fine_steps = int(1/h)
            for t in range(fine_steps):
                curr_val = self.val_model.apply(val_fn_params[t], self.X[t])
                vel_bound = util_funcs.compute_bounds(t_h[t], A_MAX)
                X_unnorm = util_funcs.unnormalize_states(self.X[t], vel_bound, vel_bound, vel_bound, vel_bound)
                if t == fine_steps - 1:
                    target, policy = dsgda_solver_final(X_unnorm, h, iters=self.fine_iters)
                else:
                    target, policy = dsgda_solver(val_model=self.val_model,
                                                  model_params=val_fn_params[t+1],
                                                  coords=X_unnorm,
                                                  t=t_h[t],
                                                  tau=h,
                                                  init_policies=self.policies[h][t] if t in self.policies else None,
                                                  iters=self.fine_iters)
                self.policies[h][t] = policy
                self.residuals[h][t] = curr_val.reshape(-1, ) - target

            if i != 0: # if not the finest grid then transfer residual from previous finer grid
                restricted_res = restrict_residuals(fine_steps, self.residuals[self.hs[i-1]])
                # add the restricted residual
                for key in self.residuals[h].keys():
                    self.residuals[h][key] += restricted_res[key]

        # coarse solve
        # restrict residual to coarse grid
        coarse_residuals = restrict_residuals(len(self.t_H), self.residuals[self.hs[-1]])  # a dictionary
        v_Hs = restrict_value_fn(self.val_model, len(self.t_H), self.Vh_params[self.hs[-1]])
        for t in reversed(range(len(self.t_H))):
            if t == len(self.t_H) - 1:
                correction = -coarse_residuals[t]
            else:
                augmented_val_fn = augment_correction_fn(self.val_model, v_Hs[t+1], self.epsilon_H_params[t+1])
                correction = coarse_solvers_adaptive(augmented_val_fn, v_Hs[t+1], self.X[t], t, self.H, self.hs[-1],
                                            init_policies=self.policies[self.hs[-1]][2*t])
                correction -= coarse_residuals[t]

            self.coarse_corr[t] = correction.reshape(-1, 1)
            # TODO: train coarse correction model and store the params in the dictionary
            dataset = jdl.ArrayDataset(self.X[t], correction.reshape(-1, 1))
            dataloader = jdl.DataLoader(dataset, backend='jax', batch_size=256, shuffle=True)
            curr_params = training.train(model=self.val_model,
                                         model_params=self.epsilon_H_params[t],
                                         config=self.val_model.config,
                                         optimizer=self.optim,
                                         num_epoch=10,
                                         train_dataloader=dataloader,
                                         key=jax.random.PRNGKey(self.seed))
            self.epsilon_H_params[t] = curr_params

        # Up-cycle
        # Prolong coarsest correction to fine correction then prolong to the rest
        for t in range(0, len(self.t_H)):
            if t == len(self.t_H) - 1:
                self.fine_corrs[self.hs[-1]][2*t+1] = -self.residuals[self.hs[-1]][2*t+1].reshape(-1, 1)

            self.fine_corrs[self.hs[-1]][2*t] = self.coarse_corr[t]
            if 2*t-1 > 0:
                self.fine_corrs[self.hs[-1]][2*t-1] = self.coarse_corr[t]

        # prolong correction from finer to finest grid
        for i in reversed(range(len(self.hs)-1)):
            h = self.hs[i+1]
            fine_steps = int(1/h)
            coarse_level_corr = self.fine_corrs[self.hs[i+1]]
            for t in range(0, fine_steps):
                if t == fine_steps - 1:
                    self.fine_corrs[self.hs[i]][2*t+1] = -self.residuals[self.hs[i]][2*t+1].reshape(-1, 1)

                self.fine_corrs[self.hs[i]][2*t] = coarse_level_corr[t]
                if 2*t-1 > 0:
                    self.fine_corrs[self.hs[i]][2*t-1] = coarse_level_corr[t]

        # update based on corrections
        # Fit the fine value networks to the correction
        for i in range(len(self.hs)):
            h = self.hs[i]
            val_fn_params = self.Vh_params[h]
            fine_steps = int(1/h)
            fine_corr = self.fine_corrs[h]
            for t in range(fine_steps-1):
                curr_params = val_fn_params[t]
                # pdb.set_trace()
                curr_target = self.val_model.apply(curr_params, self.X[t]) + fine_corr[t]
                dataset = jdl.ArrayDataset(self.X[t], curr_target)
                dataloader = jdl.DataLoader(dataset, backend='jax', batch_size=256, shuffle=True)
                new_params = training.train(model=self.val_model,
                                            model_params=curr_params,
                                            config=self.val_model.config,
                                            optimizer=self.optim,
                                            num_epoch=10,
                                            train_dataloader=dataloader,
                                            key=jax.random.PRNGKey(self.seed))
                self.Vh_params[h][t] = new_params

        # TODO: Smoothing -- Fit the value networks to a new minimax solution (initialized with the stored policies)
        #  backward in time
        for i in range(len(self.hs)):
            h = self.hs[i]
            val_fn_params = self.Vh_params[h]
            t_h = np.arange(0, 1, h)
            fine_steps = int(1/h)
            for t in reversed(range(fine_steps)):
                # TODO: write a function that collects data with few iterations of dsgda
                dataset, policy = get_smoothing_data(self.X[t],
                                             self.policies[h][t],
                                             self.val_model,
                                             val_fn_params[t+1] if t+1 < fine_steps else None,
                                             t_h[t],
                                             h,
                                             self.fine_iters)
                self.policies[h][t] = policy # update policy
                dataloader = jdl.DataLoader(dataset, backend='jax', batch_size=256, shuffle=True)
                new_params = training.train(model=self.val_model,
                                            model_params=val_fn_params[t],
                                            config=self.val_model.config,
                                            optimizer=self.optim,
                                            num_epoch=10,
                                            train_dataloader=dataloader,
                                            key=jax.random.PRNGKey(self.seed))
                self.Vh_params[h][t] = new_params

        # compute sum residuals
        # self.sum_residuals = jnp.linalg.norm(jnp.array([*self.residuals.values()]))
        self.sum_residuals = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(self.residuals)))

        return self.sum_residuals


    def run_vcycle(self, num_iters):
        residuals = []

        with tqdm(total=num_iters) as pbar:
            for _ in range(num_iters):
                curr_res = self.cycle()
                residuals.append(curr_res)
                print(f'Residual: {curr_res:.3f}')
                pbar.update(1)












