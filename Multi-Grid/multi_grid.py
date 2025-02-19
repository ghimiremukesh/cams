import os
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
import pdb

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
    def __init__(self, val_model, optim, h=0.1, H=0.2, fine_steps=10, coarse_steps=5, seed=0, batch_size=10000, fine_iters=10000):
        self.H = H # coarse time discretization
        self.h = h # fine time discretization
        self.fine_steps = fine_steps
        self.coarse_steps = coarse_steps
        self.val_model = val_model
        self.batch_size = batch_size
        self.t_h = np.arange(0, 1, h)
        self.t_H = np.arange(0, 1, H)
        self.seed = seed
        self.fine_iters = fine_iters
        self.optim = optim
        # define models for value functions for fine grid and correction functions for coarse grid
        key = jax.random.PRNGKey(seed)
        dummy_input = jnp.zeros((batch_size, val_model.config.in_features))
        self.Vh_params = [val_model.init(key, dummy_input) for _ in range(fine_steps)] # init. value nn params
        self.epsilon_H_params = [val_model.init(key, dummy_input) for _ in range(coarse_steps)] # init. coarse nn params
        self.policies = defaultdict()
        self.residuals = defaultdict()
        self.coarse_corr = defaultdict()
        self.fine_corr = defaultdict()
        self.X = None
        self.sum_residuals = 0

    def cycle(self):
        """
        Implement a V-cycle with Full Approximation Scheme (FAS)

        Updates the value functions at fine grid
        """
        # sample states and belief for each time-step
        self.X = {self.t_h[t]: sample_states(t, self.batch_size) for t in range(self.fine_steps)}

        # pdb.set_trace()
        # compute target and residual for value function at each time step
        for t in range(self.fine_steps):
            curr_val = self.val_model.apply(self.Vh_params[t], self.X[self.t_h[t]])
            vel_bound = util_funcs.compute_bounds(self.t_h[t], A_MAX)
            X_unnorm = util_funcs.unnormalize_states(self.X[self.t_h[t]], vel_bound, vel_bound, vel_bound, vel_bound)
            if t == self.fine_steps - 1:
                target, policy = dsgda_solver_final(X_unnorm, self.h, iters=60000) # can afford to train fully
            else:
                # pdb.set_trace()
                target, policy = dsgda_solver(val_model=self.val_model,
                                              model_params=self.Vh_params[t+1],
                                              coords=X_unnorm,
                                              t=self.t_h[t],
                                              tau=self.h,
                                              init_policies=self.policies[t] if t in self.policies else None,
                                              iters=self.fine_iters)
            self.policies[t] = policy
            self.residuals[t] = curr_val.reshape(-1, ) - target

        # restrict residual to coarse grid
        # pdb.set_trace()
        coarse_residuals = restrict_residuals(self.coarse_steps, self.residuals)  # a dictionary
        v_Hs = restrict_value_fn(self.val_model, self.coarse_steps, self.Vh_params)
        for t in reversed(range(self.coarse_steps)):
            # pdb.set_trace()
            if t == self.coarse_steps - 1:
                correction = -coarse_residuals[t]
            else:
                augmented_val_fn = augment_correction_fn(self.val_model, v_Hs[t+1], self.epsilon_H_params[t+1])
                # pdb.set_trace()
                correction = coarse_solvers_adaptive(augmented_val_fn, v_Hs[t+1], self.X[self.t_H[t]], t, self.H, self.h, init_policies=self.policies[2*t])
                correction -= coarse_residuals[t]

            self.coarse_corr[t] = correction.reshape(-1, 1)
            # train coarse correction model and store the params in the dictionary
            dataset = jdl.ArrayDataset(self.X[self.t_H[t]], correction.reshape(-1, 1))
            # pdb.set_trace()
            dataloader = jdl.DataLoader(dataset, backend='jax', batch_size=256, shuffle=True)
            curr_params = training.train(model=self.val_model,
                                         model_params=self.epsilon_H_params[t],
                                         config=self.val_model.config,
                                         optimizer=self.optim,
                                         num_epoch=10,
                                         train_dataloader=dataloader,
                                         key=jax.random.PRNGKey(self.seed))
            self.epsilon_H_params[t] = curr_params


        # Prolong coarse correction to fine correction
        # pdb.set_trace()
        for t in range(0, self.coarse_steps):
            self.fine_corr[2*t] = self.coarse_corr[t]
            if 2*t-1 > 0:
                self.fine_corr[2*t-1] = self.coarse_corr[t]


        # Fit the fine value networks to the correction
        for t in range(self.fine_steps-1):
            curr_params = self.Vh_params[t]
            # pdb.set_trace()
            curr_target = self.val_model.apply(curr_params, self.X[self.t_h[t]]) + self.fine_corr[t]
            dataset = jdl.ArrayDataset(self.X[self.t_h[t]], curr_target)
            dataloader = jdl.DataLoader(dataset, backend='jax', batch_size=256, shuffle=True)
            new_params = training.train(model=self.val_model,
                                        model_params=curr_params,
                                        config=self.val_model.config,
                                        optimizer=self.optim,
                                        num_epoch=10,
                                        train_dataloader=dataloader,
                                        key=jax.random.PRNGKey(self.seed))
            self.Vh_params[t] = new_params

        #  Smoothing -- Fit the value networks to a new minimax solution (initialized with the stored policies)
        #  backward in time
        for t in reversed(range(self.fine_steps)):
            # write a function that collects data with few iterations of dsgda
            dataset, policy = get_smoothing_data(self.X[self.t_h[t]],
                                         self.policies[t],
                                         self.val_model,
                                         self.Vh_params[t+1] if t+1 < self.fine_steps else None,
                                         self.t_h[t],
                                         self.h,
                                         self.fine_iters)
            self.policies[t] = policy # update policy
            dataloader = jdl.DataLoader(dataset, backend='jax', batch_size=256, shuffle=True)
            new_params = training.train(model=self.val_model,
                                        model_params=self.Vh_params[t],
                                        config=self.val_model.config,
                                        optimizer=self.optim,
                                        num_epoch=10,
                                        train_dataloader=dataloader,
                                        key=jax.random.PRNGKey(self.seed))
            self.Vh_params[t] = new_params

        # compute sum residuals
        self.sum_residuals = jnp.linalg.norm(jnp.array([*self.residuals.values()]))
        
        return self.sum_residuals

    def run_vcycle(self, num_iters):

        residuals = []

        with tqdm(total=num_iters) as pbar:
            for _ in range(num_iters):
                curr_res = self.cycle()
                residuals.append(curr_res)
                print(f'Residual: {curr_res:.3f}')
                pbar.update(1)
















