import jax.random
import scipy.io as scio
import torch
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset, default_collate
import utils_jax_old
from jax.tree_util import tree_map

EPS = 1e-6

class TrainInterTime(Dataset):
    def __init__(self, matfile):
        self.data = scio.loadmat(matfile)
        self.coords = self.data['states']
        self.values = self.data['values']

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx, :]
        value = self.values[idx, :]

        # return {'coords': torch.tensor(coord, dtype=torch.float32)}, \
        #        {'values': torch.tensor(value, dtype=torch.float32)}
        return torch.tensor(coord, dtype=torch.float32), torch.tensor(value, dtype=torch.float32)

class JaxTrainLoader(Dataset):
    def __init__(self, matfile):
        self.data = scio.loadmat(matfile)
        self.coords = self.data['states']
        self.values = self.data['values']

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx, :]
        value = self.values[idx, :]

        return np.array(coord, dtype=jnp.float32), np.array(value, dtype=jnp.float32)


class PINNLoader(Dataset):
    def __init__(self, numpoints, t_min=0, t_max=1, a_min=-12, a_max=12, counter_start=0, counter_end=100e3,
                 pretrain=True, pretrain_iters=10000, num_src_samples=1000, seed=0):
        super().__init__()
        self.key = jax.random.PRNGKey(seed)
        self.numpoints = numpoints
        self.t_min = t_min
        self.t_max = t_max
        self.num_states = 4  # dim of joint state
        self.N_src_samples = num_src_samples

        self.pre_train = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.counter = counter_start
        self.full_count = counter_end
        self.a_min = a_min
        self.a_max = a_max

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.

        key1, key2, key3 = jax.random.split(self.key, 3)  # keys for sampling randomly

        # sample states
        pos = jax.random.uniform(key1, shape=(self.numpoints, self.num_states), minval=-1, maxval=1)
        vel = jax.random.uniform(key2, shape=(self.numpoints, self.num_states), minval=self.a_min,
                                 maxval=self.a_max)
        p = jax.random.uniform(key3, shape=(self.numpoints, 1), minval=EPS, maxval=1-EPS)

        states = jnp.concatenate((pos[:, :2], vel[:, :2], pos[:, 2:4], vel[:, 2:4], p), axis=1)

        boundary_values = jax.vmap(utils_jax_old.final_cost_function)(states, p)

        if self.pre_train:
            # only sample in time around initial condition
            time = jnp.ones((self.numpoints, 1)) * start_time
            coords = jnp.concatenate((time, states), axis=1)
        else:
            # slowly grow time
            key = jax.random.PRNGKey(1)
            time = self.t_min + jax.random.uniform(key, shape=(self.numpoints, 1), minval=0,
                                                   maxval=(self.t_max - self.t_min) * (self.counter / self.full_count))
            coords = jnp.concatenate((time, states), axis=1)

            # make sure we have training samples at the initial time
            coords = coords.at[-self.N_src_samples:, 0].set(start_time)

        if self.pre_train:
            dirichlet_mask = jnp.ones((coords.shape[0], 1)) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pre_train:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pre_train and self.pretrain_counter == self.pretrain_iters:
            self.pre_train = False
        
        if len(np.array(coords[:, 0][~dirichlet_mask.reshape(-1, )])) >= 1:
            dt = min(np.array(coords[:, 0][~dirichlet_mask.reshape(-1, )]))
        else:
            dt = 0.0
            
        dt = 0.05 if dt >= 0.05 else dt
        
      
        return np.array(coords), {'bc': np.array(boundary_values), 'mask': np.array(dirichlet_mask), 'dt': dt}
