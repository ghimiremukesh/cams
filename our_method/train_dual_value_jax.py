"""
Script to train value function. The neural network uses a Partially Convex Neural Network (PICNN) architecture.

"""

import os
import sys

import configargparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flax_picnn_dual as picnn_dual
import numpy as np
import dataio
from torch.utils.data import DataLoader
import training_jax as training
import jax
import loss_functions
import optax
import os
import shutil
from utils_jax import numpy_collate

# jax.config.update("jax_disable_jit", True)


# parser
p = configargparse.ArgumentParser()
p.add_argument('--start', type=int, default=1, help='time-step to train')

opt = p.parse_args()

logging_root = f'logs/dual_value_more_data'
save_root = f'train_data/dual_value_more_data'

# key = jax.random.key(0)
# params = model.init(key, jnp.ones((config.in_features, )))

start = opt.start
dt = 0.1
t = np.arange(0, 1.1, dt)
ts = np.arange(dt, 1+dt, dt)

num_epochs = 80
lr = 1e-5

mat_files = [f'train_data_t_{dt:.2f}.mat' for dt in ts]

print(f'\n Training for timestep: t = {t[start]}')

# load dataset
dataset = dataio.JaxTrainLoader(os.path.join(save_root, mat_files[start - 1]))
train_dataloader = DataLoader(dataset, shuffle=True, batch_size=128, pin_memory=True, num_workers=0,
                              collate_fn=numpy_collate)


root_path = os.path.join(logging_root, f't_{start}/')

if os.path.exists(root_path):
    shutil.rmtree(root_path)

os.makedirs(root_path)

config = picnn_dual.ModelConfig
model = picnn_dual.PICNN(config)

optim = optax.adam(lr)
# training.train(model=model, optimizer=optim, params=params, train_dataloader=train_dataloader, num_epochs=num_epochs,
#                model_dir=root_path)
key = jax.random.PRNGKey(0)
training.train(model, config, optim, num_epochs, train_dataloader, model_dir=root_path, key=key)



