import flax.serialization
import jax
import optax
from flax.training import train_state
from jax import numpy as jnp
from copy import deepcopy
from flax.training import checkpoints
from functools import partial
from tqdm import tqdm
import os
# import orbax.checkpoint
# from flax.training import orbax_utils
# from flax.serialization import to_state_dict
from utils_jax import cond_mkdir

def intial_cvx(old_params):
    params = deepcopy(old_params)  # not sure if need to stop gradient here
    for key in params['params'].keys():
        if 'cvx_layer' in key:
            params['params'][key]['kernel'] = jnp.abs(params['params'][key]['kernel'])

    return params

def make_cvx(old_params):
    params = deepcopy(old_params)  # not sure if need to stop gradient here
    for key in params['params'].keys():
        if 'cvx_layer' in key:
            params['params'][key]['kernel'] = params['params'][key]['kernel'].clip(0)

    return params


def create_train_step(key, model, config, optimizer, batch_size):
    params = model.init(key, jnp.zeros((batch_size, config.in_features)))
    params = intial_cvx(params)
    opt_state = optimizer.init(params)

    def loss_fn(params, x, gt, key):
        # reduce_dims = list(range(1, len(x.shape)))
        model_out = model.apply(params, x)
        mse_loss = optax.l2_loss(model_out, gt).mean()

        return mse_loss

    @jax.jit
    def train_step(params, opt_state, x, gt, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, gt, key)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        params = make_cvx(params)

        return params, opt_state, loss

    return train_step, params, opt_state


def train(model, config, optimizer, num_epoch, train_dataloader, model_dir, key):
    key, model_key = jax.random.split(key)

    train_step, params, opt_state = create_train_step(model_key, model, config, optimizer, len(train_dataloader))

    freq = 1000
    total_steps = 0
    train_losses = []
    with tqdm(total=len(train_dataloader) * num_epoch) as pbar:
        for epoch in range(num_epoch):
            for i, (batch, gt) in enumerate(train_dataloader):
                key, subkey = jax.random.split(key)

                batch = batch.reshape(len(batch), config.in_features)
                params, opt_state, loss = train_step(params, opt_state, batch, gt, subkey)

                train_losses.append(loss)

                total_steps += 1
                pbar.update(1)
                if not total_steps % freq:
                    print(f"epoch {epoch} | loss: {loss:.6f}")

    ckpt_dir = os.path.abspath(model_dir)

    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=params, step=0)
