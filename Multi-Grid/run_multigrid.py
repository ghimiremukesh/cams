from multi_grid import MultigridFAS
from utils import nn_modules
import optax
from flax.training import checkpoints
import flax
import os, shutil

flax.config.update('flax_use_orbax_checkpointing', False)


if __name__=='__main__':
    model_config = nn_modules.NoTimeConfig
    val_model = nn_modules.PICNN(model_config)
    optim = optax.adam(learning_rate=1e-4)

    log_dir = 'logs_10stage_5k_200iters/'
    
    if os.path.exists(log_dir):
        if input(f'Directory "{log_dir}" exists. Delete (Y/N)? ').lower() == 'y':
            shutil.rmtree(log_dir)
        else:
            raise OSError("Delete or set new log directory!")
    
    os.makedirs(log_dir)



    multigrid = MultigridFAS(val_model=val_model, optim=optim, h=0.1, H=0.2, fine_steps=10, coarse_steps=5, seed=0, batch_size=5000, fine_iters=5000)

    multigrid.run_vcycle(num_iters=200)

    # save fine network params
    for t in range(multigrid.fine_steps):
        checkpoints.save_checkpoint(ckpt_dir=log_dir, target=multigrid.Vh_params[t], step=t, keep=multigrid.fine_steps, overwrite=False)

    print('Model Params Saved!')
    print('End Training')
    



