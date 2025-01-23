from multi_grid_dual import MultigridFAS
from utils import nn_modules
import optax
from flax.training import checkpoints
import flax
import os, shutil

flax.config.update('flax_use_orbax_checkpointing', False)


if __name__=='__main__':
    model_config = nn_modules.NoTimeDualConfig
    val_model = nn_modules.PICNN_Dual(model_config)
    optim = optax.adam(learning_rate=1e-4)

    log_dir = 'dual/logs_4stage_5k_150iters/'
    
    if os.path.exists(log_dir):
        if input(f'Directory "{log_dir}" exists. Delete (Y/N)? ').lower() == 'y':
            shutil.rmtree(log_dir)
        else:
            raise OSError("Delete or set new log directory!")
    
    os.makedirs(log_dir)



    multigrid = MultigridFAS(val_model=val_model, optim=optim, h=0.25, H=0.5, fine_steps=4, coarse_steps=2, seed=0, batch_size=5000, fine_iters=5000)

    multigrid.run_vcycle(num_iters=150)

    # save fine network params
    for t in range(multigrid.fine_steps):
        checkpoints.save_checkpoint(ckpt_dir=log_dir, target=multigrid.Vh_params[t], step=t, keep=multigrid.fine_steps, overwrite=False)

    print('Model Params Saved!')
    print('End Training')
    



