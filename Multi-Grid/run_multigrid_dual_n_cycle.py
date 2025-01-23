from multi_grid_dual_n_cycle import MultigridFAS
from utils import nn_modules
import optax
from flax.training import checkpoints
import flax
import os, shutil
import argparse
# import pdb

flax.config.update('flax_use_orbax_checkpointing', False)

p = argparse.ArgumentParser()

p.add_argument("--kmax", type=int, default=3)
p.add_argument("--kmin", type=int, default=1)

args = p.parse_args()

if __name__=='__main__':
    model_config = nn_modules.NoTimeDualConfig
    val_model = nn_modules.PICNN_Dual(model_config)
    optim = optax.adam(learning_rate=1e-4)

    log_dir = 'n_cycle/dual/4_cycle/logs_16stage_5k_150itrs/'
    
    if os.path.exists(log_dir):
        if input(f'Directory "{log_dir}" exists. Delete (Y/N)? ').lower() == 'y':
            shutil.rmtree(log_dir)
        else:
            raise OSError("Delete or set new log directory!")
    
    os.makedirs(log_dir)

    h_f = 2 ** (-args.kmax)
    finest_steps = 2 ** args.kmax
    

    multigrid = MultigridFAS(val_model=val_model, optim=optim, kmax=args.kmax, kmin=args.kmin, seed=0, batch_size=5000, fine_iters=1000)

    multigrid.run_vcycle(num_iters=150)

    # pdb.set_trace()
    # save fine network params
    for t in range(finest_steps):
        checkpoints.save_checkpoint(ckpt_dir=log_dir, target=multigrid.Vh_params[h_f][t], step=t, keep=finest_steps, overwrite=False)

    print('Model Params Saved!')
    print('End Training')
    



