### Supplementary Code for ICLR Paper #3245 "Two-Player Zero-Sum Differential Games with One-Sided Information and Continuous Actions"


1. setup the conda environment using the file `env.yml`
2. Navigate to `our_method/visualization_scipts/` to use existing trained models to generate trajectories
3. Navigate to `our_method/train_xyz` to train the value network for different cases -- unconstrained, contrained and their dual versions and the 3d case. 
4. To train deep cfr policy networks, run `run_cfr_3.py` for $|A|=9$, and `run_cfr` for $|A|=16$.
5. To compare our method with cfr, run the notebook `our_method/hexner_last_step-stopping.ipynb`
6. To generate trajectories using deepcfr, run the notebook `DeepCFR_Trajectory.ipynb`
