### Supplementary Code for the AAAI MARW Workshop"


1. setup the conda environment using the file `env.yml`

2. Navigate to `our_method/visualization_scipts/` to use existing trained models to generate trajectories:
    - `simulation_latest_for_gt_comparison` simulates the trajectories for the 4-stage game


3. Navigate to `our_method/` to train the value network for different cases -- unconstrained, contrained and their dual versions and the 3d case.
    - run `./train_our_method_for_cfr.sh` to train the comparison case (against DeepCFR)

4. To train deep cfr policy networks, run `run_cfr_3.py` for $|A|=9$, and `run_cfr` for $|A|=16$.
5. To compare our method with cfr, run the notebook `our_method/hexner_last_step-stopping.ipynb`
6. To generate trajectories using deepcfr, run the notebook `DeepCFR_Trajectory.ipynb`
