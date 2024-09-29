### Supplementary Code for ICLR Paper #3245 "Two-Player Zero-Sum Differential Games with One-Sided Information and Continuous Actions"


1. setup the conda environment using the file `env.yml`

2. Navigate to `our_method/visualization_scipts/` to use existing trained models to generate trajectories:
    - `simulation_latest_for_gt_comparison` simulates the trajectories for the 4-stage game
    - `simulation_latest.py` simulates the unconstrained case
    - `simulation_latest_primal_dual.py` simulates the unconstrained case with both primal and dual policies
    - `simulation_latest_cons.py` simulates the constrained case
    - `simulation_latest_cons_primal_dual.py` simulates the constrained case with both primal and dual policies

3. Navigate to `our_method/` to train the value network for different cases -- unconstrained, contrained and their dual versions and the 3d case.
    - run `./train_our_method.sh` to train the primal unconstrained case
    - run `./train_our_method_for_cfr.sh` to train the comparison case (against DeepCFR)
    - run `./train_our_method_dual.sh` to train the dual unconstrained case
    - run `./train_our_method_cons.sh` to train the primal constrained case
    - run `./train_our_method_cons_dual.sh` to train the dual constrained case
    - run `./train_our_method_3d.sh` to train the primal high dimensional case
4. To train deep cfr policy networks, run `run_cfr_3.py` for $|A|=9$, and `run_cfr` for $|A|=16$.
5. To compare our method with cfr, run the notebook `our_method/hexner_last_step-stopping.ipynb`
6. To generate trajectories using deepcfr, run the notebook `DeepCFR_Trajectory.ipynb`
