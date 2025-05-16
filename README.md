### Supplementary Code for "2p0s Differential Games with One-Sided Payoff Information and Continuous Actions".


1. setup the conda environment using the file `env.yml`

2. Navigate to `our_method/visualization_scipts/` to use existing trained models (using CAMS) to generate trajectories:
    - `simulation_latest_for_gt_comparison` simulates the trajectories for the 4-stage game
    - `simulation_latest.py` simulates the unconstrained case
    - `simulation_latest_primal_dual.py` simulates the unconstrained case with both primal and dual policies
    - `simulation_latest_cons.py` simulates the constrained case
    - `simulation_latest_cons_primal_dual.py` simulates the constrained case with both primal and dual policies

3. Navigate to `our_method/` to train the value network for different cases (CAMS) -- unconstrained, contrained and their dual versions and the 3d case.
    - run `./train_our_method.sh` to train the primal unconstrained case
    - run `./train_our_method_for_cfr.sh` to train the 4-stage comparison case (against DeepCFR and JPSPG)
    - run `./train_our_method_dual.sh` to train the dual unconstrained case
    - run `./train_our_method_cons.sh` to train the primal constrained case
    - run `./train_our_method_cons_dual.sh` to train the dual constrained case
    - run `./train_our_method_3d.sh` to train the primal high dimensional case
4. To train deep cfr policy networks, run `run_cfr_3.py` for $|A|=9$, and `run_cfr` for $|A|=16$.
5. To train JPSPG Policies, navigate to `JPSPG/` and run `...incomplete_2d.py` for the normal-form game and `...2d_multi_stage.py` for 4-stage game. To visualize trajectories in the 4-stage game, run `JPSPG/trajectory_jpspg.py`, which uses pre-trained policies for P1 and P2. 
6. To use Multigrid method to train the value networks, Navigate to `Multi-Grid/`.
    - run `run_multigrid.py` to train value networks using 2-level multigrid.
    - run `run_multigrid_n_cycle.py` to train value networks using n-level multigrid. Level can be specified with arguments `--kmax` and `--kmin`. Note `kmin >= 1`.
7. To train the SOTA RL Models on the Hexner's game, navigate to `IIG-RL-Benchmark` and run `diff_game_main_exp.py`, which uses [`IIG-RL-Benchmark`](https://github.com/nathanlct/IIG-RL-Benchmark) repo. The environment for the Hexner's game is in `open_spiel_games`. To change the time discretization and action space, go to `open_spiel_games/hexners_game_fixed_state.py` and make the necessary changes. For exploitability, openspiel's native algorithm is used. The eval scripts are in `SOTA_Test_Scripts`.
8. To train CAMS-DRL models, navigate to `CAMS-RL` and run `PPO_pub_belief_trainer.py` to train the PPO agent and `MMD_pub_belief_trainer.py` to train the MMD agent. For evaluation, run `eval_game_pub_belief.py`
