# Test script that runs all algorithms on a single game for 10k steps (without exploitability computation).

# PPO
echo "\nTEST PPO\n"
python main.py algorithm=ppo game=classical_phantom_ttt seed=1 group_name=test max_steps=5000 \
algorithm.learning_rate=0.001 algorithm.num_steps=128 algorithm.gamma=3.96 algorithm.gae_lambda=15.2 algorithm.num_minibatches=16 \
algorithm.update_epochs=1 algorithm.clip_coef=0.025 algorithm.ent_coef=0.02 algorithm.vf_coef=0.5 algorithm.max_grad_norm=0.125 \
compute_exploitability=False 2>&1 | tee tests/ppo_test.txt

# MMD
echo "\nTEST MMD\n"
python main.py algorithm=mmd game=classical_phantom_ttt seed=1 group_name=test max_steps=5000 \
algorithm.learning_rate=0.001 algorithm.num_steps=128 algorithm.gamma=3.96 algorithm.gae_lambda=15.2 algorithm.num_minibatches=16 \
algorithm.update_epochs=1 algorithm.clip_coef=0.025 algorithm.ent_coef=0.02 algorithm.vf_coef=0.5 algorithm.max_grad_norm=0.125 \
algorithm.kl_coef=0.8 \
compute_exploitability=False  2>&1 | tee tests/mmd_test.txt

# PPG
echo "\nTEST PPG\n"
python main.py algorithm=ppg game=classical_phantom_ttt seed=1 group_name=test max_steps=5000 \
algorithm.learning_rate=0.001 algorithm.num_steps=128 algorithm.gamma=3.96 algorithm.gae_lambda=15.2 algorithm.num_minibatches=16 \
algorithm.update_epochs=1 algorithm.clip_coef=0.025 algorithm.ent_coef=0.02 algorithm.vf_coef=0.5 algorithm.max_grad_norm=0.125 \
algorithm.n_iteration=2 algorithm.e_policy=4 algorithm.v_value=16 algorithm.e_auxiliary=12 algorithm.beta_clone=1.0 \
algorithm.num_aux_rollouts=32 algorithm.n_aux_grad_accum=16 \
compute_exploitability=False  2>&1 | tee tests/ppg_test.txt

# NFSP
echo "\nTEST NFSP\n"
python main.py algorithm=nfsp game=classical_phantom_ttt seed=1 group_name=test max_steps=5000 \
algorithm.inner_rl_agent.replay_buffer_capacity=800000 algorithm.reservoir_buffer_capacity=250000 algorithm.min_buffer_size_to_learn=2000 algorithm.anticipatory_param=0.0625 algorithm.batch_size=1024 \
algorithm.learn_every=1024 algorithm.sl_learning_rate=0.000625 algorithm.inner_rl_agent.learning_rate=0.001 algorithm.inner_rl_agent.update_target_network_every=250 algorithm.inner_rl_agent.epsilon_decay_duration=10000000 \
algorithm.inner_rl_agent.epsilon_start=0.12 algorithm.inner_rl_agent.epsilon_end=0.004 \
compute_exploitability=False  2>&1 | tee tests/nfsp_test.txt

# PSRO
echo "\nTEST PSRO\n"
python main.py algorithm=psro game=classical_phantom_ttt seed=1 group_name=test max_steps=5000 \
algorithm.sims_per_entry=1600 algorithm.number_training_episodes=250 algorithm.inner_rl_agent.batch_size=256 algorithm.inner_rl_agent.learning_rate=0.04 algorithm.inner_rl_agent.update_target_network_every=250 \
algorithm.inner_rl_agent.learn_every=1 \
compute_exploitability=False  2>&1 | tee tests/psro_test.txt

# RNAD
echo "\nTEST RNAD\n"
python main.py algorithm=rnad game=classical_phantom_ttt seed=1 group_name=test max_steps=5000 \
algorithm.batch_size=2048 algorithm.learning_rate=0.0002 algorithm.clip_gradient=2500 algorithm.target_network_avg=0.0005 algorithm.eta_reward_transform=0.4 \
algorithm.c_vtrace=0.125 \
compute_exploitability=False  2>&1 | tee tests/rnad_test.txt

# ESCHER
echo "\nTEST ESCHER\n"
python main.py algorithm=escher game=classical_phantom_ttt seed=1 group_name=test max_steps=100 \
algorithm.num_traversals=12 algorithm.num_val_fn_traversals=6 algorithm.regret_train_steps=50 algorithm.val_train_steps=3 \
algorithm.policy_net_train_steps=200 algorithm.batch_size_regret=2048 algorithm.batch_size_val=4096 \
compute_exploitability=False 2>&1  | tee tests/escher_test.txt

# compute_exploitability=True
echo "\nTEST EXPLOITABILITY\n"
python main.py algorithm=ppo game=classical_phantom_ttt seed=1 group_name=test max_steps=5000 \
algorithm.learning_rate=0.001 algorithm.num_steps=128 algorithm.gamma=3.96 algorithm.gae_lambda=15.2 algorithm.num_minibatches=16 \
algorithm.update_epochs=1 algorithm.clip_coef=0.025 algorithm.ent_coef=0.02 algorithm.vf_coef=0.5 algorithm.max_grad_norm=0.125 \
compute_exploitability=True 2>&1 | tee tests/expl_test.txt
