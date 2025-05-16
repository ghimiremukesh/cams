import torch.optim as optim
from algorithms.actor_critic_base_simplified import ActorCritic

class PPOAgent:
    def __init__(self, state_dim, belief_dim, action_hist_dim, action_dim, num_type, action_bounds,
                 has_private_type=True, hidden_sizes=[64], learning_rate=2.5e-4, clip_coef=0.2,
                 ppo_epochs=10, mini_batch_size=64, gamma=0.99, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, device='cpu'):
        self.device = device
        self.model = ActorCritic(state_dim, belief_dim, hidden_sizes, num_type, action_dim, action_bounds,
                                    has_private_type).to(device) #hidden_sizes, n_components, action_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip_coef = clip_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
