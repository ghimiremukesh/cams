from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Distribution
import ipdb
import numpy as np

def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ConditionalGMM(Distribution):
    def __init__(self, logits: torch.Tensor, means: torch.Tensor, log_stds: torch.Tensor, temp: float=1.0):
        batch_shape, event_shape = logits.shape[:-1], means.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=False)
        self.logits = logits      # [B, K]
        self.cat = Categorical(logits=self.logits)
        self.means = means          # [B, K, D]
        self.log_stds = log_stds     # [B, K, D]


    def sample(self, action=None):
        k = self.cat.sample()  # [B]
        B, K, D = *self.means.shape[:2], self.means.size(-1)
        idx = torch.arange(B, device=k.device)
        mu = self.means[idx, k]       # [B, D]
        sd = self.log_stds.exp()[idx, k]  # [B, D]

        dist = Normal(mu, sd)

        if action is None:
            action = dist.rsample()  # same as doing mu + sd * N(0, 1)

        # ipdb.set_trace()
        logp = (dist.log_prob(action).sum(dim=-1) +
                 self.cat.log_prob(k))

        ent = (dist.entropy().sum(-1) +
               self.cat.entropy())

        return action, logp, ent, k


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        belief_dim: int,
        hidden_sizes: list[int],
        n_types: int,
        action_dim: int,
        action_bounds: list[float],
        has_private_type: bool
    ):
        super().__init__()
        # Dimensions
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.n_types = n_types    # number of private types and mixture components K
        self.action_dim = action_dim
        self.has_private_type = has_private_type

        self.register_buffer("log_std_max", torch.tensor(0.))
        self.register_buffer("temp", torch.tensor(3.))
        self.register_buffer("action_bounds", torch.tensor(action_bounds, dtype=torch.float32))


        # Feature extractor on state + action history
        layers = []
        last = state_dim + self.belief_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.Tanh()]
            last = h
        self.state_net = nn.Sequential(*layers)

        # Heads for means, sigma, and conditional probs (alphas)
        self.means_head = nn.Linear(last, self.n_types * action_dim)  # for P1
        self.log_stds_head = nn.Linear(last, self.n_types * action_dim) # for P1

        # heads for br means, sigma # for P2
        self.br_means_head = nn.Linear(last, self.n_types * action_dim)
        self.br_log_stds_head = nn.Linear(last, self.n_types * action_dim)

        self.probs_head = nn.Linear(last, self.n_types)
        # nn.init.zeros_(self.probs_head.weight)
        # initial_probs = torch.tensor([1e-6, (1-1e-6)])
        # with torch.no_grad():
        #     self.probs_head.bias.copy_(initial_probs)
        # self.values_head = nn.Linear(last, self.n_types)
        self.values_head = nn.Sequential(nn.Linear(state_dim + self.belief_dim, 64),
                                          nn.LayerNorm(64),
                                          nn.Tanh(),
                                          nn.Linear(64, 64),
                                          nn.LayerNorm(64),
                                          nn.Tanh(),
                                          nn.Linear(64, self.n_types))


    def forward(self, obs: dict):
        """
        obs: dict with keys:
          'state':           [B, state_dim]
          'belief':          [B, n_types]
          'action_history':  [B, action_hist_dim]
          'player_type':     [B, n_types]
        returns:
          logits:   [B, K]
          weights:  [B, K]
          means:    [B, K, D]
          log_stds: [B, K, D]
          value:    [B]
        """
        s = obs['state']  # [B, state_dim]
        b = obs['belief'].reshape(s.size(0), -1)  # [B, n_types]

        # try adding type to common network
        s = torch.cat([s, b], dim=-1)  # add belief to state net

        # Extract features from state
        h = self.state_net(s)  # [B, last]
        K = self.n_types
        D = self.action_dim

        # means and sigmas
        raw_means = self.means_head(h).view(-1, K, D)  # P1

        means = (self.action_bounds * torch.tanh(raw_means))

        log_stds = self.log_stds_head(h).clamp(max=self.log_std_max).view(-1, K, D) # clamp log_std

        # logits and values
        if self.has_private_type:
            cond_probs = self.probs_head(h).sigmoid() # between [0, 1]
        else:
            cond_probs = self.probs_head(h) # logits (lambdas)

        values = self.values_head(s).squeeze(-1)

        # ipdb.set_trace()
        return means, log_stds, values, cond_probs

    def get_action(self, obs: dict, action: torch.Tensor = None):
        """
        Sample or evaluate action with log_prob and entropy.
        Returns:
          action: [B, D]
          logp:   [B]
          ent:    [B]
          value:  [B]
        """
        # get batch size and device for posterior computation
        B = obs['state'].size(0)
        device = obs['state'].device

        ## actual action computation
        means, log_stds, values, cond_probs = self.forward(obs)

        # get the conditional probs p(u1|1) and p(u1|2)
        alpha_1 = cond_probs[:, 0].reshape(-1, 1)
        alpha_2 = cond_probs[:, 1].reshape(-1, 1)

        eps = 1e-6

        if self.has_private_type:
            # compute marginals
            prior = obs['belief'].clamp(min=eps, max=(1-eps)).reshape(-1, 1)
            lam_1 = torch.exp(alpha_1.log() + prior.log()) + torch.exp(alpha_2.log() + (1-prior).log())
            lam_2 = 1 - lam_1

            # compute posterior
            post_1_log = alpha_1.log() + prior.log() - lam_1.log()
            post_2_log = (1-alpha_1).log() + prior.log() - lam_2.log()

            alphas_t1 = torch.cat((alpha_1, 1-alpha_1), dim=1).reshape(-1, 2)  # logits for type-1 P1
            alphas_t2 = torch.cat((alpha_2, 1-alpha_2), dim=1).reshape(-1, 2)  # logits for type-2 P2
            post = torch.cat((post_1_log, post_2_log), dim=1).reshape(-1, 2)    # posteriors

            type_idxs = obs['player_type'].argmax(dim=-1) # get the index for the type

            all_policy_logits = torch.zeros((B, self.n_types, self.n_types)).to(device)  # (batch, player_types, probabilities)
            all_policy_logits[:, 0, :] = alphas_t1.log()
            all_policy_logits[:, 1, :] = alphas_t2.log()

            # now select based on type
            selected_logits = all_policy_logits[torch.arange(B), type_idxs]
            value = values[torch.arange(B), type_idxs]

            dist = ConditionalGMM(selected_logits, means, log_stds, temp=self.temp)
            if action is None:
                action, logp, ent, k = dist.sample()
                posterior = post[torch.arange(B), k]  # all posteriors picked
                posterior = posterior.exp()
            else:
                _, logp, ent, k = dist.sample(action)
                posterior = post[torch.arange(B), k]
                posterior = posterior.exp()
        else:
            selected_logits = cond_probs
            dist = ConditionalGMM(selected_logits, means, log_stds, temp=self.temp)
            value = (cond_probs.softmax(-1) * values).sum(-1)

            if action is None:
                action, logp, ent, k = dist.sample()
                posterior = None
            else:
                _, logp, ent, k = dist.sample(action)
                posterior = None

        # ipdb.set_trace()
        return action, logp, ent, value, posterior # for posterior
