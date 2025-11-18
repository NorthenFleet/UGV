import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 128), act=nn.Tanh):
        super().__init__()
        layers = []
        dims = (in_dim,) + hidden
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = MLP(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = MLP(obs_dim, 1)

    def act(self, obs):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a, logp

    def evaluate_logp(self, obs, actions):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(actions).sum(-1)
        return logp

    def value(self, obs):
        return self.critic(obs).squeeze(-1)