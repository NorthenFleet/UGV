import torch
import torch.nn as nn
import numpy as np

class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.01, device="cpu"):
        self.model = model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.device = device

    def act(self, obs_np):
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            a, logp = self.model.act(obs)
            v = self.model.value(obs)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    def compute_gae(self, rewards, values, dones):
        adv = np.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            nextnonterminal = 1.0 - dones[t]
            nextvalue = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values[:-1]
        return adv, returns

    def update(self, obs, actions, logprobs, returns, advantages, epochs=10, batch_size=64):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        oldlogp_t = torch.as_tensor(logprobs, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        n = obs_t.shape[0]
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = start + batch_size
                mb = idx[start:end]
                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_oldlogp = oldlogp_t[mb]
                mb_ret = ret_t[mb]
                mb_adv = adv_t[mb]
                logp = self.model.evaluate_logp(mb_obs, mb_act)
                ratio = torch.exp(logp - mb_oldlogp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                v_pred = self.model.value(mb_obs)
                value_loss = 0.5 * (mb_ret - v_pred).pow(2).mean()
                mu = self.model.actor(mb_obs)
                std = torch.exp(self.model.log_std)
                dist = torch.distributions.Normal(mu, std)
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }