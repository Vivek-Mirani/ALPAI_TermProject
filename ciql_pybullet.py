#!/usr/bin/env python3
import gym
import d4rl_pybullet           # registers “-bullet-” D4RL envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
# temporarily restore the old bool8 alias so Gym’s checker doesn’t break
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# ----------------------------
# 1) NETWORKS
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hiddens, act=nn.ReLU):
        super().__init__()
        layers, last = [], in_dim
        for h in hiddens:
            layers += [nn.Linear(last, h), act()]
            last = h
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hiddens):
        super().__init__()
        self.body = MLP(obs_dim, hiddens)
        self.head = nn.Linear(hiddens[-1], 1)
    def forward(self, o):
        return self.head(self.body(o)).squeeze(-1)

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hiddens):
        super().__init__()
        self.body = MLP(obs_dim + act_dim, hiddens)
        self.head = nn.Linear(hiddens[-1], 1)
    def forward(self, o, a):
        x = torch.cat([o, a], dim=-1)
        return self.head(self.body(x)).squeeze(-1)

class DoubleQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hiddens):
        super().__init__()
        self.q1 = QNet(obs_dim, act_dim, hiddens)
        self.q2 = QNet(obs_dim, act_dim, hiddens)
    def forward(self, o, a):
        return self.q1(o,a), self.q2(o,a)

# ----------------------------
# 2) TANH-GAUSSIAN POLICY
# ----------------------------
LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0
EPS = 1e-6

class TanhGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hiddens):
        super().__init__()
        self.body    = MLP(obs_dim, hiddens)
        self.mean    = nn.Linear(hiddens[-1], act_dim)
        self.log_std = nn.Linear(hiddens[-1], act_dim)

    def forward(self, o):
        h     = self.body(o)
        m     = self.mean(h)
        log_s = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return m, log_s.exp()

    def sample(self, o):
        m, s   = self(o)
        dist   = torch.distributions.Normal(m, s)
        z      = dist.rsample()
        a      = torch.tanh(z).clamp(-0.999,0.999)
        logp   = dist.log_prob(z) - torch.log(1 - a.pow(2) + EPS)
        return a, logp.sum(-1,keepdim=True)

    def log_prob(self, o, a):
        a_ = a.clamp(-0.999,0.999)
        z  = 0.5*(torch.log1p(a_) - torch.log1p(-a_))
        m, s   = self(o)
        dist   = torch.distributions.Normal(m, s)
        logp   = dist.log_prob(z) - torch.log(1 - a_.pow(2) + EPS)
        return logp.sum(-1,keepdim=True)

    def log_prob_raw(self, obs, act):
        """
        Log-prob of *raw* actions under the pre-tanh Normal(m,s).
        Use for AWR on dataset actions (which are not in (-1,1)).
        """
        m, s = self(obs)
        dist = torch.distributions.Normal(m, s)
        # no tanh-correction here
        return dist.log_prob(act).sum(-1, keepdim=True)

# ----------------------------
# 3) REPLAY BUFFER
# ----------------------------
class ReplayBuffer:
    def __init__(self, data):
        self.obs      = data['observations']
        self.acts     = data['actions']
        self.rews     = data['rewards'].reshape(-1,1)
        self.next_obs = data['next_observations']
        self.dones    = data['terminals'].reshape(-1,1)
        self.size     = self.obs.shape[0]

    def sample(self, B):
        idx = np.random.randint(0, self.size, B)
        return dict(
            obs      = self.obs[idx],
            act      = self.acts[idx],
            rew      = self.rews[idx],
            next_obs = self.next_obs[idx],
            done     = self.dones[idx],
        )

    def sample_actions(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.acts[idx]

# ----------------------------
# 4) CIQL AGENT
# ----------------------------
class CIQLAgent:
    def __init__(self,
                 obs_dim, act_dim,
                 hiddens=(256,256),
                 gamma=0.99,
                 polyak=0.005,
                 expectile=0.8,
                 beta=1.0,
                 alpha=1.0,
                 num_neg=10,
                 lr=3e-4,
                 device='cuda'):

        self.device    = device
        self.gamma     = gamma
        self.polyak    = polyak
        self.expectile = expectile
        self.beta      = beta
        self.alpha     = alpha
        self.num_neg   = num_neg

        self.value       = ValueNet(obs_dim, hiddens).to(device)
        self.critic      = DoubleQ(obs_dim, act_dim, hiddens).to(device)
        self.critic_targ = DoubleQ(obs_dim, act_dim, hiddens).to(device)
        self.policy      = TanhGaussianPolicy(obs_dim, act_dim, hiddens).to(device)

        # Copy weights → targets
        for p, pt in zip(self.critic.parameters(),
                         self.critic_targ.parameters()):
            pt.data.copy_(p.data)

        self.v_opt  = Adam(self.value.parameters(),  lr=lr)
        self.q_opt  = Adam(self.critic.parameters(), lr=lr)
        self.pi_opt = Adam(self.policy.parameters(), lr=lr)

    def expectile_loss(self, diff):
        w = torch.where(diff>0, self.expectile, 1-self.expectile)
        return (w * diff.pow(2)).mean()

    def update(self, batch, buffer, step, warmup=1000):
        obs      = torch.as_tensor(batch['obs'],      device=self.device)
        act      = torch.as_tensor(batch['act'],      device=self.device)
        rew      = torch.as_tensor(batch['rew'],      device=self.device)
        next_obs = torch.as_tensor(batch['next_obs'], device=self.device)
        done     = torch.as_tensor(batch['done'],     device=self.device)
        B        = obs.size(0)

        # 1) V-step (IQL expectile)
        with torch.no_grad():
            q1_t, q2_t = self.critic_targ(obs, act)
            q_bar      = torch.min(q1_t, q2_t)
        v_pred = self.value(obs)
        v_loss = self.expectile_loss(q_bar - v_pred)
        self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()

        # 2) Q-step (Bellman + clamped CQL)
        with torch.no_grad():
            v2 = self.value(next_obs).unsqueeze(-1)
            y  = rew + self.gamma * (1 - done) * v2

        q1, q2 = self.critic(obs, act)
        mse1   = F.mse_loss(q1.unsqueeze(-1), y)
        mse2   = F.mse_loss(q2.unsqueeze(-1), y)

        # Off-data penalty
        a_neg   = buffer.sample_actions(B * self.num_neg)
        a_neg   = torch.as_tensor(a_neg, device=self.device)
        obs_rep = obs.unsqueeze(1).repeat(1, self.num_neg, 1).view(-1, obs.size(-1))
        q1n, q2n= self.critic(obs_rep, a_neg)
        q1n = q1n.view(B, self.num_neg); q2n = q2n.view(B, self.num_neg)

        c1 = F.relu(q1n.mean(1).mean() - q1.mean())
        c2 = F.relu(q2n.mean(1).mean() - q2.mean())

        q_loss = mse1 + mse2 + self.alpha * (c1 + c2)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # Polyak update
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(),
                             self.critic_targ.parameters()):
                pt.data.mul_(1 - self.polyak)
                pt.data.add_(self.polyak * p.data)

        # 3) π-step (AWR)
        pi_loss = torch.tensor(0., device=self.device)
        if step > warmup:
            with torch.no_grad():
                q1p, q2p = self.critic(obs, act)
                qb       = torch.min(q1p, q2p)
                adv      = (qb - self.value(obs)).clamp(min=0)   # [B]

            # stable softmax weighting
            x      = self.beta * adv                          # [B]
            x_max  = x.max()                                  # scalar
            exp_x  = torch.exp(x - x_max)                     # [B], in (0,1]
            wts    = exp_x / (exp_x.sum() + 1e-8)             # [B], sum=1
            wts    = wts.unsqueeze(-1)                        # [B,1]

            # raw log-prob of dataset actions (no atanh)
            logp   = self.policy.log_prob_raw(obs, act)      # [B,1]
            pi_loss = - (wts * logp).sum()                    # scalar

            self.pi_opt.zero_grad()
            pi_loss.backward()
            self.pi_opt.step()
        
        return v_loss.item(), q_loss.item(), pi_loss.item()


# ----------------------------
# 5) TRAIN + PLOTTING
# ----------------------------
def train():
    seed         = 0
    env_name     = 'halfcheetah-bullet-medium-v0'
    batch_size   = 256
    max_steps    = int(5e5)
    eval_int     = 5000
    log_int      = 1000

    env = gym.make(env_name)
    env.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    ds      = env.get_dataset()
    obs     = ds['observations']
    acts    = ds['actions']
    rews    = ds['rewards'].reshape(-1,1)
    terms   = ds['terminals'].reshape(-1,1)
    next_obs= np.concatenate([obs[1:], obs[-1:]], axis=0)

    buffer = ReplayBuffer({
        'observations':      obs,
        'actions':           acts,
        'rewards':           rews,
        'next_observations': next_obs,
        'terminals':         terms,
    })

    # normalize
    obs_mean = obs.mean(0)
    obs_std  = obs.std(0) + 1e-3

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent   = CIQLAgent(obs_dim, act_dim, device='cuda')

    eval_steps, eval_returns = [], []

    for step in range(1, max_steps+1):
        batch = buffer.sample(batch_size)
        batch['obs']      = (batch['obs']      - obs_mean) / obs_std
        batch['next_obs'] = (batch['next_obs'] - obs_mean) / obs_std

        v_l, q_l, pi_l = agent.update(batch, buffer, step)

        if step % log_int == 0:
            print(f"[{step:>7}] V:{v_l:.3f}  Q:{q_l:.3f}  π:{pi_l:.3f}")

        if step % eval_int == 0:
            rets = []
            for _ in range(5):
                out  = env.reset()
                state= out[0] if isinstance(out, tuple) else out
                done, ep_ret = False, 0.0
                while not done:
                    s_n = (state - obs_mean) / obs_std
                    s_t = torch.tensor(s_n, device='cuda', dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        a, _ = agent.policy.sample(s_t)
                    a_np = a.cpu().numpy()[0]
                    res  = env.step(a_np)
                    if len(res)==5:
                        state, r, done, _, _ = res
                    else:
                        state, r, done, _    = res
                    ep_ret += r
                rets.append(ep_ret)

            avg_ret = np.mean(rets)
            eval_steps.append(step)
            eval_returns.append(avg_ret)
            print(f" → Eval @ {step}: {avg_ret:.1f}")

    env.close()

    # Plot evaluation curve
    plt.figure()
    plt.plot(eval_steps, eval_returns)
    plt.xlabel('Training Step')
    plt.ylabel('Average Return')
    plt.title('CIQL Evaluation Returns')
    plt.savefig('eval_returns.png')
    plt.show()

if __name__ == '__main__':
    train()
