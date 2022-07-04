import numpy as np
import torch as t
from torch.distributions import MultivariateNormal

from random_env.envs import RandomEnv, RunningStats
from my_agents_utils import timeit

# env = RandomEnv.load_from_dir('common_envs')
env = RandomEnv(5, 5, True)
obs_dim = env.obs_dimension
act_dim = env.act_dimension

batch_size = 100
ep_max_size = 10


def get_action(o):
    dist = MultivariateNormal(t.zeros(act_dim), t.diag(t.ones(act_dim)))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob.item()


def compute_rtgs(r, lens=None):
    if isinstance(r, list): r = t.tensor(r)
    pows = t.pow(0.99, t.arange(r.size(-1)))
    temp = pows * r
    rev_disc_rew = temp.flip(-1)
    returns = t.cumsum(rev_disc_rew, dim=-1).flip(-1) / pows
    return returns


def rollout1(*args):
    t.random.manual_seed(123)
    np.random.seed(123)

    B, K = batch_size, ep_max_size

    batch_obs = []
    batch_acts = []
    batch_log_probs = []
    batch_rews = []
    batch_lens = []
    batch_rtgs = []

    b = 0
    while b < B:
        o = env.reset()
        for k in range(1, K + 1):
            a, log_prob = get_action(0)
            a = a.detach().numpy().tolist()
            otp1, r, d, _ = env.step(a)
            batch_obs.extend(o.tolist())
            batch_acts.extend(a)
            batch_rews.append(r)
            batch_log_probs.append(log_prob)
            o = otp1
            if d:
                break
        batch_lens.append(k)
        batch_rtgs.extend(compute_rtgs(batch_rews[-k:]))
        b += k

    batch_obs = t.tensor(batch_obs)
    batch_acts = t.tensor(batch_acts)
    batch_rews = t.tensor(batch_rews)
    batch_log_probs = t.tensor(batch_log_probs)
    batch_rtgs = t.tensor(batch_rtgs)
    batch_lens = t.tensor(batch_lens)
    return batch_obs, batch_acts, batch_rews, batch_log_probs, batch_rtgs, batch_lens


def rollout2(*args):
    t.random.manual_seed(123)
    np.random.seed(123)

    K = ep_max_size
    B = batch_size

    batch_obs = t.zeros(B * obs_dim)  # batch observations
    batch_acts = t.zeros(B * act_dim)  # batch actions
    batch_log_probs = t.zeros(B)  # log probs of each action
    batch_rews = t.zeros_like(batch_log_probs)  # batch rewards
    batch_lens = t.zeros_like(batch_log_probs)  # episodic lengths in batch
    batch_rtgs = t.zeros_like(batch_log_probs)

    b = 0
    while b < B:
        o = env.reset()
        stop_idx = np.random.choice(K)
        for k in range(K):
            a, log_prob = get_action(o)
            obs_idx = b * obs_dim
            batch_obs[obs_idx:obs_idx + obs_dim] = t.tensor(o)
            o, r, d, _ = env.step(a)
            act_idx = b * obs_dim
            batch_acts[act_idx:act_idx + act_dim] = a
            batch_log_probs[b] = log_prob
            batch_rews[b] = r
            if k == stop_idx: break
            if d: break
        batch_rtgs[b:b + k] = compute_rtgs(batch_rews[-k:])
        batch_lens[b] = k + 1
        b += k

    batch_obs = batch_obs.reshape(-1, obs_dim)
    batch_acts = batch_acts.reshape(-1, act_dim)
    batch_rews = batch_rews.reshape(-1)
    batch_log_probs = batch_log_probs.reshape(-1)
    batch_rtgs = batch_rtgs.reshape(-1)

    return batch_obs, batch_acts, batch_rews, batch_log_probs, batch_rtgs, batch_lens


t2, batch2 = timeit(rollout2, (None,), 1)
t1, batch1 = timeit(rollout1, (None,), 1)

print(f'{(t1 / t2) * 100.0}% improvement')
print(f'B={batch_size}, K={ep_max_size}')
print(f't_old={t1}\nt_new={t2}')
print(batch1[2].reshape(-1, obs_dim), batch1[4].reshape(-1, obs_dim))
print(batch1[4].shape, batch2[4].shape)
