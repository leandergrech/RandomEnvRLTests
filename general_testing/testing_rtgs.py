import torch as t
from deep_rl.my_agents_utils import timeit

arr = t.rand(10000).reshape(1000, 10)

gamma = 0.99


def compute1(batch_rews):
    global gamma
    b, k = batch_rews.shape
    batch_rtgs = t.zeros_like(batch_rews, dtype=t.float)
    pows = t.pow(gamma, t.arange(k))

    discounted_rews = pows * t.cat([batch_rews[:, 1:], t.zeros(b, 1)], dim=-1)
    batch_rtgs = t.cumsum(discounted_rews.flip(-1), dim=-1).flip(-1) / pows

    return batch_rtgs


def compute2(batch_rews):
    global gamma
    b, k = batch_rews.shape
    batch_rtgs = t.zeros_like(batch_rews, dtype=t.float)

    rev_idx = k - 2
    for rew_at_timestep in batch_rews.flip(-1).transpose(0, 1):
        next_rtgs = batch_rtgs[:, rev_idx + 1]
        batch_rtgs[:, rev_idx] = rew_at_timestep + gamma * next_rtgs
        if rev_idx == 0: break
        rev_idx -= 1

    return batch_rtgs


def compute3(batch_rews):
    global gamma
    batch_rtgs = []
    for ep_rews in reversed(batch_rews):  # reverse maintains order on append
        discounted_r = 0
        for r in reversed(ep_rews):
            batch_rtgs.insert(0, discounted_r)
            discounted_r = r + discounted_r * gamma

    return t.tensor(batch_rtgs, dtype=t.float).reshape(batch_rews.shape)


c1_time = timeit(compute1, (arr,))
print('compute1 time: ', c1_time)
c1 = compute1(arr)
print(c1)

c2_time = timeit(compute2, (arr,))
print('compute2 time: ', c2_time)
c2 = compute2(arr)
print(c2)

c3_time = timeit(compute3, (arr,))
print('compute3 time: ', c3_time)
c3 = compute3(arr)
print(c2)

print(f'\n{(c3_time / c1_time) * 100.0:.2f}% speedup on compute1')
c_mean, c_std = t.mean(c1 - c3), t.std(c1 - c3)
print(c_mean, c_std)

print(f'\n{(c3_time / c2_time) * 100.0:.2f}% speedup on compute2')
c_mean, c_std = t.mean(c2 - c3), t.std(c2 - c3)
print(c_mean, c_std)
