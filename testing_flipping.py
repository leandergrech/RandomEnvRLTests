import torch as t

arr = t.rand(2000).reshape(20, 100)

gamma = 0.99

def compute1(batch_rews):
	global gamma

	q_vals = []
	for rewards in batch_rews:
		discount_pows = t.pow(gamma, t.arange(0, rewards.size(0)).float())
		discounted_rewards = rewards * discount_pows
		disc_reward_sums = t.cumsum(discounted_rewards.flip(0), dim=-1).flip(0)
		trajectory_q_vals = disc_reward_sums / discount_pows
		q_vals.append(trajectory_q_vals.numpy())

	return t.tensor(q_vals)


def compute2(batch_rews):
	global gamma

	q_vals = t.zeros_like(batch_rews)
	for i, rewards in enumerate(batch_rews):  # reverse maintains order on append
		reverse = t.arange(rewards.size(0) - 1, -1, -1)
		discount_pows = t.pow(gamma, t.arange(0, rewards.size(0)).float())
		discounted_rewards = rewards * discount_pows
		disc_reward_sums = t.cumsum(discounted_rewards[reverse], dim=-1)[reverse]
		trajectory_q_vals = disc_reward_sums / discount_pows
		q_vals[i] = trajectory_q_vals

	return q_vals

from datetime import datetime as dt
def timeit(func, inputs, number=1000):
	start = dt.now()
	for _ in range(number):
		func(*inputs)
	end = dt.now()

	return end - start

c1_time = timeit(compute1, (arr,))
print('compute1 time: ', c1_time)
c1 = compute1(arr)
print(c1)

c2_time = timeit(compute2, (arr,))
print('compute2 time: ', c2_time)
c2 = compute2(arr)
print(c2)

print(f'\n{(c2_time/c1_time - 1)*100.0:.2f}% speedup')
c_mean, c_std = t.mean(c1 - c2), t.std(c1 - c2)
print(c_mean, c_std)
