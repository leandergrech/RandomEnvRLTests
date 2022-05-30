import os
import numpy as np
from datetime import datetime as dt

import torch as t
import matplotlib.pyplot as plt

from random_env.envs import RandomEnv

def generate_trajectory(env, policy, n_eps):
	n_obs, n_act = env.obs_dimension, env.act_dimension
	max_steps = env.max_steps

	o_list = np.zeros((n_eps, max_steps, n_obs))
	otp1_list = np.zeros((n_eps, max_steps, n_obs))
	a_list = np.zeros((n_eps, max_steps, n_act))
	r_list = np.zeros((n_eps, max_steps))
	l_list = np.zeros(n_eps) # episode lengths

	for ep_idx in range(n_eps):
		o = env.reset()
		for step in range(max_steps):
			a = policy(o)

			o_list[ep_idx, step] = o
			a_list[ep_idx, step] = a
			o,r,d,_ = env.step(a)
			otp1_list[ep_idx, step] = o
			r_list[ep_idx, step] = r
			if d:
				break
		l_list[ep_idx] = step + 1

	return dict(obs=o_list, act=a_list, obstp1=otp1_list, rew=r_list, eplen=l_list)

sz = 3
env = RandomEnv(sz, sz, estimate_scaling=True)
N_EPS = 10
MAX_STEPS = env.max_steps
GAMMA = 0.9

eps = 1e-1

"""
LSTD algorithm Sutton&Barto p.230
"""
invA = 1/eps * np.diag(np.ones(sz))
b = np.zeros(sz)

policy = lambda s: 0.2 * env.get_optimal_action(s)

traj_invA = []
traj_b = []
traj_w = []
l_list = []

for ep in range(N_EPS):
	o = env.reset()
	for step in range(MAX_STEPS):
		a = policy(o)
		otp1,r,d,_ = env.step(a)

		v = np.transpose(invA).dot(o - GAMMA * otp1)
		invA = invA - (invA.dot(o)).dot(np.transpose(v))/(1 + np.transpose(v).dot(o))
		b = b + r * o

		w = invA.dot(b)

		traj_invA.append(invA.copy())
		traj_b.append(b.copy())
		traj_w.append(w.copy())

		o = otp1

		if d:
			break
	l_list.append(step)

print(f'average ep_len={np.mean(l_list):.2f} +/- {np.std(l_list):2f}')
fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.plot(np.array(traj_invA).reshape(-1, sz*sz))
ax1.set_title('invA')
ax1.set_yscale('symlog')
ax2.plot(traj_b)
ax2.set_title('b')
ax3.plot(traj_w)
ax3.set_title('w')
fig.tight_layout()
plt.show()


