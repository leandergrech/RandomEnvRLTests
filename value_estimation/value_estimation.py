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

			o_list[ep_idx, step] = o.copy()
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
N_EPS = 1
MAX_STEPS = env.max_steps
GAMMA = 0.9

eps = 1e-1

"""
LSTD algorithm Sutton&Barto p.230
"""

traj_invA = []
traj_b = []
traj_w = []
traj_v = []
traj_num = []
traj_den = []
l_list = []

a_traj = []
o_traj = []
r_traj = []

invA = (1/eps) * np.diag(np.ones(sz))
b = np.zeros(sz)

policy = lambda s: 0.2 * env.get_optimal_action(s)
get_features = lambda s: np.copy(s)

for ep in range(N_EPS):
	o = env.reset()
	x = get_features(o)
	for step in range(MAX_STEPS):
		a = policy(o)
		otp1, r, d,_ = env.step(a)
		xtp1 = get_features(otp1)

		rhs = x - GAMMA * xtp1
		v = invA.T.dot(rhs)

		num = invA.dot(x).dot(v.T)
		den = 1 + v.T.dot(x)
		invA = invA - num/den

		b = b + r * x

		w = invA.dot(b)

		traj_invA.append(invA.copy())
		traj_b.append(b.copy())
		traj_w.append(w.copy())
		traj_v.append(v.copy())
		traj_num.append(num.copy())
		traj_den.append(den.copy())

		a_traj.append(a)
		o_traj.append(o.copy())
		r_traj.append(r)

		o = otp1

		if d:
			break
	l_list.append(step+1)

print(f'average ep_len={np.mean(l_list):.2f} +/- {np.std(l_list):2f}')
fig, (ax1, ax2, ax3) = plt.subplots(3)

for ax, dat, lab in zip(fig.axes, (invA, b), ())
ax1.plot(np.array(traj_invA).reshape(-1, sz*sz))
ax1.set_title('invA')
ax1.set_yscale('symlog')
ax2.plot(traj_b)
ax2.set_title('b')
ax3.plot(traj_w)
ax3.set_title('w')
fig.tight_layout()

fig2, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(o_traj)
ax1.set_title('o')
ax2.plot(a_traj)
ax2.set_title('a')
ax3.plot(r_traj)
ax3.set_title('r')
fig2.tight_layout()

l_marker_pos = np.cumsum(l_list)
for lm in l_marker_pos:
	for ax in fig.axes:
		ax.axvline(lm, color='k', alpha=0.5)

plt.show()


