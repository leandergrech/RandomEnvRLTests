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
    l_list = np.zeros(n_eps)  # episode lengths

    for ep_idx in range(n_eps):
        o = env.reset()
        for step in range(max_steps):
            a = policy(o)

            o_list[ep_idx, step] = o.copy()
            a_list[ep_idx, step] = a
            o, r, d, _ = env.step(a)
            otp1_list[ep_idx, step] = o
            r_list[ep_idx, step] = r
            if d:
                break
        l_list[ep_idx] = step + 1

    return dict(obs=o_list, act=a_list, obstp1=otp1_list, rew=r_list, eplen=l_list)


sz = 3
rm = np.diag(np.ones(sz))
pi = rm.copy()
env = RandomEnv(sz, sz, estimate_scaling=False)
env.rm = rm
env.pi = pi

N_EPS = 5
MAX_STEPS = env.max_steps
GAMMA = 1.0

eps = 1e-8

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

policy = lambda s: 0.2 * env.get_optimal_action(s)


def get_features(s):
    x1 = s.copy()
    x2 = np.multiply(x1[:-1], x1[1:])
    # return x1
    # x2 = np.square(s)
    return np.concatenate([x1, x2])


d = len(get_features(np.zeros(sz)))
invA = (1 / eps) * np.diag(np.ones(d))
b = np.zeros(d)

for ep in range(N_EPS):
    o = env.reset()
    x = get_features(o)
    for step in range(MAX_STEPS):
        a = policy(o)
        otp1, r, d, _ = env.step(a)
        xtp1 = get_features(otp1)

        rhs = x - GAMMA * xtp1
        v = invA.T @ rhs

        num = (invA @ x) @ (v.T)
        den = 1 + (v.T) @ x
        invA = invA - num / den

        b = b + r * x

        w = invA @ b

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
    l_list.append(step + 1)

print(f'average ep_len={np.mean(l_list):.2f} +/- {np.std(l_list):2f}')
fig, _ = plt.subplots(3, 2)
for ax, dat, lab in zip(fig.axes,
                        (np.array(traj_invA).reshape(-1, d * d), traj_b, traj_w, traj_v, traj_num, traj_den),
                        ('invA', 'b', 'w', 'v', 'num', 'den')):
    ax.plot(dat, label=lab)
    ax.legend(loc='upper right')

fig.tight_layout()

fig2, (ax1, ax2, ax3) = plt.subplots(3)
for ax, dat, lab in zip(fig2.axes,
                        (o_traj, a_traj, r_traj),
                        ('o', 'a', 'r')):
    ax.plot(dat, label=lab)
    ax.legend(loc='upper right')
fig2.tight_layout()

l_marker_pos = np.cumsum(l_list)
for lm in l_marker_pos:
    for ax in fig.axes:
        ax.axvline(lm, color='k', alpha=0.5)

plt.show()
