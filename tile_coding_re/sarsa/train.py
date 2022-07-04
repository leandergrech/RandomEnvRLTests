import os
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
from collections import deque

import numpy as np
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions
from tile_coding_re.tile_coding import get_tilings_from_env, QValueFunction2
from tile_coding_re.utils import TrajBuffer
from tile_coding_re.sarsa.constants import *

"""
Sarsa
Epsilon-greedy on-policy TD-control
"""

'''
Create RE to train on
'''
env = RandomEnvDiscreteActions(n_obs=N_OBS, n_act=N_ACT)

'''
Create tilings
'''
tilings = get_tilings_from_env(env, NB_TILINGS, NB_BINS, range_scale=1.)

'''
Create tabular Q-function
'''
all_actions = get_discrete_actions(N_ACT)
qvf = QValueFunction2(tilings, all_actions, lr=LR)

'''
Directory  handling
'''
# par_dir = f'{repr(env)}_{NB_BINS}bins_{NB_TILINGS}tilings_{LR}lr_{GREEDY_EPS}eps-greedy'
if os.path.exists(par_dir):
	print(f"Run with these hparams already made: {par_dir}")
	ans = input('Continue? [Y/n]')
	if ans.lower() == 'y' or ans == '':
		title_found = False
		title_idx = 1
		temp = None
		while not title_found:
			temp = par_dir + f'_{title_idx}'
			title_found = not os.path.exists(temp)
			title_idx += 1
		par_dir = temp
		print(f'Continuing in directory: {par_dir}')
	else:
		exit(42)
save_path = os.path.join(par_dir, 'saves')
os.makedirs(save_path)
# save training parameters
with open('constants.py', 'r') as readfile, open(os.path.join(par_dir, 'info.md'), 'a') as writefile:
	for line in readfile:
		writefile.write(line)

'''
Save dynamics
'''
env.save_dynamics(save_path)

'''
Training
'''
T = 0
buffer = TrajBuffer()


def policy(state):
	if np.random.rand() < GREEDY_EPS:  # or T < NB_INIT_STEPS:
		return env.action_space.sample()
	else:
		# greedy selection of action with the largest value
		return qvf.greedy_action(state)


MAX_STEPS_POSSIBLE = NB_TRAINING_EPS * env.max_steps
'''Tracking info'''
time_steps_vs_ep = np.zeros(MAX_STEPS_POSSIBLE)
ep_lens = np.zeros(NB_TRAINING_EPS)
td_errors_rolling = deque(maxlen=env.max_steps)
td_errors_mean = np.zeros(NB_TRAINING_EPS)
nb_discretization_bins = 50
state_discretization = np.linspace(-10, 10, nb_discretization_bins - 1)
visited_states = np.zeros((env.obs_dimension, nb_discretization_bins))

training_t = 0
for ep in trange(NB_TRAINING_EPS):
	ep_t = 0
	d = False

	o = env.reset()
	a1 = policy(o)
	while not d:
		otp1, r, d, info = env.step(a1)
		a2 = policy(otp1)

		if r < -1.:
			r *= 10.0

		if info['success']:
			target = r  # setting value of terminal state to zero
		else:
			target = r + GAMMA * qvf.value(otp1, a2)

		qvf.update(o, a1, target)

		o = otp1
		a1 = a2

		time_steps_vs_ep[training_t] = ep
		td_errors_rolling.append(target - qvf.value(o, a1))
		for dim, o_ in enumerate(otp1):
			bidx = np.digitize(o_, state_discretization)
			visited_states[dim, bidx] += 1

		ep_t += 1
		training_t += 1

	ep_lens[ep] = ep_t
	td_errors_mean[ep] = np.mean(td_errors_rolling)

	# logging
	if (ep + 1) % SAVE_EVERY == 0:
		# print(f"Episode {ep+1}")
		qvf.save(os.path.join(save_path, f'{ep + 1}ep'))
time_steps_vs_ep = time_steps_vs_ep[:training_t]

np.save(os.path.join(par_dir, 'training_ep_lens.npy'), ep_lens)
np.save(os.path.join(par_dir, 'td_errors_mean.npy'), td_errors_mean)
np.save(os.path.join(par_dir, 'time_steps_vs_ep.npy'), time_steps_vs_ep)

tok = par_dir.split('_')
fig_title = ' '.join(tok[:4]) + '\n' + ' '.join(tok[4:])

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle(fig_title)
ax1.plot(ep_lens, label='Raw')
ep_lens_smooth1 = pd.Series(ep_lens).rolling(25).mean().to_numpy()
ep_lens_smooth2 = pd.Series(ep_lens).rolling(50).mean().to_numpy()
ax1.plot(ep_lens_smooth1, label='Mean (window=25')
ax1.plot(ep_lens_smooth2, label='Mean (window=50')
ax1.set_ylabel('Episode length')

ax2.plot(td_errors_mean)
ax2.set_ylabel('TD errors')

min_state, max_state = state_discretization[0], state_discretization[-1]
barx = np.linspace(min_state, max_state, nb_discretization_bins)
ax3.bar(barx, visited_states[0])
ax3.bar(barx, visited_states[1])
ax3.set_ylabel('Visited states counter')

plt.minorticks_on()
for ax in fig.axes:
	ax.set_xlabel('Training episodes')
	ax.legend(loc='best')
	ax.grid(which='major')
	ax.grid(which='minor', ls=':')
fig.tight_layout()
plt.savefig(os.path.join(par_dir, 'training_ep_lens_td_errors_visited_states.png'))

fig, ax = plt.subplots()
fig.suptitle(fig_title)
ax.plot(np.arange(MAX_STEPS_POSSIBLE),
		np.repeat(np.arange(NB_TRAINING_EPS), env.max_steps),
		c='k', ls='dashed', lw=0.5, label='Worst case')
ax.plot(np.arange(training_t), time_steps_vs_ep, c='r', label='Training')
ax.set_xlabel('Time steps')
ax.set_ylabel('Training episodes')
ax.legend(loc='best')
plt.savefig(os.path.join(par_dir, 'training_time_steps_vs_ep.png'))
