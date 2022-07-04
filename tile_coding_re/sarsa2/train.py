import os
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
from collections import deque

import numpy as np
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions
from tile_coding_re.tile_coding import get_tilings_from_env, QValueFunction
from tile_coding_re.utils import TrajBuffer
from tile_coding_re.sarsa2.constants import *

"""
Sarsa
With Double-Q to mitigate maximization bias
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
qvf1 = QValueFunction(tilings, all_actions, lr=LR)
qvf2 = QValueFunction(tilings, all_actions, lr=LR)


def get_average_action_value(state, action):
	val1 = qvf1.value(state, action)
	val2 = qvf2.value(state, action)
	return (val1 + val2) / 2


def get_average_value(state):
	val1 = max([qvf1.value(state, a_) for a_ in all_actions])
	val2 = max([qvf2.value(state, a_) for a_ in all_actions])
	return (val1 + val2) / 2


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


def greedy_policy(state):
	# greedy selection of action with the largest value
	vals = [get_average_action_value(state, a_) for a_ in all_actions]

	return all_actions[np.argmax(vals)]


def randomly_swap_q():
	if np.random.choice(2):
		return qvf1, qvf2
	else:
		return qvf2, qvf1


MAX_STEPS_POSSIBLE = NB_TRAINING_EPS * env.max_steps
'''Tracking info'''
time_steps_vs_ep = np.zeros(MAX_STEPS_POSSIBLE)
ep_lens = np.zeros(NB_TRAINING_EPS)

td_errors_rolling_1 = deque(maxlen=env.max_steps)
td_errors_mean_1 = np.zeros(NB_TRAINING_EPS)
td_errors_rolling_2 = deque(maxlen=env.max_steps)
td_errors_mean_2 = np.zeros(NB_TRAINING_EPS)

nb_discretization_bins = 50
state_discretization = np.linspace(-10, 10, nb_discretization_bins - 1)
visited_states = np.zeros((env.obs_dimension, nb_discretization_bins))

training_t = 0
for ep in trange(NB_TRAINING_EPS):
	ep_t = 0
	d = False

	o = env.reset()
	qvfa, qvfb = qvf1, qvf2

	a1 = greedy_policy(o)
	while not d:
		otp1, r, d, info = env.step(a1)
		if np.random.rand() < GREEDY_EPS(ep):
			a2 = env.action_space.sample()
		else:
			a2 = greedy_policy(otp1)

		if r < -1.:
			r = -2.

		if info['success']:
			target = r  # setting value of terminal state to zero
		else:
			target = r + GAMMA * qvfb.value(otp1, a2)

		qvfa.update(o, a1, target)

		o = otp1
		a1 = a2

		qvfa, qvfb = randomly_swap_q()

		# save stuff
		time_steps_vs_ep[training_t] = ep
		td_errors_rolling_1.append(target - qvf1.value(o, a1))
		td_errors_rolling_2.append(target - qvf2.value(o, a1))
		for dim, o_ in enumerate(otp1):
			bidx = np.digitize(o_, state_discretization)
			visited_states[dim, bidx] += 1

		# step counters
		ep_t += 1
		training_t += 1

	ep_lens[ep] = ep_t
	td_errors_mean_1[ep] = np.mean(td_errors_rolling_1)
	td_errors_mean_2[ep] = np.mean(td_errors_rolling_2)

	# logging
	if (ep + 1) % SAVE_EVERY == 0:
		# print(f"Episode {ep+1}")
		qvf1.save(os.path.join(save_path, f'{ep + 1}ep', 'qvf1.npz'))
		qvf2.save(os.path.join(save_path, f'{ep + 1}ep', 'qvf2.npz'))
time_steps_vs_ep = time_steps_vs_ep[:training_t]

np.save(os.path.join(par_dir, 'training_ep_lens.npy'), ep_lens)
np.save(os.path.join(par_dir, 'td_errors_mean_1.npy'), td_errors_mean_1)
np.save(os.path.join(par_dir, 'td_errors_mean_2.npy'), td_errors_mean_2)
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

ax2.plot(td_errors_mean_1)
ax2.plot(td_errors_mean_2)
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
