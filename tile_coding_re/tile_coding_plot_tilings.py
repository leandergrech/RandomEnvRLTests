import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from tile_coding_re.tile_coding import get_tilings_from_env, QValueFunction, get_tile_coding
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions

nb_tilings = 4
nb_bins = 4

n_obs = 2
n_act = 2

env = RandomEnvDiscreteActions(n_obs, n_act)
all_actions = get_discrete_actions(n_act)
tilings = get_tilings_from_env(env, nb_tilings, nb_bins, asymmetrical=True)
print(tilings)
qvf = QValueFunction(tilings, all_actions, 0.0)
print(repr(env))
print(f'{nb_tilings} tilings, {nb_bins} bins')
print(f'Tilings shape = {tilings.shape}')
print(f'Q-table shape = {qvf.q_tables.shape}')
random_states = np.array([[-0.5, 0.5], [0.6, -0.6]])
for random_state in random_states:
	coding = get_tile_coding(random_state, tilings)
	print(f'Random state = {random_state} -> Coding = {coding}')

fig, ax = plt.subplots()
color = iter(plt.cm.jet(np.linspace(0, 1, nb_tilings)))
for tiling, c, ls in zip(tilings, color, cycle(('-', '--', ':', '-.'))):
	for tile, axfunc in zip(tiling, (ax.axhline, ax.axvline)):
		for t in tile:
			axfunc(t, color=c, linestyle=ls, linewidth=2)
plt.scatter(random_states.T[0], random_states.T[1], c='k', marker='x')
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
plt.show()
