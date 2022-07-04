import os
import numpy as np
# import seaborn as sns; sns.set_theme(style='white')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mpltick

from humanize import naturalsize

from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions
from tile_coding_re.tile_coding import QValueFunction2, get_tilings_from_env


def check_constants_py_args():
	from tile_coding_re.mc_method.constants import N_OBS, N_ACT, NB_TILINGS, NB_BINS, LR
	env = RandomEnvDiscreteActions(n_obs=N_OBS, n_act=N_ACT)
	tilings = get_tilings_from_env(env, NB_TILINGS, NB_BINS)
	actions = get_discrete_actions(N_ACT)

	qvf = QValueFunction2(tilings, actions, LR)

	n_elems = qvf.q_tables.size
	elem_size = qvf.q_tables.itemsize
	print(f'Number of actions permutations = {len(actions)}')
	print(f'Env dimensions =    {N_OBS}x{N_ACT}')
	print(f'Number of bins =    {NB_BINS}')
	print(f'Number of tilings = {NB_TILINGS}')
	print(f'Q-table n_elems =   {n_elems}')
	print(f'q-table size =      {naturalsize(n_elems * elem_size)}')


def heat_map_qtable_size():
	save_path = 'q_table_size_heatmaps'

	DUMMY_LR = 0.1

	nb_tilings_range = np.arange(2, 11)
	nb_bins_range = np.arange(2, 11)
	env_sz_range = np.arange(2, 11)
	for n_obs in env_sz_range:
		# for n_act in env_sz_range:
		# n_obs = n_act = env_sz
		n_act = n_obs
		# if n_obs == n_act:
		#     continue
		env = RandomEnvDiscreteActions(n_obs, n_act)
		actions = get_discrete_actions(n_act)
		sizes = np.zeros((len(nb_tilings_range), len(nb_bins_range)))
		for i, nb_tilings in enumerate(nb_tilings_range):
			for j, nb_bins in enumerate(nb_bins_range):
				size = nb_tilings * nb_bins ** n_obs * len(actions)
				sizes[i, j] = size

		fig, ax = plt.subplots()
		fig.suptitle(f'Q-table size chart\n{repr(env)}')
		im = ax.imshow(sizes, norm=LogNorm(),
					   extent=(min(nb_bins_range), max(nb_bins_range), min(nb_tilings_range), max(nb_tilings_range)))

		cb = fig.colorbar(im, ax=ax, ticks=mpltick.LogLocator())
		cb.set_label('Size (bytes)', rotation=90)

		ax.set_xlabel('Nb of bins')
		plt.xticks(nb_bins_range)
		ax.set_ylabel('Nb of tilings')
		plt.yticks(nb_tilings_range)
		fig.tight_layout()
		plt.savefig(os.path.join(save_path, repr(env)))


if __name__ == '__main__':
	# check_constants_py_args()
	heat_map_qtable_size()
