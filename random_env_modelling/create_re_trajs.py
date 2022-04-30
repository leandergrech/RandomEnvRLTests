import os
import numpy as np
import torch as th
from random_env.envs import RandomEnv
from my_agents_utils import make_path_exist, compute_returns

from constants import DISCOUNT, NB_TRAJS, DATA_DIR_NAME

make_path_exist(DATA_DIR_NAME)


for n_act in range(2, 11, 2):
	for n_obs in range(2, 11, 2):
		obses = np.zeros(shape=(NB_TRAJS, n_obs))
		obsestp1 = np.zeros(shape=(NB_TRAJS, n_obs))
		acts = np.zeros(shape=(NB_TRAJS, n_act))
		rews = np.zeros(NB_TRAJS)
		rets = np.zeros(NB_TRAJS)

		env = RandomEnv(n_obs, n_act, estimate_scaling=True)

		path = os.path.join(DATA_DIR_NAME, f'{n_obs}x{n_act}')
		make_path_exist(path)
		env.save_dynamics(path)

		t = 0
		o = env.reset()
		k = 0
		while t < NB_TRAJS:
			obses[t] = o

			a = env.action_space.sample()
			acts[t] = a

			o, r, d, _ = env.step(a)
			obsestp1[t] = o
			rews[t] = r

			t += 1
			k += 1
			if d:
				d = False
				o = env.reset()
				rets[t - k:t] = compute_returns(rews[t - k:t], DISCOUNT)
				k = 0

		np.savez(os.path.join(path, f'{NB_TRAJS}_{repr(env)}_trajecotries'), obses=obses, obsestp1=obsestp1,
		         acts=acts, rews=rews, rets=rets)



