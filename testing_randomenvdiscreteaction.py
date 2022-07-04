import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from random_env.envs import RandomEnvDiscreteActions


def quick_testing_randomenvdiscreteactions():
	n_obs = 5
	n_act = 5

	env = RandomEnvDiscreteActions(n_obs, n_act)

	record = defaultdict(list)

	d = False
	o = env.reset()
	record['o'].append(o)
	while not d:
		# a = env.action_space.sample()
		a = env.get_optimal_action(o)
		o, r, d, _ = env.step(a)
		aa = env.get_actual_actions().copy()

		record['a'].append(a)
		record['aa'].append(aa)
		record['o'].append(o.copy())
		record['r'].append(r)

	fig, (ax1, ax2, ax3) = plt.subplots(3)
	ax1.set_title('Observations')
	ax1.plot(record['o'], c='b')
	# ax2.bar(np.repeat([range(len(record['a']))], n_act, axis=0), record['a'], c='r')
	ax2.set_title('Multi-Discrete Actions')
	ax2.plot(record['a'], c='r', ls='solid')
	ax3.set_title('Dynamics Actions')
	ax3.plot(record['aa'], c='r', ls='dashed')
	fig.tight_layout()
	# plt.show()


if __name__ == '__main__':
	for _ in range(5):
		quick_testing_randomenvdiscreteactions()
	plt.show()
