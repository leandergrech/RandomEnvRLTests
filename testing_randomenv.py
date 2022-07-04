import numpy as np
import matplotlib.pyplot as plt

from random_env.envs import RandomEnv


def quick_testing_randomenv():
	n_obs = 10
	n_act = 10

	# env = RandomEnv(n_obs, n_act, estimate_scaling=True)
	env = RandomEnv.load_from_dir("C:\\Users\\Leander\\Code\\RandomEnvRLTests\\common_envs")

	d = False
	o1 = env.reset()
	o_list = [o1]
	a_list = []

	# print(np.linalg.det(env.pi),np.linalg.det(env.rm))

	fig, (ax1, ax2) = plt.subplots(2)
	ax1.set_title('Observations')
	ax2.set_title('Actions')

	lobs, = ax1.plot(np.zeros(n_obs))
	lact, = ax2.plot(np.zeros(n_act))

	ax1.axhline(-env.GOAL, color='g', ls='--', lw=1.5)
	ax1.axhline(env.GOAL, color='g', ls='--', lw=1.5)

	plt.show(block=False)
	for ax in fig.axes:
		ax.set_ylim((-1, 1))
		ax.axhline(0.0, color='k', ls='--', lw=0.7)

	cur_step = 0
	while not d:
		a = env.get_optimal_action(o1)
		o2, r, d, _ = env.step(a)

		lobs.set_ydata(o2)
		lact.set_ydata(a)
		fig.suptitle(f'Step {cur_step}, Done = {d}\n' +
					 f'std(o1 - o2))) = {np.std(o1 - o2):.4f}\treward = {r:.4f}')
		plt.pause(0.1)

		o_list.append(o2.copy())
		a_list.append(a)
		o1 = np.copy(o2)

		cur_step += 1
	plt.pause(2)

	o_list = np.array(o_list)
	a_list = np.array(a_list)

	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.plot(o_list, label='States')
	ax2.plot(a_list, label='Actions')
	ax1.axhline(-env.GOAL, color='k', ls='--')
	ax1.axhline(env.GOAL, color='k', ls='--')

	# for a in fig.axes:
	# 	a.legend(loc='best')
	plt.show()


if __name__ == '__main__':
	# quick_testing_randomenv()
	env = RandomEnv.load_from_dir('common_envs')
