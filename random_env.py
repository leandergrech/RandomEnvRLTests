from collections import deque

from gym import Env
from gym.spaces import Box

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class RandomEnv(Env):
	EPISODE_LENGTH_LIMIT = 50
	REWARD_DEQUE_SIZE = 1
	GOAL = 0.05

	def __init__(self, n_obs, n_act, seed=123):
		self.obs_dimension, self.act_dimension = n_obs, n_act
		self.seed(seed)
		self.create_model()

		''' State and action space'''
		self.observation_space = Box(low=-1.0,
		                                 high=1.0,
		                                 shape=(self.obs_dimension,),
		                                 dtype=np.float32)
		self.action_space = Box(low=-np.ones(self.act_dimension),
		                            high=np.ones(self.act_dimension),
		                            dtype=np.float32)

		''' RL related parameters'''
		self.current_state = None
		self._reward = None
		self.reward_thresh = self.objective([RandomEnv.GOAL] * self.obs_dimension)
		self.reward_deque = deque(maxlen=RandomEnv.REWARD_DEQUE_SIZE)
		self._it = 0

	def __repr__(self):
		return f'RandomEnv_{self.obs_dimension}obsx{self.act_dimension}act'

	def reset(self, init_state=None):
		if init_state is None:
			init_state = self.observation_space.sample()

		self.current_state = init_state
		self.reward_deque.clear()
		self._it = 0

		return np.copy(init_state)

	def step(self, action):
		trim_state = self.rm.dot(action)

		self.current_state += trim_state
		r = self.objective(self.current_state)
		self.reward = r
		done, success = self._is_done()

		return self.current_state, r, done, dict(success=success)

	def objective(self, state):
		state_reward = -np.sum(np.square(state)) / self.obs_dimension

		return state_reward

	def _is_done(self):
		self._it += 1

		done = False
		success = False

		# Reach goal
		if len(self.reward_deque) == RandomEnv.REWARD_DEQUE_SIZE and np.max(np.abs(self.current_state
		                                                                           )) <= RandomEnv.GOAL:
			done = True
			success = True
		elif self._it >= RandomEnv.EPISODE_LENGTH_LIMIT:
			done = True

		return done, success

	def get_optimal_action(self, state, scale):
		action = -self.pi.dot(state)
		return action * scale

	@property
	def reward(self):
		return self._reward

	@reward.setter
	def reward(self, r):
		self._reward = r
		self.reward_deque.append(r)

	def seed(self, seed=None):
		np.random.seed(seed)

	def create_model(self):
		n_obs, n_act = self.obs_dimension, self.act_dimension

		# Instantiate left & right singular vectors, and singular value matrices
		u = stats.ortho_group.rvs(n_obs)
		s = np.diag(sorted(np.random.uniform(0, 1, min(n_obs, n_act)), reverse=True))
		vh = stats.ortho_group.rvs(n_act)

		# Padding logic for s
		if n_obs > n_act:
			first_pad = n_obs - n_act
			second_pad = 0
		elif n_act > n_obs:
			first_pad = 0
			second_pad = n_act - n_obs
		else:
			first_pad, second_pad = 0, 0

		# Pad s to match sizes of actions and states
		s = np.pad(s, ((0, first_pad), (0, second_pad)))

		# Get inverse components
		sinv = np.where(1 / s == np.inf, 0, 1 / s).T
		uh = u.T
		v = vh.T

		# Get Response Matrix and its Pseudo-Inverse
		self.rm = u.dot(s.dot(vh))
		self.pi = v.dot(sinv.dot(uh))


if __name__ == '__main__':
	n_obs = 100
	n_act = 100

	env = RandomEnv(n_obs, n_act)

	d = False
	o1 = env.reset()
	o_list = [o1]
	a_list = []

	print(np.linalg.det(env.pi),np.linalg.det(env.rm))

	fig, ax = plt.subplots()
	l, = ax.plot(np.zeros(n_obs))
	plt.show(block=False)
	ax.set_ylim((-1,1))
	while not d:
		a = env.get_optimal_action(o1, 0.1)
		o2, r, d, _ = env.step(a)
		l.set_ydata(o2)
		plt.pause(0.1)

		o_list.append(o2.copy())
		a_list.append(a)
		o1 = np.copy(o2)

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
