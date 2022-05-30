import os
import numpy as np
from datetime import datetime as dt

import torch as t
import matplotlib.pyplot as plt

from random_env.envs import RandomEnv

def generate_trajectory(env, policy, n_eps):
	n_obs, n_act = env.obs_dimension, env.act_dimension
	max_steps = env.max_steps

	o_list = t.zeros((n_eps, max_steps, n_obs))
	otp1_list = t.zeros((n_eps, max_steps, n_obs))
	a_list = t.zeros((n_eps, max_steps, n_act))
	r_list = t.zeros((n_eps, max_steps))
	l_list = t.zeros(n_eps) # episode lengths

	for ep_idx in range(n_eps):
		o = env.reset()
		for step in range(max_steps):
			a = policy(o)

			o_list[ep_idx, step] = o
			a_list[ep_idx, step] = a
			o,r,d,_ = env.step(a)
			otp1_list[ep_idx, step] = o
			if d:
				break

