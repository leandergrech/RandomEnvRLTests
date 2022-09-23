import numpy as np
import matplotlib.pyplot as plt

from  random_env.envs import RandomEnv

"""
In this script I am showing that we can initialise a RandomEnv (RE) with the same size of
QFBEnv and then use the inverse dynamics to optimally control the environment.
"""

n_obs, n_act = 2, 16
env = RandomEnv(n_obs, n_act)
kp = 10.0
ki = 0.0

fig, axs = plt.subplots(2)

obs_bars = axs[0].bar(np.arange(n_obs), np.zeros(n_obs), facecolor='b')
act_bars = axs[1].bar(np.arange(n_act), np.zeros(n_act), facecolor='r')
axs[0].axhline(-env.GOAL, c='g', ls='--')
axs[0].axhline(env.GOAL, c='g', ls='--')
axs[0].set_ylim((-1, 1))
axs[1].set_ylim((-0.2, 0.2))
plt.ion()


def update_obs_line(obs):
    for bar, ob in zip(obs_bars, obs):
        bar.set_height(ob)


def update_act_line(act):
    for bar, ac in zip(act_bars, act):
        bar.set_height(ac)


o = env.reset([1,1])
o_err = o
update_obs_line(o)
plt.pause(1)
d = False
step = 0
while not d:
    a = env.get_optimal_action(o_err)*0.1
    otp1, r, d, _ = env.step(a)

    o_err = kp * otp1 + ki * (otp1 - o)

    o = otp1
    step += 1

    fig.suptitle(step)
    update_obs_line(otp1)
    update_act_line(a)

    plt.pause(1)

plt.ioff()
plt.show()
