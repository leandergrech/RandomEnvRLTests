import os
import numpy as np
import matplotlib.pyplot as plt

from tile_coding_re.tile_coding import get_tilings_from_env, QValueFunction2
from tile_coding_re.buffers import TrajBuffer
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions

env = RandomEnvDiscreteActions(2, 2)
env.TRIM_FACTOR = 4.5
actions = get_discrete_actions(2)

buffer = TrajBuffer()

fig, (ax1, ax2) = plt.subplots(2, gridspec_kw=dict(height_ratios=[2,1]))
fig.suptitle(f'{repr(env)}\nTrims scale = {1/env.TRIM_FACTOR:.2f}')
ax1.add_patch(plt.Circle([0,0], env.GOAL, edgecolor='g', facecolor='None', zorder=15))
ax1.set_xlim((-1, 1))
ax1.set_ylim((-1, 1))
ax2.set_ylim((1, 20))
ep_lens = []
for ep in range(100):
    buffer.reset()
    o = env.reset()
    d = False
    while not d:
        a = env.get_optimal_action(o)
        otp1, r, d, _ = env.step(a)

        buffer.add(otp1, a, r)
        o = otp1

    obses = np.array(buffer.o).T
    ep_lens.append(obses.shape[1])
    ax1.plot(obses[0], obses[1], lw=0.5,  zorder=10)
    ax1.scatter(obses[0][0], obses[1][0], marker='o', s=1, c='k', zorder=15)
    ax1.scatter(obses[0][-1], obses[1][-1], marker='x', s=4, c='k', zorder=15)

ax1.legend(handles=[plt.Line2D([],[], marker='o', c='k'), plt.Line2D([],[], marker='x', c='k')], labels=['Episode start', 'Episode end'])

ax1.set_xlabel('State[0]')
ax1.set_ylabel('State[1]')
ax2.scatter(range(len(ep_lens)), ep_lens)
ax2.set_ylabel('Episode length')
ax2.set_xlabel('Episode')
fig.tight_layout()

plt.show()
