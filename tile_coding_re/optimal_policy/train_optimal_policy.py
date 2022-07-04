import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from tqdm import trange
import pandas as pd
from collections import deque
from itertools import product

from tile_coding_re.tile_coding import get_tilings_from_env, QValueFunction
from tile_coding_re.utils import TrajBuffer
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions


def play_ep_get_obs():
    obs = []
    o = np.random.uniform(-0.8, 0.8, 2)
    env.reset(o)
    d = False
    while not d:
        a = qvf.greedy_action(o)
        otp1, r, d, _ = env.step(a)

        buffer.add(o, a, r)
        o = otp1
        obs.append(o.copy())
    obs = np.array(obs).T
    return obs[0], obs[1]


np.random.seed(123)

env = RandomEnvDiscreteActions(2, 2)
tilings = get_tilings_from_env(env, 8, 8, 1.2)
actions = get_discrete_actions(2)
qvf = QValueFunction(tilings, actions, 1e-2)
buffer = TrajBuffer()

nb_bins = 51
tracking_abslim = 1.2
tracking_states = np.array([list(item) for item in product(*np.repeat(
    np.linspace(-tracking_abslim, tracking_abslim, nb_bins).reshape(1, -1), 2, axis=0))])
nb_tracked = tracking_states.shape[0]

nb_eps = 500
eval_step = 100
heatmaps_at = [1, *np.arange(eval_step, nb_eps + 1, eval_step)]
# heatmaps_at = [nb_eps]

tracked_vals = np.zeros((nb_eps, nb_tracked))
tracked_returns_sum = np.zeros(nb_tracked)
tracked_returns_counts = np.zeros(nb_tracked)

errors = []

nb_discretization_bins = 21
state_discretization = np.linspace(-1.2, 1.2, nb_discretization_bins - 1)
visited_states = np.zeros((env.obs_dimension, nb_discretization_bins))

matched_trained_optimal = deque(maxlen=100)
try:
    for ep in trange(nb_eps):
        buffer.reset()
        o = np.random.uniform(-1.5, 1.5, 2)
        env.reset(o)
        d = False
        while not d:
            # exploration
            # if np.random.rand() < 0.1:
            #     a = list(env.action_space.sample())
            # else:
            a = list(env.get_optimal_action(o))
            otp1, r, d, _ = env.step(a)
            # if r < -1:
            #     r = -10

            buffer.add(o, a, r)
            o = otp1

            # # state visitation - when optimal policy used, centered around zero - not interesting
            # for dim, o_ in enumerate(otp1):
            #     bidx = np.digitize(o_, state_discretization)
            #     visited_states[dim, bidx] += 1

            # early termination - if we can choose the optimal action many consecutive times = success
            if sum(matched_trained_optimal) == matched_trained_optimal.maxlen:
                print(f'{matched_trained_optimal.maxlen} consecutive optimal predictions')
                raise ('Training done')
            elif a == list(qvf.greedy_action(o)):
                matched_trained_optimal.append(1)
            else:
                matched_trained_optimal.append(0)

        while buffer:
            o, a, g = buffer.pop_target_tuple()
            # errors.append(g - qvf.value(o, a))
            closes_idx = np.argmin(np.mean(np.abs(np.subtract(tracking_states, o)), axis=1))
            tracked_returns_sum[closes_idx] += g
            tracked_returns_counts[closes_idx] += 1
            qvf.update(o, a, g)

        # every episode, track values around the state-space
        if ep + 1 in heatmaps_at:
            for i, ts in enumerate(tracking_states):
                tracked_vals[ep, i] = np.mean([qvf.value(ts, a_) for a_ in actions])
except Exception as e:
    print(e)

# Plot heatmap of average returns
tracked_returns = np.divide(tracked_returns_sum, tracked_returns_counts).reshape(nb_bins, nb_bins)
# tracked_returns = np.where(np.isnan(tracked_returns),
#                            np.tile(np.nanmin(tracked_returns), tracked_returns.shape),
#                            tracked_returns)

fig, axhm = plt.subplots()
fig.suptitle('Average returns')
cmap = 'gist_stern'
im = axhm.imshow(tracked_returns, extent=(-tracking_abslim, tracking_abslim, -tracking_abslim, tracking_abslim),
                 cmap=cmap)
# norm=SymLogNorm(linthresh=1e-2), cmap=cmap)
fig.colorbar(im, ax=axhm)
fig.tight_layout()

for i in heatmaps_at:
    fighm, axhm = plt.subplots()
    fighm.suptitle(f'Estimated value - Episode {i}')
    vals = tracked_vals[i - 1].reshape(nb_bins, nb_bins)
    im = axhm.imshow(vals, extent=(-tracking_abslim, tracking_abslim, -tracking_abslim, tracking_abslim), cmap=cmap,
                     aspect='equal')
    for _ in range(10):
        state1, state2 = play_ep_get_obs()
        axhm.plot(state1, state2, marker='x')
        axhm.scatter(state1[0], state2[0], marker='o', c='k')
    axhm.set_ylim((-tracking_abslim, tracking_abslim))
    axhm.set_xlim((-tracking_abslim, tracking_abslim))
    # norm=SymLogNorm(linthresh=1e-2), cmap=cmap)
    fighm.colorbar(im, ax=axhm)
    fighm.tight_layout()

# plt.show()

# # Plot visited states histogram
# fig, ax = plt.subplots()
# min_state, max_state = state_discretization[0], state_discretization[-1]
# barx = np.linspace(min_state, max_state, nb_discretization_bins)
# bar_width = np.mean(np.diff(state_discretization))
# ax.bar(barx, visited_states[0], width=bar_width)
# ax.bar(barx, visited_states[1], width=bar_width, alpha=0.5)
# ax.set_ylabel('Visited states counter')
#
# Plot q-values of tracked states
# fig, ax = plt.subplots()
# ax.plot(tracked_vals)
# for ts, tv in zip(tracking_states, tracked_vals[-1]):
#     print(f'State={ts}\tVal={tv}')

# # Plot MC error during training
# fig, ax = plt.subplots()
# # ax.plot(errors)
# s = pd.Series(errors)
# errors_smooth1 = s.rolling(100).mean().to_numpy()
# errors_smooth2 = s.rolling(500).mean().to_numpy()
# ax.plot(errors_smooth1)
# ax.plot(errors_smooth2)


plt.show()
