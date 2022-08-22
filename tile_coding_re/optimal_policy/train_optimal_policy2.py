import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from collections import deque
from itertools import product

from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from rl_utils.buffers import TrajBuffer
from training_utils import lr, play_ep_get_obs
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions

np.random.seed(123)

n_obs, n_act = 2, 2
env = RandomEnvDiscreteActions(n_obs, n_act)

nb_tilings = 16
nb_tiling_bins = 4
ranges = [[-1., 1.] for _ in range(n_obs)]
actions = get_discrete_actions(n_act)
tilings = Tilings(nb_tilings, nb_tiling_bins, ranges, 2 ** 23)

lr = lr(1e-1, 0)
qvf = QValueFunctionTiles3(tilings, actions, lr)


def play_ep_get_obs(env, policy):
    obs = []
    # o = np.random.uniform(-0.8, 0.8, 2)
    o = env.reset()
    d = False
    while not d:
        a = policy(o)
        otp1, r, d, _ = env.step(a)

        o = otp1
        obs.append(o.copy())
    obs = np.array(obs).T
    return obs[0], obs[1]


def perform_ep_len_stats():
    ep_lens = []
    for _ in range(100):
        ep_lens.append(len(play_ep_get_obs(env, qvf.greedy_action)[0]))
    return np.mean(ep_lens), np.std(ep_lens)


# Set up grid of states for heatmaps
nb_bins = 21
tracking_abslim = 1.7
tracking_states = np.array([list(item) for item in product(*np.repeat(
    np.linspace(-tracking_abslim, tracking_abslim, nb_bins).reshape(1, -1), 2, axis=0))])
nb_tracked = tracking_states.shape[0]

# Training and evaluation info
nb_eps = 500
eval_step = 10
heatmaps_at = [1, *np.arange(eval_step, nb_eps + 1, eval_step)]
heatmaps_at_static = [1, nb_eps]
# heatmaps_at = [nb_eps]

# Arrays used to store heatmap info
tracked_vals = np.zeros((nb_eps, nb_tracked))
tracked_returns_sum = np.zeros(nb_tracked)
tracked_returns_counts = np.zeros(nb_tracked)

# errors = []

# Discretize state space for visitation frequency stats
nb_discretization_bins = 21
state_discretization = np.linspace(-1.2, 1.2, nb_discretization_bins - 1)
visited_states = np.zeros((env.obs_dimension, nb_discretization_bins))

# For early termination - if greedy action == optimal action, append 1, else 0
matched_trained_optimal = deque(maxlen=100)

# Initialize heatmaps for interative plotting - overkill? No.
cmap = 'gist_stern'
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.ion()
im_returns = ax1.imshow(tracked_returns_sum.reshape(nb_bins, nb_bins), cmap=cmap,
                        extent=(-tracking_abslim, tracking_abslim, -tracking_abslim, tracking_abslim))
fig.colorbar(im_returns, ax=ax1)
ax1.set_title('Actual returns')
im_estimates = ax2.imshow(tracked_vals[0].reshape(nb_bins, nb_bins), cmap=cmap,
                          extent=(-tracking_abslim, tracking_abslim, -tracking_abslim, tracking_abslim))
fig.colorbar(im_estimates, ax=ax2)
fig.tight_layout()

buffer = TrajBuffer()
min_g, max_g = np.inf, -np.inf  # for actual return heatmap colorbar scale
ep_lens_training = []
try:  # Early termination raises Exception
    for ep in trange(nb_eps):
        ep_len = 0
        buffer.reset()

        o = np.random.uniform(-1., 1., 2)
        env.reset(o)
        d = False
        while not d:
            # exploration
            if np.random.rand() < np.exp(-ep / 50):  # 0.1:
                a = list(env.action_space.sample())
            else:
                a = qvf.greedy_action(o)
            # a = list(env.get_optimal_action(o))
            otp1, r, d, _ = env.step(a)
            # if r < -1:
            #     r = -10

            buffer.add(o, a, r)
            o = otp1
            ep_len += 1

        # # state visitation - when optimal policy used, centered around zero - not interesting
        # for dim, o_ in enumerate(otp1):
        #     bidx = np.digitize(o_, state_discretization)
        #     visited_states[dim, bidx] += 1

        # early termination - if we can choose the optimal action many consecutive times = success
        # if sum(matched_trained_optimal) == matched_trained_optimal.maxlen:
        #     print(f'{matched_trained_optimal.maxlen} consecutive optimal predictions')
        #     raise 'Training done'
        # elif a == list(qvf.greedy_action(o)):
        #     matched_trained_optimal.append(1)
        # else:
        #     matched_trained_optimal.append(0)

        # if (ep+1)%eval_step==0:
        #     ep_lens_training.append(perform_ep_len_stats())

        while buffer:  # consume buffer
            o, a, g = buffer.pop_target_tuple()
            qvf.update(o, a, g)

            min_g = min((min_g, g))
            max_g = max((max_g, g))

            # errors.append(g - qvf.value(o, a))

            # Update returns information for heatmap
            closes_idx = np.argmin(np.sum(np.abs(np.subtract(tracking_states, o)), axis=1))
            tracked_returns_sum[closes_idx] += g
            tracked_returns_counts[closes_idx] += 1

        # Track values around the state-space - for heatmaps
        if ep + 1 in heatmaps_at:
            for i, ts in enumerate(tracking_states):
                tracked_vals[ep, i] = np.mean([qvf.value(ts, a_) for a_ in actions])
            # tracked_vals[ep, i] = np.max([qvf.value(ts, a_) for a_ in actions])

            val_array = tracked_vals[ep].reshape(nb_bins, nb_bins)
            im_estimates.set_data(val_array)
            im_estimates.set_clim((np.min(val_array), np.max(val_array)))

            ret_array = np.divide(tracked_returns_sum, tracked_returns_counts).reshape(nb_bins, nb_bins)
            im_returns.set_data(ret_array)
            im_returns.set_clim((ret_array.min(), ret_array.max()))
            ax2.set_title(f'Estimated returns\nTraining ep: {ep + 1:4d}\nNb tiles create: {tilings.count}')
            plt.pause(0.1)
except Exception as e:
    print(e)

plt.ioff()

fig, ax = plt.subplots()
ax.plot(ep_lens_training)
ax.set_xlabel('Training episodes')
ax.set_ylabel('Training episode lengths')

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
fig.colorbar(im, ax=axhm)
fig.tight_layout()

for i in heatmaps_at_static:
    fighm, axhm = plt.subplots()
    fighm.suptitle(f'Estimated value - Episode {i}')
    vals = tracked_vals[i - 1].reshape(nb_bins, nb_bins)
    im = axhm.imshow(vals, extent=(-tracking_abslim, tracking_abslim, -tracking_abslim, tracking_abslim), cmap=cmap,
                     aspect='equal')
    for _ in range(2):
        state1, state2 = play_ep_get_obs(env, qvf.greedy_action)
        lim_ep_len = 20
        state1, state2 = state1[:lim_ep_len], state2[:lim_ep_len]
        axhm.plot(state1, state2, lw=2, marker='x')
        axhm.scatter(state1[0], state2[0], s=50, marker='o', c='k')
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
