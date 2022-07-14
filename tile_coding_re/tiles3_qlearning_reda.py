import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pandas import Series
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions, VREDA
from tile_coding_re.heatmap_utils import make_heatmap, update_heatmap
# from tile_coding_re.training_utils import lr
from tile_coding_re.buffers import TrajBuffer
import gym
from copy import deepcopy

'''
Environment info
'''
n_obs = 1
n_act = 1
env = VREDA(n_obs, n_act, estimate_scaling=False)
# env = REDA(n_obs, n_act)
# eval_env = VREDA(n_obs, n_act, model_info=env.model_info)
eval_env = deepcopy(env)

'''
Tiling info
'''
nb_tilings = 8
nb_bins = 2

ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)] + [[0.0, 0.1]]
max_tiles = 2 ** 20

tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
actions = get_discrete_actions(n_act, 3)

'''
Hyper parameters
'''
init_lr = 1.5e-1
lr_decay_rate = 0.9
lr_decay_every_eps = 10
lr_fun = lambda ep_i: init_lr * lr_decay_rate**(ep_i//lr_decay_every_eps)
lr_str = f'Step decay LR: {init_lr}x{lr_decay_rate}^(ep_idx//{lr_decay_every_eps})'

# Step-wise decaying exploration
init_exploration = 0.5
exploration_decay_rate = 0.9
exploration_decay_every_eps = 10
exploration_fun = lambda ep_i: init_exploration * exploration_decay_rate**(ep_i//exploration_decay_every_eps)
exploration_str = f'Step decay EXP: {init_exploration}x{exploration_decay_rate}^(ep_idx//{exploration_decay_every_eps})'

gamma = 0.9

nb_eps = 500
eval_every_t_timesteps = 250

# Training counters
T = 0
ep = 0

'''
Q-function tables and training methods
'''
qvf1 = QValueFunctionTiles3(tilings, actions)#, lr)
qvf2 = QValueFunctionTiles3(tilings, actions)#, lr)


def swap_q():
    if np.random.rand() < 0.5:
        return qvf1, qvf2
    else:
        return qvf2, qvf1


def get_qvals(state, qvf):
    return [qvf.value(state, a_) for a_ in actions]


def get_total_greedy_action(state):
    global actions, qvf1, qvf2
    val1 = get_qvals(state, qvf1)
    val2 = get_qvals(state, qvf2)
    tot_val = [v1 + v2 for v1, v2 in zip(val1, val2)]
    action_idx = np.argmax(tot_val)

    return actions[action_idx]


def play_ep_get_obs():
    obs = []
    o = eval_env.reset()
    _d = False
    while not _d:
        a = get_total_greedy_action(o)
        otp1, r, _d, _ = eval_env.step(a)

        o = otp1.copy()
        obs.append(o)
    obs = np.array(obs).T
    return obs[0], obs[1]


'''
Initialise the states that will have their values and visitations tracked
'''
dim_size = int(nb_bins * nb_tilings)
tracking_lim = 1.0
# tracking_ranges = np.multiply(ranges[:n_obs], tracking_lim)
tracking_ranges = np.multiply(ranges, tracking_lim)
tracking_states = np.array([list(item) for item in product(*np.array([np.linspace(l, h, dim_size) for l, h in tracking_ranges]))])
nb_tracked = tracking_states.shape[0]
reshape_to_map = lambda arr: arr.reshape(dim_size, dim_size).T

# Arrays used to store heatmap info
tracked_vals = np.zeros(nb_tracked)
state_visitation = np.repeat(np.nan, nb_tracked)

'''
Initialise all plots
'''
suptitle_suffix = f'Tilings: {nb_tilings} - Bins: {nb_bins}\n' \
                  f'Gamma: {gamma}\n' \
                  f'{lr_str}\n' \
                  f'{exploration_str}'
plt.ion()
figsize = (10, 8)
gs_kw = dict(width_ratios=[1, 1], height_ratios=[3, 1, 1])
fig, axd = plt.subplot_mosaic([['vals', 'sv'],
                               ['cum_rew', 'cum_rew'],
                               ['err', 'err']],
                              gridspec_kw=gs_kw, figsize=figsize,
                              constrained_layout=True)
ax_val = axd['vals']
ax_sv = axd['sv']
ax_cum_rew = axd['cum_rew']
ax_err = axd['err']

nb_heatmaps = 2
fig.suptitle(f'\n\n{suptitle_suffix}')

# Initialise heatmaps
im_val = make_heatmap(ax_val, reshape_to_map(tracked_vals), *tracking_ranges, '\n')
ax_val.set_title('Estimated values')
im_sv = make_heatmap(ax_sv, reshape_to_map(state_visitation), *tracking_ranges, '\n')
ax_sv.set_title('State visitation')

for ax in (ax_val, ax_sv):
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    if n_obs > 1:
        ax.add_patch(plt.Circle((0., 0.), env.GOAL, edgecolor='g', facecolor='None', label='Threshold'))
    else:
        ax.axvline(-env.GOAL, c='g', label='Threshold')
        ax.axvline(env.GOAL, c='g')

# fig.tight_layout()

# Initialise sample episode traces
nb_eval_eps = 20
ep_lines = [[] for _ in range(nb_heatmaps)]
ep_starting_points = [[] for _ in range(nb_heatmaps)]
ep_terminal_points = [[] for _ in range(nb_heatmaps)]
for ep_nb in range(nb_eval_eps):
    for i, ax in enumerate((ax_val,)):
        if ep_nb == 0:
            label_line = 'Episode traces'
            label_point_start = 'Start state'
            label_point_finish = 'Terminal state'
        else:
            label_line = None
            label_point_start = None
            label_point_finish = None

        ep_lines[i].append(ax.plot([], [], marker='x', c='m', lw=0.1, label=label_line)[0])
        ep_starting_points[i].append(ax.plot([], [], c='k', marker='o', markersize=2, label=label_point_start)[0])
        ep_terminal_points[i].append(ax.plot([], [], c='r', marker='s', markersize=4, label=label_point_finish)[0])
for ax in (ax_val, ax_sv):
    ax.legend(loc='best', prop=dict(size=8))

# Tracking errors
cum_rew_line, = ax_cum_rew.plot([], [])
ax_cum_rew.set_ylabel('Total rewards')
cum_rews = []

errors_line1, = ax_err.plot([], [])
errors_line2, = ax_err.plot([], [])
ax_err.set_ylabel('TD errors')
errors1 = []
errors2 = []

for ax in (ax_cum_rew, ax_err):
    ax.set_xlabel('Training episodes')
    ax.set_yscale('symlog')

'''
Evaluation info
'''
is_eval = lambda: (T + 1) % eval_every_t_timesteps == 0 or T == 0
lr, exploration = None, None
def eval():
    fig.suptitle(f'Episode {ep+1:4d} - Step {T+1:4d}\nIHT count {tilings.count():6d}\n'
                 f'LR={lr:.4f}\tEXP={exploration:.2f}\n{suptitle_suffix}')

    # Update estimated value heatmaps
    for j, ts in enumerate(tracking_states):
        ts = np.concatenate([ts, [np.random.rand()*0.15]])
        v = max([(v1 + v2)/2. for v1, v2 in zip(get_qvals(ts, qvf1), get_qvals(ts, qvf2))])
        tracked_vals[j] = v
    update_heatmap(im_val, reshape_to_map(tracked_vals))

    # Update state visitation heatmaps
    update_heatmap(im_sv, reshape_to_map(state_visitation))

    # Update with fresh episode traces
    states = [play_ep_get_obs() for _ in range(nb_eval_eps)]
    for i in range(1):
        for j, state in enumerate(states):
            ep_lines[i][j].set_data(state[0], state[1])
            ep_starting_points[i][j].set_data(state[0][0], state[1][0])
            ep_terminal_points[i][j].set_data(state[0][-1], state[1][-1])

    # Update cumulative reward plot
    cum_rews_averaging_window = 1
    if len(cum_rews) > cum_rews_averaging_window:
        cum_rews_mean = Series(cum_rews).rolling(cum_rews_averaging_window).mean().to_numpy().tolist()
        cum_rew_line.set_data(range(len(cum_rews_mean)), cum_rews_mean)
        ax_cum_rew.set_xlim((0, len(cum_rews_mean)))
        ax_cum_rew.set_ylim((np.nanmin(cum_rews_mean), np.nanmax(cum_rews_mean)))

    # Update error plot
    error_averaging_window = 1
    errors1_mean = Series(errors1).rolling(error_averaging_window).mean().to_numpy().tolist()
    errors2_mean = Series(errors2).rolling(error_averaging_window).mean().to_numpy().tolist()
    errors_line1.set_data(range(len(errors1)), errors1_mean)
    errors_line2.set_data(range(len(errors2)), errors2_mean)
    if len(errors1) > error_averaging_window and len(errors2) > error_averaging_window:
        ax_err.set_xlim((0, max([len(errors1), len(errors2)])))
        ax_err.set_ylim((np.nanmin(errors1_mean + errors2_mean), np.nanmax(errors1_mean + errors2_mean)))
    plt.pause(0.1)


def update_state_visitations(state):
    # state = state[:n_obs]
    if sum(np.abs(state) > tracking_lim) == 0:
        sv_index = np.argmin(np.mean(np.abs(np.subtract(tracking_states, state)), axis=1))
        if np.isnan(state_visitation[sv_index]):
            state_visitation[sv_index] = 1
        else:
            state_visitation[sv_index] += 1

'''
Training
'''
for ep in trange(nb_eps):
    o = env.reset()
    done = False
    cum_rew = 0.0

    err1, err2 = 0.0, 0.0
    step = 0
    while not done:
        # Explore or exploit
        exploration = exploration_fun(ep)
        if np.random.rand() < exploration:
            a = env.action_space.sample().tolist()
        else:
            a = get_total_greedy_action(o)

        # Step in environment dynamics
        otp1, r, done, info = env.step(a)

        # Calculate targets and update Q-functions
        qvfa, qvfb = swap_q()
        target = r + gamma * qvfb.value(otp1, qvfa.greedy_action(otp1))

        lr = lr_fun(ep)
        error = qvfa.update(o, a, target, lr)

        # Increment cumulative reward
        cum_rew += r

        # Increment state visitations for heatmap
        update_state_visitations(o)

        # Log errors for plotting
        if qvfa == qvf1:
            err1 += error
        else:
            err2 += error

        # Cycle variables and increment counters
        o = otp1.copy()
        T += 1
        step += 1

        # Evaluation phase
        if is_eval() or ((ep == nb_eps - 1) and done):
            eval()
    cum_rews.append(cum_rew)
    errors1.append(err1)
    errors2.append(err2)

plt.ioff()
# plt.savefig(os.path.join('results', f'doubleQLearning_mountainCar_{nb_tilings}Tilings_{nb_bins}Bins.png'))
plt.show()
