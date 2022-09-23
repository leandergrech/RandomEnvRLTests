from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pandas import Series
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from random_env.envs import get_discrete_actions
from utils.heatmap_utils import make_heatmap, update_heatmap
# from tile_coding_re.training_utils import lr
import gym

"""
Test script using Tilings and QValueFunctionTiles3 classes with MountainCar environment and Q-learning.
Detailed real-time plotting during training.
"""

env = gym.make('MountainCar-v0')
eval_env = gym.make('MountainCar-v0')

n_obs = env.observation_space.shape[0]
n_act = 1

nb_tilings = 16
nb_bins = 4

ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
max_tiles = 2 ** 20

tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
actions = get_discrete_actions(n_act)

# Hyper parameters
# lr = lr(1e-1, 30000)
init_lr = 1e-1
final_lr = 1e-1
lr_decay_eps = 1000
lr_fun = lambda ep_i: final_lr + (init_lr - final_lr) * max(0, (1 - ep_i/lr_decay_eps))
init_exploration = 1.0
final_exploration = 0.0
exploration_decay_eps = 1000
exploration_fun =  lambda ep_i: final_exploration + (init_exploration - final_exploration) * max(0, (1. - ep_i/exploration_decay_eps))
gamma = 0.99

suptitle_suffix = f'Tilings: {nb_tilings} - Bins: {nb_bins}\n' \
                  f'Gamma: {gamma}\n' \
                  f'Linear decay LR:  {init_lr:.2}->{final_lr} in {lr_decay_eps} episodes\n' \
                  f'Linear decay EPS: {init_exploration}->{final_exploration} in {exploration_decay_eps} episodes'

# Training and evaluation info
nb_eps = 2000
eval_every_t_timesteps = 50

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


# Set up grid of states for heatmaps
dim_size = 20
tracking_states = np.array([list(item) for item in product(*np.array([np.linspace(l, h, dim_size) for l, h in ranges]))])
nb_tracked = tracking_states.shape[0]
reshape_to_map = lambda arr: arr.reshape(dim_size, dim_size).T

# Arrays used to store heatmap info
tracked_vals = np.zeros(nb_tracked)
state_visitation = np.repeat(np.nan, nb_tracked)

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

# fig, (ax_val, ax_sv) = plt.subplots(1, 2, figsize=figsize)
nb_heatmaps = 2
fig.suptitle(f'\n\n{suptitle_suffix}')

# Initialise heatmaps
im_val = make_heatmap(ax_val, reshape_to_map(tracked_vals), *ranges, '\n')
ax_val.set_title('Estimated values')
im_sv = make_heatmap(ax_sv, reshape_to_map(state_visitation), *ranges, '\n')
ax_sv.set_title('State visitation')

for ax in (ax_val, ax_sv):
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.axvline(env.goal_position, c='c', label='Goal position')

# fig.tight_layout()

# Initialise sample episode traces
nb_eval_eps = 5
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

        ep_lines[i].append(ax.plot([], [], marker='x', c='m', lw=0.2, label=label_line)[0])
        ep_starting_points[i].append(ax.plot([], [], c='k', marker='o', markersize=2, label=label_point_start)[0])
        ep_terminal_points[i].append(ax.plot([], [], c='k', marker='s', markersize=4, label=label_point_finish)[0])
for ax in (ax_val, ax_sv):
    ax.legend(loc='best')

# Tracking errors
# fig2, ax3 = plt.subplots()
cum_rew_line, = ax_cum_rew.plot([], [])
ax_cum_rew.set_xlabel('Training episodes')
ax_cum_rew.set_ylabel('Cumulative rewards')
cum_rews = []

errors_line1, = ax_err.plot([], [])
errors_line2, = ax_err.plot([], [])
ax_err.set_xlabel('Time steps')
ax_err.set_ylabel('Error=Target-Estimate')
errors1 = []
errors2 = []

# Training counters
T = 0
ep = 0

# Evaluation steps
is_eval = lambda: (T + 1) % eval_every_t_timesteps == 0 or T == 0
def eval():
    fig.suptitle(f'Episode {ep+1:4d} - Step {T+1:4d}\nIHT count {tilings.count():6d}\n{suptitle_suffix}')

    # Update estimated value heatmaps
    for j, ts in enumerate(tracking_states):
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
    error_averaging_window = 10
    errors1_mean = Series(errors1).rolling(error_averaging_window).mean().to_numpy().tolist()
    errors2_mean = Series(errors2).rolling(error_averaging_window).mean().to_numpy().tolist()
    errors_line1.set_data(range(len(errors1)), errors1_mean)
    errors_line2.set_data(range(len(errors2)), errors2_mean)
    if len(errors1) > error_averaging_window and len(errors2) > error_averaging_window:
        ax_err.set_xlim((0, max([len(errors1), len(errors2)])))
        ax_err.set_ylim((np.nanmin(errors1_mean + errors2_mean), np.nanmax(errors1_mean + errors2_mean)))
    plt.pause(0.1)


# Start training
for ep in trange(nb_eps):
    o = env.reset()
    done = False
    cum_rew = 0.0
    while not done:
        exploration = exploration_fun(ep)
        if np.random.rand() < exploration:
            a = env.action_space.sample()
        else:
            a = get_total_greedy_action(o)
        
        otp1, r, done, info = env.step(a)

        qvfa, qvfb = swap_q()
        target = r + gamma * qvfb.value(otp1, qvfa.greedy_action(otp1))

        lr = lr_fun(ep)
        error = qvfa.update(o, a, target, lr)

        # Increment cumulative reward
        cum_rew += r

        # Log state visitations for heatmap
        sv_index = np.argmin(np.mean(np.abs(np.subtract(tracking_states, o)), axis=1))
        temp = np.isnan(state_visitation[sv_index])
        if np.isnan(state_visitation[sv_index]):
            state_visitation[sv_index] = 1
        else:
            state_visitation[sv_index] += 1

        # Log errors for plotting
        if qvfa == qvf1:
            errors1.append(error)
        else:
            errors2.append(error)

        # Cycle variables and increment counters
        o = otp1.copy()
        T += 1

        # Evaluation phase
        if is_eval() or ((ep == nb_eps - 1) and done):
        # if ((ep == nb_eps - 1) and done):
            eval()
    cum_rews.append(cum_rew)
plt.ioff()
# plt.savefig(os.path.join('results', f'doubleQLearning_mountainCar_{nb_tilings}Tilings_{nb_bins}Bins.png'))
plt.show()
