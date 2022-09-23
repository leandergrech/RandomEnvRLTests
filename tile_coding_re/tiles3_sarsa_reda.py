from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from random_env.envs import get_discrete_actions, REDAClip
from utils.heatmap_utils import make_heatmap, update_heatmap
# from tile_coding_re.training_utils import lr
from copy import deepcopy

"""
Test script using Tilings and QValueFunctionTiles3 classes with REDA environment and SARSA.
Detailed real-time plotting during training.
"""

'''
Environment info
'''
n_obs = 2
n_act = 2
env = REDAClip(n_obs, n_act, estimate_scaling=True)
# env.TRIM_FACTOR = 2.
# env.load_dynamics('env_dynamics')

# env = REDA(n_obs, n_act)
# eval_env = VREDA(n_obs, n_act, model_info=env.model_info)
eval_env = deepcopy(env)

'''
Tiling info
'''
nb_tilings = 16
nb_bins = 2

# # VREDA ranges
# ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)] + [[0.0, 0.05]]
# REDA ranges
ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
max_tiles = 2 ** 18

tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
actions = get_discrete_actions(n_act, 3)
n_discrete_actions = len(actions)

'''
Hyper parameters
'''
# init_lr = 1.5e-1
# lr_decay_rate = 0.99
# lr_decay_every_eps = 10
# lr_fun = lambda ep_i: init_lr * lr_decay_rate**(ep_i//lr_decay_every_eps)
# lr_str = f'Step decay LR: {init_lr}x{lr_decay_rate}^(ep_idx//{lr_decay_every_eps})'
lr_val = 1e-1
lr_fun = lambda ep_i: lr_val
lr_str = f'Constant LR: {lr_val}'

# Step-wise decaying exploration
init_exploration = 1.0
exploration_decay_rate = 0.99
exploration_decay_every_eps = 10
exploration_fun = lambda ep_i: init_exploration * exploration_decay_rate**(ep_i//exploration_decay_every_eps)
exploration_str = f'Step decay EXP: {init_exploration}x{exploration_decay_rate}^(ep_idx//{exploration_decay_every_eps})'
# exp_val = 0.2
# exploration_fun = lambda ep_i: exp_val
# exploration_str = f'Constant EXP: {exp_val}'

gamma = 0.9

nb_eps = 200
eval_every_t_timesteps = 4000

# Training counters
T = 0
ep = 0

'''
Q-function tables and training methods
'''
qvf = QValueFunctionTiles3(tilings, n_discrete_actions)#, lr)


def get_qvals(state):
    return [qvf.value(state, a_) for a_ in range(n_discrete_actions)]


def get_greedy_action(state):
    return qvf.greedy_action(state)


def play_ep_get_obs_and_cumrew():
    obs = []
    cumrew = 0.0
    o = eval_env.reset()
    _d = False
    while not _d:
        a = get_greedy_action(o)
        otp1, r, _d, _ = eval_env.step(actions[a])

        cumrew += r

        o = otp1.copy()
        obs.append(o)
    obs = np.array(obs).T
    return obs, cumrew


'''
Initialise the states that will have their values and visitations tracked
'''
tracking_lim = 1.6
dim_size = int(nb_bins * nb_tilings * tracking_lim)
if n_obs == 1:
    tracking_ranges = np.multiply(ranges, tracking_lim)
else:
    tracking_ranges = np.multiply(ranges[:n_obs], tracking_lim)

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
gs_kw = dict(width_ratios=[1, 1], height_ratios=[3, 1, 1, 1])
fig, axd = plt.subplot_mosaic([['vals', 'sv'],
                               ['err', 'err'],
                               ['cum_rew', 'info'],
                               ['eplens', 'info']],
                              gridspec_kw=gs_kw, figsize=figsize,
                              constrained_layout=True)
ax_val = axd['vals']
ax_sv = axd['sv']
ax_cum_rew = axd['cum_rew']
ax_err = axd['err']
ax_eplen = axd['eplens']
ax_info = axd['info']
ep_range = np.arange(nb_eps)
ax_info.plot(ep_range, [exploration_fun(ep) for ep in ep_range], c='r')
ax_info.yaxis.label.set_color('r')
ax_info.twinx().plot(ep_range, [lr_fun(ep) for ep in ep_range], c='k')
ax_info.legend(handles=[plt.Line2D([],[],c='k'), plt.Line2D([],[],c='r')], labels=['LR', 'EXP'])
ax_info.set_xlabel('Training Episodes')
info_line = ax_info.axvline(0.0)

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
nb_eval_eps = 10
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

        # ep_lines[i].append(ax.plot([], [], c='m', lw=0.1, label=label_line, marker='x')[0])
        ep_lines[i].append(ax.plot([], [], c='m', lw=0.5, label=label_line)[0])
        ep_starting_points[i].append(ax.plot([], [], c='k', marker='o', markersize=2, label=label_point_start)[0])
        ep_terminal_points[i].append(ax.plot([], [], c='r', marker='s', markersize=2, label=label_point_finish)[0])
# for ax in (ax_val, ax_sv):
#     ax.legend(loc='best', prop=dict(size=8))

# Create lines
cum_rew_line, = ax_cum_rew.plot([], [])
ax_cum_rew.set_ylabel('Total rewards')
ax_cum_rew.set_xlabel('Time steps')
cum_rews = []

errors_line1, = ax_err.plot([], [])
errors_line2, = ax_err.plot([], [])
ax_err.set_ylabel('TD errors')
ax_err.set_xlabel('Training episodes')
errors1 = []

eplens_line, = ax_eplen.plot([], [])
ax_eplen.set_ylabel('Ep. lens')
ax_eplen.set_xlabel('Time steps')
ax_eplen.set_ylim((-10, 110))
eplens = []


for ax in (ax_cum_rew, ax_err):
    ax.set_yscale('symlog')

'''
Evaluation info
'''
is_eval = lambda: (T + 1) % eval_every_t_timesteps == 0 or T == 0
lr, exploration = None, None


def update_line(l, v, x=None):
    if x is None:
        x = range(len(v))
    l.set_data(x, v)
    l.set_ylim((np.nanmin(v), np.nanmax(v)))
    l.set_xlim((x[0], x[-1]))


def eval():
    fig.suptitle(f'Episode {ep+1:4d} - Step {T+1:4d}\nIHT count {tilings.count():6d}\n'
                 f'LR={lr:.4f}\tEXP={exploration:.2f}\n{suptitle_suffix}')

    # Update estimated value heatmaps
    for j, ts in enumerate(tracking_states):
        vals = []
        for vel in np.linspace(0.0, 0.15, 5):
            ts = np.concatenate([ts, [vel]])
            vals.append(max([v for v in get_qvals(ts)]))
        tracked_vals[j] = np.mean(vals)
    update_heatmap(im_val, reshape_to_map(tracked_vals))

    # Update state visitation heatmaps
    update_heatmap(im_sv, reshape_to_map(state_visitation))

    # Update with fresh episode traces
    temp_eplens = []
    temp_cum_rews = []
    for j in range(nb_eval_eps):
        state, cum_rew = play_ep_get_obs_and_cumrew()
        i = 0
        ep_lines[i][j].set_data(state[0], state[1])
        ep_starting_points[i][j].set_data(state[0][0], state[1][0])
        ep_terminal_points[i][j].set_data(state[0][-1], state[1][-1])

        temp_cum_rews.append(cum_rew)
        temp_eplens.append(len(state[0]))

    eplens.append(np.mean(temp_eplens))
    cum_rews.append(np.mean(temp_cum_rews))

    if len(errors1) > 0:
        xrange_train = np.arange(len(errors1))
        errors_line1.set_data(xrange_train, errors1)
        ax_err.set_xlim((0, xrange_train[-1]))
        ax_err.set_ylim((min(errors1), max(errors1)))

    info_line.set_xdata(len(errors1))

    xrange_eval = np.arange(len(eplens)) * eval_every_t_timesteps
    eplens_line.set_data(xrange_eval, eplens)
    ax_eplen.set_xlim((0, T+1))
    ax_eplen.set_ylim((0, 100))

    xrange_eval = np.arange(len(cum_rews)) * eval_every_t_timesteps
    cum_rew_line.set_data(xrange_eval, cum_rews)
    ax_cum_rew.set_xlim((0, T+1))
    ax_cum_rew.set_ylim((min(cum_rews), max(cum_rews)))

    # # Update cumulative reward plot
    # cum_rews_averaging_window = 1
    # if len(cum_rews) > cum_rews_averaging_window:
    #     cum_rews_mean = Series(cum_rews).rolling(cum_rews_averaging_window).mean().to_numpy().tolist()
    #     cum_rew_line.set_data(xrange[-len(cum_rews_mean):], cum_rews_mean)
    #     ax_cum_rew.set_ylim((np.nanmin(cum_rews_mean), np.nanmax(cum_rews_mean)))
    #
    # # Update ep lens plot
    # eplens_averaging_window = 1
    # if len(eplens) > eplens_averaging_window:
    #     eplens_mean = Series(eplens).rolling(eplens_averaging_window).mean().to_numpy().tolist()
    #     eplens_line.set_data(xrange[-len(eplens_mean):], eplens_mean)
    #     ax_eplen.set_ylim((np.nanmin(eplens_mean), np.nanmax(eplens_mean)))

    # # Update error plot
    # error_averaging_window = 1
    # errors1_mean = Series(errors1).rolling(error_averaging_window).mean().to_numpy().tolist()
    # errors2_mean = Series(errors2).rolling(error_averaging_window).mean().to_numpy().tolist()
    # errors_line1.set_data(range(len(errors1)), errors1_mean)
    # errors_line2.set_data(range(len(errors2)), errors2_mean)
    # if len(errors1) > error_averaging_window and len(errors2) > error_averaging_window:
    #     ax_err.set_xlim((0, max([len(errors1), len(errors2)])))
    #     ax_err.set_ylim((np.nanmin(errors1_mean + errors2_mean), np.nanmax(errors1_mean + errors2_mean)))
    plt.pause(0.1)


def update_state_visitations(state):
    if n_obs > 1:
        state = state[:n_obs]
    if sum(np.abs(state) > tracking_lim) == 0:
        sv_index = np.argmin(np.mean(np.abs(np.subtract(tracking_states, state)), axis=1))
        if np.isnan(state_visitation[sv_index]):
            state_visitation[sv_index] = 1
        else:
            state_visitation[sv_index] += 1

'''
Training
'''
def obs_init_func():
    r = np.random.normal(0.8, 0.1)
    theta = 2 * np.pi * np.random.rand()
    return np.array([r * np.cos(theta), r * np.sin(theta)])


for ep in trange(nb_eps):
    o = env.reset(obs_init_func())
    a = get_greedy_action(o)
    done = False
    cum_rew = 0.0

    err1 = 0.0
    step = 0
    while not done:
        # Step in environment dynamics
        otp1, r, done, info = env.step(actions[a])

        # Explore or exploit
        exploration = exploration_fun(ep)
        if np.random.rand() < exploration:
            a_ = np.random.choice(n_discrete_actions)
        else:
            a_ = get_greedy_action(o)

        # if step < env.EPISODE_LENGTH_LIMIT - 1 and done:
        #     r = 100.
        # else:
        #     r = -1.

        # Calculate targets and update Q-functions
        target = r + gamma * qvf.value(otp1, a_)

        lr = lr_fun(ep)
        error = qvf.update(o, a, target, lr)

        # Increment cumulative reward
        cum_rew += r

        # Increment state visitations for heatmap
        update_state_visitations(o)

        # Log errors for plotting
        err1 += error

        # Cycle variables and increment counters
        o = otp1.copy()
        a = a_
        T += 1
        step += 1

        # Evaluation phase
        if is_eval() or ((ep == nb_eps - 1) and done):
            eval()
    cum_rews.append(cum_rew)
    errors1.append(err1)

plt.ioff()
# plt.savefig(os.path.join('results', f'doubleQLearning_mountainCar_{nb_tilings}Tilings_{nb_bins}Bins.png'))
plt.show()
