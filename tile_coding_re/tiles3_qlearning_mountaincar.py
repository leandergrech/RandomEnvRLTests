from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions
from tile_coding_re.heatmap_utils import make_heatmap, update_heatmap
from tile_coding_re.training_utils import lr
from tile_coding_re.buffers import TrajBuffer
import gym

env = gym.make('MountainCar-v0')
n_obs = env.observation_space.shape[0]
n_act = 1

nb_tilings = 16
nb_bins = 4
ranges = [[l, h] for l, h in zip(env.observation_space.low,
                                 env.observation_space.high)]
max_tiles = 2 ** 20

tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
actions = get_discrete_actions(n_act)
lr = lr(1e-2, 0)
gamma = 0.99

qvf1 = QValueFunctionTiles3(tilings, actions, lr)
qvf2 = QValueFunctionTiles3(tilings, actions, lr)


def swap_q():
    if np.random.rand() < 0.5:
        return qvf1, qvf2
    else:
        return qvf2, qvf1


def get_total_greedy_action(state):
    global actions
    val1 = [qvf1.value(state, a_) for a_ in actions]
    val2 = [qvf2.value(state, a_) for a_ in actions]
    tot_val = [v1 + v2 for v1, v2 in zip(val1, val2)]
    action_idx = np.argmax(tot_val)

    return actions[action_idx][0], val1, val2


def play_ep_get_obs():
    obs = []
    o = env.reset()
    d = False
    while not d:
        a = get_total_greedy_action(o)[0]
        otp1, r, d, _ = env.step(a)

        o = otp1
        obs.append(o.copy())
    obs = np.array(obs).T
    return obs[0], obs[1]


# Set up grid of states for heatmaps
dim_size = 30

tracking_states = np.array(
    [list(item) for item in product(*np.array([np.linspace(l, h, dim_size) for l, h in ranges]))])
nb_tracked = tracking_states.shape[0]
reshape_to_map = lambda arr: arr.reshape(dim_size, dim_size)

# Training and evaluation info
nb_eps = 100
eval_every = 1

# Arrays used to store heatmap info
nb_heatmaps = 1  # len(actions)
tracked_vals = np.zeros((nb_heatmaps, nb_tracked))

plt.ion()
figsize = (10, 10)
axs_square_side = int(np.ceil(np.sqrt(nb_heatmaps)))
fig, axs = plt.subplots(figsize=figsize)
fig.suptitle('\n\n')

axs.set_xlabel('Position')
axs.set_ylabel('Velocity')

axs = np.array([axs])
# axs = axs.ravel()[:nb_heatmaps]
# Initialise heatmaps
im_list = []
for i, ax in enumerate(axs):
    im = make_heatmap(ax, reshape_to_map(tracked_vals[i]), *ranges, '\n')
    im_list.append(im)

axs[0].axvline(env.goal_position, c='c', label='Goal position')
fig.tight_layout()

# Initialise sample episode traces
nb_eval_eps = 10
ep_lines = [[] for _ in range(nb_heatmaps)]
ep_starting_points = [[] for _ in range(nb_heatmaps)]
ep_terminal_points = [[] for _ in range(nb_heatmaps)]
for ep_nb in range(nb_eval_eps):
    for i, ax in enumerate(axs):
        if ep_nb == 0:
            label_line = 'Episode traces'
            label_point_start = 'Start state'
            label_point_finish = 'Terminal state'
        else:
            label_line = None
            label_point_start = None
            label_point_finish = None

        ep_lines[i].append(ax.plot([], [], marker='x', c='m', label=label_line)[0])
        ep_starting_points[i].append(ax.plot([], [], c='k', marker='o', markersize=2, label=label_point_start)[0])
        ep_terminal_points[i].append(ax.plot([], [], c='k', marker='s', markersize=4, label=label_point_finish)[0])
axs[0].legend(loc='best')

# Tracking errors
fig2, ax3 = plt.subplots()
errors_line1, = ax3.plot([], [])
errors_line2, = ax3.plot([], [])
errors1 = []
errors2 = []

is_eval = lambda ep_idx: (ep_idx + 1) % eval_every == 0 or ep_idx == 0 or ep_idx == nb_eps - 1

# Start training
for ep in trange(nb_eps):
    o = env.reset()
    d = False
    while not d:
        a = get_total_greedy_action(o)[0]
        otp1, r, d, _ = env.step(a)

        qvfa, qvfb = swap_q()
        target = r + gamma * qvfb.value(otp1, qvfa.greedy_action(otp1))
        error = qvfa.update(o, a, target)

        if is_eval(ep):
            if qvfa == qvf1:
                errors1.append(error)
            else:
                errors2.append(error)

        o = otp1

    # Evaluation phase
    if is_eval(ep):
        print(f'Episode {ep}')
        fig.suptitle(f'Episode {ep:4d}\nIHT count {tilings.count():6d}')
        states = [play_ep_get_obs() for _ in range(nb_eval_eps)]
        for i in range(nb_heatmaps):
            # a_ = actions[i]
            for j, ts in enumerate(tracking_states):
                v = max([qvf1.value(ts, a_) + qvf2.value(ts, a_) for a_ in actions])
                tracked_vals[i][j] = v
            title = f'Action={actions[i]}'
            update_heatmap(im_list[i], reshape_to_map(tracked_vals[i]), title)

            for j, state in enumerate(states):
                ep_lines[i][j].set_data(state[0], state[1])
                ep_starting_points[i][j].set_data(state[0][0], state[1][0])
                ep_terminal_points[i][j].set_data(state[0][-1], state[1][-1])

        errors_line1.set_data(range(len(errors1)), errors1)
        errors_line2.set_data(range(len(errors2)), errors2)
        ax3.set_xlim((0, max([len(errors1), len(errors2)])))
        ax3.set_ylim((min(errors1 + errors2), max(errors1 + errors2)))
        plt.pause(1)

plt.ioff()

plt.show()
