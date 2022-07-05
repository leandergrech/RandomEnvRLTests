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

n_obs, n_act = 2, 2
env = REDA(n_obs, n_act)

nb_tilings = 32
nb_bins = 2
ranges = [[l, h] for l, h in zip(env.observation_space.low,
                                 env.observation_space.high)]
ranges.append([0.0, 0.15]) # for velocity dimension - Approximate values from [testing_randomenvdiscreteaction.py:testing_reda_velocities()]

max_tiles = 2 ** 20

tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
actions = get_discrete_actions(n_act)
lr = lr(1e-2, 10000)
gamma = 0.99

qvf1 = QValueFunctionTiles3(tilings, actions, lr)
qvf2 = QValueFunctionTiles3(tilings, actions, lr)

def get_vel(state, next_state):
    return np.sqrt(np.sum(np.square(np.subtract(next_state, state))))

def swap_q():
    if np.random.rand() < 0.5:
        return qvf1, qvf2
    else:
        return qvf2, qvf1


def get_total_greedy_action(state, prev_state=None):
    global actions
    if prev_state is None:
        vel = 0.0
    else:
        vel = get_vel(prev_state, state)
    
    state_with_vel = np.concatenate([state, [vel]])
        
    val1 = [qvf1.value(state_with_vel, a_) for a_ in actions]
    val2 = [qvf2.value(state_with_vel, a_) for a_ in actions]
    tot_val = [v1 + v2 for v1, v2 in zip(val1, val2)]
    action_idx = np.argmax(tot_val)

    return actions[action_idx], val1, val2


def play_ep_get_obs():
    obs = []
    o = env.reset()
    otm1 = None
    d = False
    while not d:
        a = get_total_greedy_action(o, otm1)[0]        
        otm1 = o
        otp1, r, d, _ = env.step(a)
        
        o = otp1.copy()
        
        obs.append(o.copy())
    obs = np.array(obs).T
    return obs[0], obs[1]


# Set up grid of states for heatmaps
dim_size = 30

# tracking_states = np.array([list(item) for item in product(*np.array([np.linspace(l, h, dim_size) for l, h in ranges]))])
tracking_abslim = 2
tracking_states = np.array([list(item) for item in product(*np.repeat(
    np.linspace(-tracking_abslim, tracking_abslim, dim_size).reshape(1, -1), n_obs, axis=0))])
nb_tracked = tracking_states.shape[0]

reshape_to_map = lambda arr: arr.reshape(dim_size, dim_size)

# Training and evaluation info
nb_eps = 100
eval_step = 10000

# Arrays used to store heatmap info

# ## for tracking individual actions
# nb_heatmaps = len(actions)
# tracked_vals = np.zeros((nb_heatmaps, nb_tracked))
# plt.ion()
# figsize = (15, 15)
# axs_square_side = int(np.ceil(np.sqrt(nb_heatmaps)))
# fig, axs = plt.subplots(axs_square_side, axs_square_side, figsize=figsize)
# fig.suptitle('\n\n')
# axs = axs.ravel()[:nb_heatmaps]


## for tracking greedy value
nb_heatmaps = 1
tracked_vals = np.zeros((nb_heatmaps, nb_tracked))
plt.ion()
figsize = (15, 15)
fig, axs = plt.subplots()
fig.suptitle('\n\n')
axs = [axs]



# Initialise heatmaps
im_list = []
for i, ax in enumerate(axs):
    im = make_heatmap(ax, reshape_to_map(tracked_vals[i]), (-tracking_abslim, tracking_abslim),
                      (-tracking_abslim, tracking_abslim), '\n')
    # im = make_heatmap(ax, reshape_to_map(tracked_vals[i]), *ranges, '\n')
    im_list.append(im)
fig.tight_layout()

# Initialise sample episode traces
nb_eval_eps = 5
ep_lines = [[] for _ in range(nb_heatmaps)]
ep_starting_points = [[] for _ in range(nb_heatmaps)]
for _ in range(nb_eval_eps):
    for i, ax in enumerate(axs):
        ep_lines[i].append(ax.plot([], [], marker='x', c='m')[0])
        ep_starting_points[i].append(ax.plot([], [], c='k', marker='o', markersize=2)[0])

# Tracking errors
fig2, ax3 = plt.subplots()
errors_line1, = ax3.plot([], [])
errors_line2, = ax3.plot([], [])
errors1 = []
errors2 = []


def update_error_plot():
    errors_line1.set_data(range(len(errors1)), errors1)
    errors_line2.set_data(range(len(errors2)), errors2)
    ax3.set_xlim((0, max([len(errors1), len(errors2)])))
    ax3.set_ylim((min(errors1 + errors2), max(errors1 + errors2)))


is_eval = lambda ep_idx: (ep_idx + 1) % eval_step == 0# or ep_idx == 0 or ep_idx == nb_eps - 1
ep_lens = []
# Start training
for ep in trange(nb_eps):
    o = env.reset()
    otp1 = o.copy()
    vel = [0.0]
    o_with_vel = np.concatenate([o, vel])
    d = False
    ep_len = 0
    while not d:
        a = get_total_greedy_action(otp1, o)[0]
        
        otp1, r, d, _ = env.step(a)
        
        veltp1 = [get_vel(o, otp1)]
        
        otp1_with_vel = np.concatenate([otp1, veltp1])
            
        qvfa, qvfb = swap_q()
        # target = r + gamma * qvfb.value(otp1, qvfa.greedy_action(otp1))
        # error = qvfa.update(o, a, target)
        target = r + gamma * qvfb.value(otp1_with_vel, qvfa.greedy_action(otp1_with_vel))
        error = qvfa.update(o_with_vel, a, target)

        if is_eval(ep):
            if qvfa == qvf1:
                errors1.append(error)
            else:
                errors2.append(error)

        o = otp1.copy()
        o_with_vel = otp1_with_vel.copy()
        ep_len += 1
    ep_lens.append(ep_len)

    # Evaluation phase
    if is_eval(ep):
        states = [play_ep_get_obs() for _ in range(nb_eval_eps)]
        fig.suptitle(f'Episode {ep:4d}\nIHT count {tilings.count():6d}\nAverage ep_len {np.mean(ep_lens):3.2f}')
        ep_lens = []
        
        
        for i in range(nb_heatmaps):
            a_ = actions[i]
            for j, ts in enumerate(tracking_states):
                ts_vel0 = np.concatenate([ts, [0.0]])
                v = qvf1.value(ts_vel0, a_) + qvf2.value(ts_vel0, a_)
                tracked_vals[i][j] = v
            title = f'Action={actions[i]} @ vel=0.0'
            update_heatmap(im_list[i], reshape_to_map(tracked_vals[i]), title)

            for j, state in enumerate(states):
                ep_lines[i][j].set_data(state[0], state[1])
                ep_starting_points[i][j].set_data(state[0][0], state[1][0])

        update_error_plot()
        plt.pause(0.1)

plt.ioff()
fig, ax = plt.subplots()
ax.plot(ep_lens)

plt.show()
