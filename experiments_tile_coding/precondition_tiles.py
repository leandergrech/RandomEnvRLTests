from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from tile_coding_re.heatmap_utils import make_heatmap, update_heatmap
from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions

# Environment
n_obs, n_act = 2,2
env = REDA(n_obs, n_act, estimate_scaling=False)
actions = get_discrete_actions(n_act, 3)
n_actions = len(actions)

# Tilings
nb_tilings = 8
nb_bins = 2
feature_ranges = [[l,h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
max_tiles = 2 ** 20
tilings = Tilings(nb_tilings, nb_bins, feature_ranges, max_tiles)

# Value function
qvf = QValueFunctionTiles3(tilings, n_actions)
def get_qvals(state):
    return [qvf.value(state, a_) for a_ in range(n_actions)]

# Analystical reward function
reward_function = env.objective

# States for heatmap
dim_size = int(nb_bins * nb_tilings)
tracking_states = np.array([list(item) for item in product(*np.array([np.linspace(l, h, dim_size) for l, h in feature_ranges]))])
nb_tracked = tracking_states.shape[0]
reshape_to_map = lambda arr: arr.reshape(dim_size, dim_size).T

# Tracked values
tracking_vals = np.zeros(nb_tracked)
def get_val_from_qvals(state):
    return max(get_qvals(state))


# Precondition q-table
for tracked_state in tracking_states:
    opt_action = env.get_optimal_action(tracked_state)
    opt_action_idx = actions.index(opt_action)
    r = reward_function(tracked_state)

    action_value_vector = np.ones(n_actions) * -10.0
    action_value_vector[opt_action_idx] = r

    for a_idx, val in enumerate(action_value_vector):
        qvf.set(tracked_state, a_idx, val)

tracking_vals_after = []
for ts in tracking_states:
    tracking_vals_after.append(get_val_from_qvals(ts))
tracking_vals_after = np.array(tracking_vals_after)

# Initialise plots
fig, axs = plt.subplots(2)
ax1 = axs[0]
ax2 = axs[1]
im1 = make_heatmap(ax1, reshape_to_map(tracking_vals), *feature_ranges, '\n')
im2 = make_heatmap(ax2, reshape_to_map(tracking_vals_after), * feature_ranges, '\n')

plt.show()

