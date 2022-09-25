import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime as dt
import yaml
from tqdm import trange

from random_env.envs.random_env_discrete_actions import REDAClip, get_discrete_actions
from utils.eval_utils import play_episode, eval_agent
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from utils.heatmap_utils import make_heatmap


class OptimalAgentWrapper:
    def __init__(self, env: REDAClip):
        self.env = env
        self.actions = get_discrete_actions(env.act_dimension, 3)
        self.nb_actions = len(self.actions)

    def greedy_action(self, state):
        action = self.env.get_optimal_action(state)
        if np.random.rand() < 0.1:
            return np.random.choice(self.nb_actions)
        else:
            return self.actions.index(action)


exp_dir = dt.now().strftime(f'optimal_%m%d%y_%H%M%S')
os.makedirs(exp_dir)

# Hyperparameters
params = dict(
    n_obs=2, n_act=2,
    nb_tilings=16, nb_bins=2,
    nb_eps=500, lr=0.01, gamma=0.9,
    eval_every=100, eval_eps=20
)
with open(os.path.join(exp_dir, "train_params.yml"), "w") as f:
    f.write(yaml.dump(params))

# Create environment and action space
n_obs = params['n_obs']
n_act = params['n_act']
env = REDAClip(n_obs, n_act, state_clip=1.0)
eval_env = REDAClip(n_obs, n_act, state_clip=1.0, model_info=env.model_info)

env_lims = [[l,h] for l,h in zip(env.observation_space.low, env.observation_space.high)]
actions = get_discrete_actions(n_act, 3)
nb_actions = len(actions)

# Create optimal agent for this environment
optimal_agent = OptimalAgentWrapper(env)

# Create function approximation for tabular q function
nb_tilings, nb_bins = params['nb_tilings'], params['nb_bins']
tilings = Tilings(nb_tilings=nb_tilings, nb_bins=nb_bins, feature_ranges=env_lims, max_tiles=2**15)
q = QValueFunctionTiles3(tilings=tilings, n_discrete_actions=nb_actions)

# Training parameters
nb_eps = params['nb_eps']
lr = params['lr']
gamma = params['gamma']
eval_every = params['eval_every']

gammas = np.power(gamma, np.arange(env.EPISODE_LENGTH_LIMIT))
T = 0
all_ep_lens = []
all_returns = []
iht_counts = []
for ep in trange(nb_eps):
    obses, acts, rews = play_episode(env, optimal_agent, init_state=None)
    returns = []

    ep_len = len(rews)
    # Monte Carlo update with expert trajectories
    for i, (obs, act, rew) in enumerate(zip(obses[:-1], acts, rews)):
        g = np.sum(np.multiply(gammas[:ep_len - i], rews[i:]))
        q.update(obs, act, g, lr)
        T += 1
        if (T + 1) % eval_every == 0:
            _, returns, ep_lens = eval_agent(eval_env, q, params.get('eval_eps'))
            all_ep_lens.append(ep_lens)
            all_returns.append(returns)
            iht_counts.append(tilings.count())

# Initialise grid tracking states
n_tracking_dim = 64
tracking_lim = 1.2
tracking_ranges = [[-tracking_lim, -tracking_lim], [tracking_lim, tracking_lim]]
tracking_ranges = [[l, h] for l, h in zip(*tracking_ranges)]
tracking_states = np.array(
    [list(item) for item in product(*np.array([np.linspace(l, h, n_tracking_dim) for l, h in tracking_ranges]))])

# Estimate and store state values
tracking_estimated_vals = []
for ts in tracking_states:
    tracking_estimated_vals.append(np.mean([q.value(ts, a_) for a_ in range(nb_actions)]))

# Plot the fuckin shit already
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs = np.ravel(axs)
min_adv, max_adv = np.inf, -np.inf
ims = []
for action_idx, (ax, act) in enumerate(zip(axs, actions)):
    # Plot unit trim for action_idx'th action
    init_state = env.reset(np.zeros(n_obs)).copy()
    otp1, *_ = env.step(act)
    ax.plot(*np.vstack([init_state, otp1]).T, c='k', marker='x', zorder=20)

    # Calculate advantages
    tracked_advs = []
    for ts, tv in zip(tracking_states, tracking_estimated_vals):
        # ax.scatter(ts[0], ts[1], marker='o', c='k')
        qval = q.value(ts, action_idx)
        adv = qval - tv
        tracked_advs.append(adv)

    # For clim to have the same range
    min_adv = np.min([min_adv, *tracked_advs])
    max_adv = np.max([min_adv, *tracked_advs])

    # Magic fuckery to align array to correct heatmap orientation
    tracked_advs = np.array(tracked_advs).reshape((n_tracking_dim, n_tracking_dim))
    tracked_advs = np.flipud(np.rot90(tracked_advs))

    # Heatmap plotting
    im = make_heatmap(ax, tracked_advs, tracking_states.T[0], tracking_states.T[1], title=f'{act}')
    ims.append(im)
    ax.add_patch(mpl.patches.Circle((0, 0), env.GOAL, edgecolor='g', ls='--', facecolor='None', zorder=20))

for im in ims:
    im.set_clim((min_adv, max_adv))

fig.suptitle(f'Dir = {exp_dir}\nEnv = {repr(env)}\n'
             f'Advantages of optimal actions')
# fig.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.88, wspace=0.083, hspace=0.321)
save_path = os.path.join(exp_dir, f'individual_action_advs.png')
plt.savefig(save_path)
print(f'Saved figure to: {save_path}')

