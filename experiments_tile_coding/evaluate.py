import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl
from tqdm import tqdm as pbar
from tile_coding_re.tiles3_qfunction import QValueFunctionTiles3
from random_env.envs.random_env_discrete_actions import RandomEnvDiscreteActions as REDA, get_discrete_actions

def get_latest_experiment():
    lab_dir = '/home/leander/code/RandomEnvRLTests/experiments_tile_coding/'
    experiments = []
    for fn in os.listdir(lab_dir):
        if 'sarsa' in fn:
            experiments.append(fn)
    experiments = sorted(experiments)

    experiment_name = experiments[-1]

    return os.path.join(lab_dir, experiment_name)

get_q_func_step = lambda item: int(os.path.split(item)[-1].split('_')[2].split('.')[0])

def get_q_func_filenames(experiment_dir):
    q_func_dir = os.path.join(experiment_dir, 'q_func')
    q_func_filenames = [fn for fn in os.listdir(q_func_dir)]

    q_func_filenames = sorted(q_func_filenames, key=get_q_func_step)
    q_func_filenames = [os.path.join(q_func_dir, item) for item in q_func_filenames]

    return q_func_filenames

def get_q_func_xrange(q_func_filenames):
    return np.linspace(get_q_func_step(q_func_filenames[0]), get_q_func_step(q_func_filenames[-1]), len(q_func_filenames))

def get_val(qvf, state, nb_actions):
    return max([qvf.value(state, a_) for a_ in range(nb_actions)])

def plot_q_vals_grid_tracking_states():
    experiment_dir = get_latest_experiment()
    experiment_name = os.path.split(experiment_dir)

    env = REDA.load_from_dir(experiment_dir)
    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    q_func_filenames = get_q_func_filenames(experiment_dir)
    nb_q_funcs = len(q_func_filenames)

    # Initialise grid tracking states
    n_tracking_dim = 5
    tracking_ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
    tracking_states = np.array([list(item) for item in product(*np.array([np.linspace(l, h, n_tracking_dim) for l, h in tracking_ranges]))])
    nb_tracked = len(tracking_states)

    vals = np.zeros(shape=(nb_tracked, nb_q_funcs))
    for j, qfn in enumerate(q_func_filenames):
        q = QValueFunctionTiles3.load(qfn)
        for i, ts in enumerate(tracking_states):
            vals[i,j] = get_val(q, ts, nb_actions)

    xrange = get_q_func_xrange(q_func_filenames)
    fig, axs = plt.subplots(2, gridspec_kw=dict(height_ratios=[1,2]))
    cmap = mpl.cm.get_cmap('tab10')
    cmap_x = np.linspace(0, 1, nb_tracked)

    ax = axs[0]
    ax.set_xlabel('State dimension 0')
    ax.set_ylabel('State dimension 1')
    for i, ts in enumerate(tracking_states):
        ax.scatter(ts[0], ts[1], marker='x', c=cmap(cmap_x[i]))

    ax = axs[1]
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Estimated values')
    for i, val in enumerate(vals):
        ax.plot(xrange, val, c=cmap(cmap_x[i]), lw=1.0)

    fig.suptitle(experiment_name)
    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'tracked_vals.png'))
    plt.show()


def circular_normal_sample(r, s):
    if r > 0:
        R = np.random.normal(r, s)
        theta = 2 * np.pi * np.random.rand()
        return [R * np.cos(theta), R * np.sin(theta)]
    else:
        return np.random.normal(r, s, 2)

def circular_uniform_sample(rlow, rhigh):
    R = np.random.uniform(rlow, rhigh)
    theta = 2 * np.pi * np.random.rand()
    return [R * np.cos(theta), R * np.sin(theta)]

def plot_q_vals_region_sampling_tracking_states():
    experiment_dir = get_latest_experiment()
    experiment_name = os.path.split(experiment_dir)

    env = REDA.load_from_dir(experiment_dir)
    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    print('Searching for training data')
    q_func_filenames = get_q_func_filenames(experiment_dir)
    # qs = [QValueFunctionTiles3.load(item) for item in q_func_filenames]
    nb_q_funcs = len(q_func_filenames)

    print('Creating tracking regions')
    nb_samples_per_region = 200
    # s = 0.02 # normal
    rstep = 0.2 # uniform
    tracking_states = []
    # for r in np.arange(0, 1, 0.2): # normal
    for r in np.arange(5):
        for i in range(nb_samples_per_region):
            # ts = circular_normal_sample(r, s) # normal
            ts = circular_uniform_sample(rlow=r * rstep, rhigh=r * rstep + rstep)
            tracking_states.append(ts)
    nb_tracked = len(tracking_states)

    print('Estimating values at tracked states')
    tracked_vals = [[] for _ in range(nb_tracked)]
    for qfn in pbar(q_func_filenames):
        q = QValueFunctionTiles3.load(qfn)
        for i, ts in enumerate(tracking_states):
            tracked_vals[i].append(get_val(q, ts, nb_actions))

    print('Plotting')
    fig, axs = plt.subplots(2, figsize=(15,10))
    cmap = mpl.cm.get_cmap('jet')
    region_color = lambda i: cmap((i // nb_samples_per_region) * nb_samples_per_region/(nb_tracked - nb_samples_per_region))
    ax = axs[0]
    ax.set_xlabel('State dimension 0')
    ax.set_ylabel('State dimension 1')
    for i, ts in enumerate(tracking_states):
        ax.scatter(ts[0], ts[1], c=region_color(i), marker='x')

    ax = axs[1]
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Estimated values')
    xrange = get_q_func_xrange(q_func_filenames)
    for val_idx in np.arange(0, nb_tracked, nb_samples_per_region):
        vals = tracked_vals[val_idx:val_idx+nb_samples_per_region]
        vals_med = np.median(vals, axis=0)
        vals_min = np.min(vals, axis=0)
        vals_max = np.max(vals, axis=0)
        label = 'Median' if val_idx == 0 else None
        ax.plot(xrange, vals_med, c=region_color(val_idx), lw=1.0, label=label)
        label = 'Min-Max' if val_idx == 0 else None
        ax.fill_between(xrange, vals_min, vals_max, color=region_color(val_idx), alpha=0.2, label=label)
    ax.legend(loc='best')
    plt.savefig(os.path.join(experiment_dir, 'tracked_vals_per_region.png'))
    plt.show()

from tile_coding_re.heatmap_utils import make_heatmap
from experiments_tile_coding.eval_utils import play_episode
def plot_individual_action_qvals():
    experiment_dir = get_latest_experiment()
    experiment_name = os.path.split(experiment_dir)

    env = REDA.load_from_dir(experiment_dir)
    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    print('Searching for training data')
    q_func_filenames = get_q_func_filenames(experiment_dir)
    training_steps = [get_q_func_step(qfn) for qfn in q_func_filenames]
    # qs = [QValueFunctionTiles3.load(item) for item in q_func_filenames]
    nb_q_funcs = len(q_func_filenames)

    query_training_step = 80000-1
    qfn_at_step = q_func_filenames[np.searchsorted(training_steps, query_training_step)]
    at_training_step = get_q_func_step(qfn_at_step)

    # Initialise grid tracking states
    n_tracking_dim = 64
    tracking_lim = 1.2
    tracking_ranges = [[-tracking_lim, -tracking_lim], [tracking_lim, tracking_lim]]
    tracking_ranges = [[l, h] for l, h in zip(*tracking_ranges)]
    tracking_states = np.array(
        [list(item) for item in product(*np.array([np.linspace(l, h, n_tracking_dim) for l, h in tracking_ranges]))])
    nb_tracked = len(tracking_states)

    q = QValueFunctionTiles3.load(qfn_at_step)

    nb_test_eps = 1000
    test_obses = []
    test_acts = []
    for _ in range(nb_test_eps):
        obses, acts, _ = play_episode(env, q, circular_uniform_sample(0.8, 0.9))
        if len(obses)>50:
            test_obses.append(obses)
            test_acts.append(acts)
            if len(test_obses)>10:
                break

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = np.ravel(axs)
    for action_idx, (ax, act) in enumerate(zip(axs, actions)):
        init_state = np.array([0.0, 0.0])
        env.reset(init_state)
        otp1, *_ = env.step(act)
        ax.plot(*np.vstack([init_state, otp1]).T, c='k', marker='x', zorder=20)

        for tobses, tacts in zip(test_obses, test_acts):
            tobses = np.array(tobses).T
            ax.plot(tobses[0], tobses[1], c='k', zorder=25)


        tracked_qvals = []
        for ts in tracking_states:
            # ax.scatter(ts[0], ts[1], marker='o', c='k')
            qval = q.value(ts, action_idx)
            tracked_qvals.append(qval)
        tracked_qvals = np.array(tracked_qvals).reshape((n_tracking_dim, n_tracking_dim))
        tracked_qvals = np.flipud(np.rot90(tracked_qvals))
        make_heatmap(ax, tracked_qvals, tracking_states.T[0], tracking_states.T[1], title=f'{act}')


    fig.suptitle(f'{repr(env)}\n'
                 f'Training step = {at_training_step}')
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.88, wspace=0.083, hspace=0.321)
    plt.savefig(os.path.join(experiment_dir, f'individual_action_qvals_at-step-{at_training_step}.png'))
    plt.show()

if __name__ == '__main__':
    # plot_q_vals_grid_tracking_states()
    plot_q_vals_region_sampling_tracking_states()
    # plot_individual_action_qvals()
