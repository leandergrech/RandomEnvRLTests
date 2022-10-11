import os
from itertools import product
from collections import defaultdict
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl
from tqdm import tqdm as pbar
import re

from utils.plotting_utils import make_heatmap, grid_on
from linear_q_function import QValueFunctionLinear
from random_env.envs import REDAClip, IREDA, get_discrete_actions, RandomEnvDiscreteActions as REDA
from utils.eval_utils import get_q_func_filenames, get_q_func_xrange, get_val, get_q_func_step, get_latest_experiment
from utils.training_utils import nball_uniform_sample


def get_eval_every_hack(yaml_path):
    with open(yaml_path, 'r') as f:
        for line in f:
            if 'eval_every' in line:
                return int(re.findall(r'\d+', line)[0])


SUB_EXP_COLORS = ['r', 'g', 'b', 'k'] # I never exceed 4 sub-experiments


def plot_experiment_training_stats(exp_dir, exp_label):
    """
    Access training_stats.pkl for a single experiment and plot training stats.
    Top plot:    IHT count evolution
    Middle plot: Episode length mean+std obtained using latest q-table greedily
    Bottom plot: Return mean obtained using latest q-table greedily
    """
    stats_file = os.path.join(exp_dir, 'training_stats.pkl')

    eval_every = get_eval_every_hack(os.path.join(exp_dir, 'train_params.yml'))

    with open(stats_file, 'rb') as f:
        data = pkl.load(f)

    ep_lens = data['ep_lens']
    returns = data['returns']
    regrets = data['regrets']

    xrange = np.arange(len(ep_lens)) * eval_every

    fig, axs = plt.subplots(3, gridspec_kw=dict(height_ratios=[3, 3, 3]), figsize=(15, 10))

    ax = axs[0]
    ep_lens_mean = np.mean(ep_lens, axis=1)
    ep_lens_std = np.std(ep_lens, axis=1)
    ax.plot(xrange, ep_lens_mean, ls='dashed', label=f'{exp_label} Mean')
    ax.set_title('Using greedy policy')
    ax.set_ylabel('Episode length')
    grid_on(ax, 'y', major_loc=20, minor_loc=5, major_grid=True, minor_grid=False)

    ax = axs[1]
    returns_mean = np.mean(returns, axis=1)
    returns_std = np.std(returns, axis=1)
    ax.plot(xrange, returns_mean, ls='dashed', label=f'{exp_label} Mean')
    ax.set_ylabel('Returns')

    ax = axs[2]
    regrets_mean = np.mean(regrets, axis=1)
    regrets_std = np.std(regrets, axis=1)
    ax.plot(xrange, regrets_mean, ls='dashed', label=f'{exp_label} Mean')
    ax.set_ylabel('Returns')

    for ax in axs:
        ax.legend(loc='best', prop=dict(size=8))
        grid_on(ax, 'x')
        ax.set_xlabel('Training steps')

    fig.tight_layout()
    plt.savefig(os.path.join(exp_pardir, 'training_stats.png'))


def plot_all_experiments_training_stats(exp_pardir, exp_subdirs, exp_labels, exp_filter=''):
    """
        Access training_stats.pkl for all experiments found in exp_pardir and
        plot their combined training stats.
        Top plot:    IHT count evolution
        Middle plot: Episode length mean+std obtained using latest q-table greedily
        Bottom plot: Return mean obtained using latest q-table greedily
    """
    regrets = defaultdict(list)
    returns = defaultdict(list)
    ep_lens = defaultdict(list)
    xrange = None
    eval_every = None

    all_exp = []

    # Iterate over experiment with different environment
    for exp_name in sorted(os.listdir(exp_pardir)):
        if 'sarsa' not in exp_name or exp_filter not in exp_name or '.py' in exp_name:
            continue

        if eval_every is None:
            eval_every = get_eval_every_hack(os.path.join(exp_pardir, exp_name, exp_subdirs[0], 'train_params.yml'))

        all_exp.append(exp_name)
        experiment_dir = os.path.join(exp_pardir, exp_name)

        # Iterate over different state initialisation schemes
        for sub_exp in exp_subdirs:
            pkl_file = os.path.join(experiment_dir, sub_exp, 'training_stats.pkl')
            with open(pkl_file, 'rb') as f:
                data = pkl.load(f)

                regrets[sub_exp].append(data['regrets'])
                returns[sub_exp].append(data['returns'])
                ep_lens[sub_exp].append(data['ep_lens'])

                if xrange is None:
                    xrange = np.arange(len(data['regrets'])) * eval_every

    print(f'Found {len(all_exp)} experiments')

    fig, axs = plt.subplots(3, gridspec_kw=dict(height_ratios=[3, 3, 3]), figsize=(15, 10))
    for label, sub_exp, c in zip(exp_labels, exp_subdirs, SUB_EXP_COLORS):
        for dataset, ax, ylabel in zip((ep_lens, returns, regrets), axs, ('Episode length', 'Returns', 'Regrets')):
            data = dataset[sub_exp]
            data_mean = np.mean(np.mean(data, axis=-1), axis=0)
            # data_std = np.sqrt(np.mean(np.square(np.std(data, axis=-1)), axis=0))
            data_min = np.min(np.min(data, axis=-1), axis=0)
            data_max = np.max(np.max(data, axis=-1), axis=0)

            ax.plot(xrange, data_mean, ls='solid', c=c, label=f'{label} ' + r'$\mu$', lw=2, zorder=10)
            # for i, d in enumerate(data):
            #     ax.plot(xrange, np.mean(d, axis=-1), lw=0.5, zorder=15, label=all_exp[i])
            # ax.fill_between(xrange, data_mean - data_std, data_mean + data_std, facecolor='None', edgecolor=c, hatch='//', alpha=0.6)
            ax.fill_between(xrange, data_min, data_max, facecolor='None', edgecolor=c, hatch='//', alpha=0.6)
            # ax.set_title('Using greedy policy')
            ax.set_ylabel(ylabel)

    for ax in axs:
        ax.legend(loc='best', prop=dict(size=10))
        y_grid_on(ax)
        ax.set_xlabel('Training steps')
    fig.suptitle(f'{len(all_exp)} different environments')
    fig.tight_layout()
    plt.savefig(os.path.join(exp_pardir, 'results.png'))
    plt.show()


def plot_episodes(exp_dir: str, train_step: int, env_type: REDA, nrows=2, ncols=4, save_dir=None):
    if save_dir is None:
        save_dir = exp_dir

    env = env_type.load_from_dir(exp_dir)
    q_func_file = os.path.join(exp_dir, 'q_func', f'q_step_{train_step}.pkl')
    q = QValueFunctionLinear.load(q_func_file)
    n_obs, n_act = env.obs_dimension, env.act_dimension
    init_func = env.reset

    fig, axs = plt.subplots(nrows * 2, ncols, figsize=(30, 15))
    # axs = np.ravel(axs)
    nb_eps = nrows * ncols

    for i in range(nb_eps):
        obses = []
        acts = []
        d = False
        o = env.reset(init_func())
        obses.append(o.copy())
        acts.append(np.zeros(n_act))
        step = 1
        while not d:
            a = q.actions[q.greedy_action(o)]
            otp1, _, d, _ = env.step(a)
            o = otp1.copy()

            obses.append(o)
            acts.append(a)

            step += 1
        obses = np.array(obses)
        acts = np.array(acts) - 1

        ax_obs = axs[(i // ncols) * 2, i % ncols]
        ax_act = axs[(i // ncols) * 2 + 1, i % ncols]

        ax_obs.set_title(f'Ep {i + 1}', size=15)
        ax_obs.axhline(-env.GOAL, c='g', ls='--', lw=2)
        ax_obs.axhline(env.GOAL, c='g', ls='--', lw=2)
        ax_obs.plot(obses, c='b')
        grid_on(ax_obs, 'y', 0.1, 0.02, True, False)

        ax_act.axhline(0.0, c='k', ls='-.', lw=2)
        ax_act.plot(acts, c='r')

        for ax, ylab in zip((ax_obs, ax_act), ('States', 'Actions')):
            grid_on(ax, 'x', 5, 1, True, False)
            ax.set_xticks(np.arange(step))
            ax.set_ylabel(ylab, size=12)

        ax_obs.get_shared_x_axes().join(ax_obs, ax_act)
        ax_act.set_xlabel('Step', size=12)

    fig.suptitle(f'Environment: {repr(env)}\n'
                 f'Experiment:  {exp_dir}\n'
                 f'At step:     {train_step}')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{train_step}_step.png'))
    # plt.show()


def plot_weight_evolution(exp_dir, save_dir=None):
    if save_dir is None:
        save_dir = exp_dir

    q_func_fns = get_q_func_filenames(exp_dir)
    x = get_q_func_xrange(q_func_fns)

    weights = None
    actions = None
    for qfn in q_func_fns:
        q = QValueFunctionLinear.load(qfn)
        if actions is None:
            actions = q.actions
        if weights is None:
            weights = np.expand_dims(q.w, axis=-1)
        else:
            weights2 = np.expand_dims(q.w, axis=-1)
            weights = np.concatenate([weights, weights2], axis=-1)

    fig, axs = plt.subplots(3, 3, figsize=(20, 12))
    axs = np.ravel(axs)
    for ax, per_action_weights in zip(axs, weights):
        for i, per_action_w in enumerate(per_action_weights):
            ax.plot(x, per_action_w, label=f'w_{i}')
    for ax, a in zip(axs, actions):
        grid_on(ax, 'x', 100, 20)
        ax.set_title(np.subtract(a, 1), size=18)
        ax.set_xlabel('Training step', size=15)
        ax.set_ylabel('Weight', size=15)
        ax.legend(loc='best')

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tracked_weights.png'))
    plt.show()
    plt.show()


def create_grid_tracking_states(env, n_dim):
    """
    Utility function to create grid of states for passed env. Each env state
    dimension is split into n_dim parts. Returns 1D list with n_dim**n_obs
    total states.
    """
    tracking_ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
    tracking_states = np.array(
        [list(item) for item in product(*np.array([np.linspace(l, h, n_dim) for l, h in tracking_ranges]))])

    return tracking_states


def plot_q_vals_grid_tracking_states(experiment_dir, n_tracking_dim, env_type, save_dir=None):
    """
    Create a grid of states within the environment state limits, access the q-table for every eval
    step during training, and plot the evolution of the tracked q-values during training.
    """
    if save_dir is None:
        save_dir = experiment_dir

    experiment_name = os.path.split(experiment_dir)

    env = env_type.load_from_dir(experiment_dir)
    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    q_func_filenames = get_q_func_filenames(experiment_dir)
    nb_q_funcs = len(q_func_filenames)
    xrange = get_q_func_xrange(q_func_filenames)

    # Initialise grid tracking states
    tracking_states = create_grid_tracking_states(env, n_tracking_dim)
    nb_tracked = len(tracking_states)

    vals = np.zeros(shape=(nb_tracked, nb_q_funcs))
    for j, qfn in enumerate(q_func_filenames):
        q = QValueFunctionLinear.load(qfn)
        for i, ts in enumerate(tracking_states):
            vals[i, j] = get_val(q, ts, nb_actions)

    # Initialise figure
    fig, axs = plt.subplots(2, gridspec_kw=dict(height_ratios=[1, 2]))
    cmap = mpl.cm.get_cmap('tab10')
    cmap_x = np.linspace(0, 1, nb_tracked)

    # Plot tracked states
    ax = axs[0]
    ax.set_xlabel('State dimension 0')
    ax.set_ylabel('State dimension 1')
    for i, ts in enumerate(tracking_states):
        ax.scatter(ts[0], ts[1], marker='x', c=cmap(cmap_x[i]))

    # Plot evolution of q_values for tracked states
    ax = axs[1]
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Estimated values')
    for i, val in enumerate(vals):
        ax.plot(xrange, val, c=cmap(cmap_x[i]), lw=1.0)

    fig.suptitle(experiment_name)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tracked_vals.png'))
    plt.show()


def plot_q_vals_region_sampling_tracking_states(experiment_dir, env_type, save_dir=None):
    if save_dir is None:
        save_dir = experiment_dir

    env = env_type.load_from_dir(experiment_dir)
    n_obs = env.obs_dimension
    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    # Searching for training data
    q_func_filenames = get_q_func_filenames(experiment_dir)
    xrange = get_q_func_xrange(q_func_filenames)

    # Creating tracking regions
    nb_samples_per_region = 200
    nb_regions = 5
    rstep = 1 / nb_regions

    tracking_states = []
    for r in np.arange(nb_regions):
        for i in range(nb_samples_per_region):
            ts = nball_uniform_sample(n_obs, rlow=r * rstep, rhigh=r * rstep + rstep)
            tracking_states.append(ts)
    nb_tracked = len(tracking_states)

    # Estimating values at tracked states
    tracked_vals = [[] for _ in range(nb_tracked)]  # shape=(nb_tracked, len(q_func_filenames))
    for qfn in pbar(q_func_filenames):
        q = QValueFunctionLinear.load(qfn)
        for i, ts in enumerate(tracking_states):
            tracked_vals[i].append(get_val(q, ts, nb_actions))

    # Plotting
    fig, axs = plt.subplots(2, figsize=(15, 10))
    fig.suptitle(experiment_dir)
    cmap = mpl.cm.get_cmap('jet')

    # Gives color to each region. Regions are stored contiguously
    def region_color(idx):
        return cmap((idx // nb_samples_per_region)
                    * nb_samples_per_region / (nb_tracked - nb_samples_per_region))

    # Plot tracked states, color-coded per region
    ax = axs[0]
    ax.set_xlabel('State dimension 0')
    ax.set_ylabel('State dimension 1')
    for i, ts in enumerate(tracking_states):
        ax.scatter(ts[0], ts[1], c=region_color(i), marker='x')

    # Plot evolution of q values stats, color-coded per region
    ax = axs[1]
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Estimated values')
    for val_idx in np.arange(0, nb_tracked, nb_samples_per_region):
        vals = tracked_vals[val_idx:val_idx + nb_samples_per_region]  # split per region
        vals_min = np.min(vals, axis=0)
        vals_max = np.max(vals, axis=0)
        vals_mean = np.mean(vals, axis=0)
        vals_std = np.std(vals, axis=0)
        vals_lower = vals_mean - vals_std
        vals_upper = vals_mean + vals_std
        label = 'Mean' if val_idx == 0 else None
        ax.plot(xrange, vals_mean, c=region_color(val_idx), lw=1.0, label=label)
        c = region_color(val_idx)
        label = '$min\\rightarrow -\sigma$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_min, vals_lower, edgecolor=c, facecolor='None', hatch='//', label=label, alpha=0.4)
        label = '$-\sigma\\rightarrow\sigma$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_lower, vals_upper, facecolor=c, alpha=0.2, label=label)
        label = '$\sigma\\rightarrow max$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_upper, vals_max, edgecolor=c, facecolor='None', hatch='\\\\', label=label,
                        alpha=0.4)
    ax.legend(loc='best')
    plt.savefig(os.path.join(save_dir, 'tracked_vals_per_region.png'))
    plt.show()


if __name__ == '__main__':
    # exp_pardir = 'sarsa_101222_002618'
    exp_pardir = get_latest_experiment('.')
    exp_subdir = 'default'
    print(f'Evaluating experiment: {exp_pardir}\n'
          f'Sub-experiment: {exp_subdir}')

    exp_dir = os.path.join(exp_pardir, exp_subdir)

    # plot_episodes(exp_dir=exp_dir, train_step=720, env_type=REDA, save_dir=exp_pardir, nrows=2, ncols=3)
    plot_experiment_training_stats(exp_dir, 'Linear RL')
    plot_weight_evolution(exp_dir, save_dir=exp_pardir)
    # plot_all_experiments_training_stats('.', ['default'], ['Linear RL'], exp_filter='0254')
    # plot_q_vals_region_sampling_tracking_states(experiment_dir=exp_dir, env_type=IREDA, save_dir=exp_pardir)
