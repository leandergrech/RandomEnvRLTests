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

from tile_coding_re.tiles3_qfunction import QValueFunctionTiles3
from random_env.envs.random_env_discrete_actions import RandomEnvDiscreteActions as REDA, get_discrete_actions
from utils.plotting_utils import make_heatmap
from utils.eval_utils import play_episode

from utils.eval_utils import get_q_func_filenames, get_q_func_xrange, get_val, get_q_func_step


def unpack_stats(arr, key, rolling):
    s = Series([item[key] for item in arr])
    return s.rolling(rolling).mean()


def get_eval_every_hack(yaml_path):
    with open(yaml_path, 'r') as f:
        for line in f:
            if 'eval_every' in line:
                return int(re.findall(r'\d+', line)[0])


SUB_EXP_COLORS = ['r', 'g', 'b', 'k'] # I never exceed 4 sub-experiments


# def plot_experiment_training_stats(exp_pardir, exp_subdirs, exp_labels):
#     """
#     Access training_stats.pkl for a single experiment and plot training stats.
#     Top plot:    IHT count evolution
#     Middle plot: Episode length mean+std obtained using latest q-table greedily
#     Bottom plot: Return mean obtained using latest q-table greedily
#     """
#     stats_files = [os.path.join(exp_pardir, subdir, 'training_stats.pkl') for subdir in exp_subdirs]
#
#     eval_every = get_eval_every_hack(os.path.join(exp_pardir, exp_subdirs[0], 'train_params.yml'))
#
#     fig, axs = plt.subplots(4, gridspec_kw=dict(height_ratios=[1, 3]), figsize=(15, 10))
#     for label, pkl_fn, c in zip(exp_labels, stats_files, SUB_EXP_COLORS):
#         with open(pkl_fn, 'rb') as f:
#             data = pkl.load(f)
#             iht_counts = data['iht_counts']
#             ep_lens = data['ep_lens']
#             returns = data['returns']
#             regrets = data['regrets']
#
#             xrange = np.arange(len(iht_counts)) * eval_every
#
#             ax = axs[0]
#             ax.plot(xrange, iht_counts, label=label)
#             ax.set_title('IHT counts')
#             ax.set_ylabel('Nb tiles discovered')
#
#             ax = axs[1]
#             ep_lens_mean = np.mean(ep_lens, axis=1)
#             ep_lens_std = np.std(ep_lens, axis=1)
#             ax.plot(xrange, ep_lens_mean, ls='dashed', c=c, label=f'{label} Mean')
#             ax.set_title('Using greedy policy')
#             ax.set_ylabel('Episode length')
#
#     for ax in axs:
#         ax.legend(loc='best', prop=dict(size=8))
#         ax.set_xlabel('Training steps')
#     fig.tight_layout()
#     plt.savefig(os.path.join(exp_pardir, 'training_stats.png'))


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
    iht_counts = defaultdict(list)
    xrange = None
    eval_every = None

    all_exp = []

    # Iterate over experiment with different environment
    for exp_name in sorted(os.listdir(exp_pardir)):
        if 'sarsa' not in exp_name or exp_filter not in exp_name:
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
                iht_counts[sub_exp].append(data['iht_counts'])

                if xrange is None:
                    xrange = np.arange(len(data['iht_counts'])) * eval_every

    print(f'Found {len(all_exp)} experiments')

    fig, axs = plt.subplots(4, gridspec_kw=dict(height_ratios=[1, 3, 3, 3]), figsize=(15, 10))
    for label, sub_exp, c in zip(exp_labels, exp_subdirs, SUB_EXP_COLORS):
        ihts= iht_counts[sub_exp]
        iht_mean = np.mean(ihts, axis=0)

        els = ep_lens[sub_exp]
        el_mean = np.mean(np.mean(els, axis=-1), axis=0)
        el_std = np.sqrt(np.mean(np.square(np.std(els, axis=-1)), axis=0))

        rets = returns[sub_exp]
        rets_mean = np.mean(np.mean(rets, axis=-1), axis=0)
        rets_std = np.sqrt(np.mean(np.square(np.std(rets, axis=-1)), axis=0))

        regs = regrets[sub_exp]
        regs_mean = np.mean(np.mean(regs, axis=-1), axis=0)
        regs_std = np.sqrt(np.mean(np.square(np.std(regs, axis=-1)), axis=0))

        ax = axs[0]
        ax.plot(xrange, iht_mean, ls='solid', c=c, label=f'{label} ' + r'$\mu$')
        ax.set_title('IHT counts')
        ax.set_ylabel('Nb tiles discovered')

        ax = axs[1]
        ax.plot(xrange, el_mean, ls='solid', c=c, label=f'{label} ' + r'$\mu$')
        ax.fill_between(xrange, el_mean - el_std, el_mean + el_std, facecolor='None', edgecolor=c, hatch='//', alpha=0.6)
        # ax.set_title('Using greedy policy')
        ax.set_ylabel('Episode length')

        ax = axs[2]
        ax.plot(xrange, rets_mean, ls='solid', c=c, label=f'{label} ' + r'$\mu$')
        # ax.fill_between(xrange, rets_mean - rets_std, rets_mean + rets_std, facecolor='None', edgecolor=c, hatch='//', alpha=0.6)
        ax.set_yscale('symlog')
        ax.set_ylabel('Returns')

        ax = axs[3]
        ax.plot(xrange, regs_mean, ls='solid', c=c, label=f'{label} ' + r'$\mu$')
        # ax.fill_between(xrange, regs_mean - regs_std, regs_mean + regs_std, facecolor='None', edgecolor=c, hatch='//', alpha=0.6)
        ax.set_ylabel('Regrets')

    for ax in axs:
        ax.legend(loc='best', prop=dict(size=10))
        y_grid_on(ax)
        ax.set_xlabel('Training steps')
    fig.suptitle(f'{len(all_exp)} different environments')
    fig.tight_layout()
    plt.savefig(os.path.join(exp_pardir, 'results.png'))


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


def plot_q_vals_grid_tracking_states(experiment_dir, n_tracking_dim=5, env_type=REDA, save_dir=None):
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
        q = QValueFunctionTiles3.load(qfn)
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


def circular_normal_sample(r, s):
    if r > 0:
        R = np.random.normal(r, s)
        theta = 2 * np.pi * np.random.rand()
        return [R * np.cos(theta), R * np.sin(theta)]
    else:
        return np.random.normal(r, s, 2)


def plot_q_vals_region_sampling_tracking_states(experiment_dir, env_type=REDA, save_dir=None):
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
        q = QValueFunctionTiles3.load(qfn)
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


def plot_individual_action_advantage(experiment_dir, query_step, env_type=REDA, save_dir=None, nb_test_eps=3):
    if save_dir is None:
        save_dir = experiment_dir

    env = env_type(2, 2, state_clip=0.0)
    env.load_dynamics(os.path.join(experiment_dir, 'REDAClip_0.0clip_2obsx2act_dynamics.pkl'))
    env.state_clip = 0.0

    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    # Searching for training data
    q_func_filenames = get_q_func_filenames(experiment_dir)
    training_steps = [get_q_func_step(qfn) for qfn in q_func_filenames]

    closest_training_step_idx = np.argmin(np.abs(np.subtract(training_steps, query_step)))
    qfn_at_step = q_func_filenames[closest_training_step_idx]
    at_training_step = get_q_func_step(qfn_at_step)

    # Initialise grid tracking states
    n_tracking_dim = 64
    tracking_lim = 1.2
    tracking_ranges = [[-tracking_lim, -tracking_lim], [tracking_lim, tracking_lim]]
    tracking_ranges = [[l, h] for l, h in zip(*tracking_ranges)]
    tracking_states = np.array(
        [list(item) for item in product(*np.array([np.linspace(l, h, n_tracking_dim) for l, h in tracking_ranges]))])

    # Load Q-table
    q = QValueFunctionTiles3.load(qfn_at_step)

    # Play test episodes to superimpose on heatmaps
    test_obses = []
    test_acts = []
    for _ in range(nb_test_eps):
        obses, acts, _ = play_episode(env, q, nball_uniform_sample(0.8, 0.9))
        if len(obses) > 50:
            test_obses.append(obses)
            test_acts.append(acts)
            if len(test_obses) > 10:
                break

    # Estimate and store state values
    tracking_estimated_vals = []
    for ts in tracking_states:
        tracking_estimated_vals.append(np.mean([q.value(ts, a_) for a_ in range(nb_actions)]))

    # Plot the fuckin shit already
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = np.ravel(axs)
    min_adv = np.inf
    max_adv = -np.inf
    ims = []
    for action_idx, (ax, act) in enumerate(zip(axs, actions)):
        init_state = np.array([0.0, 0.0])
        env.reset(init_state.copy())
        otp1, *_ = env.step(act)
        ax.plot(*np.vstack([init_state, otp1]).T, c='k', marker='x', zorder=20)

        # Plot test episodes
        for tobses, tacts in zip(test_obses, test_acts):
            tobses = np.array(tobses).T
            ax.plot(tobses[0], tobses[1], c='k', zorder=25)

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
                 f'Training step = {at_training_step}')

    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.88, wspace=0.083, hspace=0.321)
    save_path = os.path.join(save_dir, f'individual_action_advs_at-step-{at_training_step}.png')
    plt.savefig(save_path)


def plot_trims_during_training(experiment_dir, env_type, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(experiment_dir, 'log_trims')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env = env_type.load_from_dir(experiment_dir)
    n_obs = env.obs_dimension
    n_act = env.act_dimension
    actions = get_discrete_actions(n_act, 3)
    nb_actions = len(actions)

    # Searching for training data
    q_func_filenames = get_q_func_filenames(experiment_dir)

    # Create tracking states
    tracking_states = create_grid_tracking_states(env, 15)
    nb_tracked = len(tracking_states)

    # Initialise array holding trim coordinates
    tracked_trims = np.array([[ts, np.zeros_like(ts)] for ts in tracking_states])

    def update_trims(qvf: QValueFunctionTiles3):
        for idx, ts in enumerate(tracking_states):
            action_idx = qvf.greedy_action(ts)
            a = actions[action_idx]
            env.reset(ts.copy())
            tstp1, *_ = env.step(a)
            tracked_trims[idx, 1] = tstp1

    for i, qfn in enumerate(q_func_filenames):
        q = QValueFunctionTiles3.load(qfn)
        update_trims(q)
        fig, ax = plt.subplots()
        for tt in tracked_trims:
            X, Y = tt.T
            dx, dy = np.array(tt[1] - tt[0]).T
            ax.arrow(X[0], Y[0], dx, dy, color='k', width=0.01, lw=.8, zorder=5)
        ax.add_patch(mpl.patches.Circle((0,0), env.GOAL, edgecolor='g', facecolor='None', ls='--'))
        q_step = get_q_func_step(qfn)
        fig.suptitle(f'Training step={q_step}')
        save_path = os.path.join(save_dir, f'{q_step}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(save_path)


