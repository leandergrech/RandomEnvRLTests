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

from utils.plotting_utils import make_heatmap, y_grid_on
from linear_q_function import QValueFunctionLinear
from random_env.envs import REDAClip


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


def plot_episodes(exp_name, sub_exp_name, train_step, nrows=2, ncols=4):
    exp_dir = os.path.join(exp_name, sub_exp_name)
    env = REDAClip.load_from_dir(exp_dir)
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
        acts = np.array(acts)

        ax_obs = axs[(i // ncols) * 2, i % ncols]
        ax_act = axs[(i // ncols) * 2 + 1, i % ncols]

        ax_obs.set_title(f'Ep {i + 1}', size=15)
        ax_obs.axhline(-env.GOAL, c='g', ls='--', lw=2)
        ax_obs.axhline(env.GOAL, c='g', ls='--', lw=2)
        ax_obs.plot(obses, c='b')
        ax_act.axhline(0.0, c='k', ls='-.', lw=2)
        ax_act.plot(acts, c='r')

        for ax, ylab in zip((ax_obs, ax_act), ('States', 'Actions')):
            y_grid_on(ax)
            ax.set_xticks(np.arange(step))
            ax.set_ylabel(ylab, size=12)

        ax_obs.get_shared_x_axes().join(ax_obs, ax_act)
        ax_act.set_xlabel('Step', size=12)

    fig.suptitle(f'Environment: {repr(env)}\n'
                 f'Experiment:  {exp_name}\n'
                 f'Sub-exp:     {sub_exp_name}\n'
                 f'At step:     {train_step}')
    fig.tight_layout()
    fig.savefig(os.path.join(exp_name, f'{sub_exp_name}_{train_step}_step.png'))
    plt.show()


def plot_weight_evolution(exp_pardir, exp_subdir):
    exp_dir = os.path.join(exp_pardir, exp_subdir)
    q_func_fn = sorted([os.path.join(exp_dir, 'q_func', item) for item in os.listdir(os.path.join(exp_dir, 'q_func'))],
                       key=lambda item: int(os.path.splitext(item)[0].split('_')[-1]))
    weights = []
    for qfn in q_func_fn:
        q = QValueFunctionLinear.load(qfn)
        weights.append(q.w)
    weights = np.array(weights)
    fig, ax = plt.subplots()
    ax.plot(weights)
    plt.show()


if __name__ == '__main__':
    # exp_pardir = 'sarsa_100922_025408_0'
    exp_pardir = 'sarsa_100922_032753_0'
    exp_subdir = 'default'
    # plot_all_experiments_training_stats('.', ['default'], ['Linear RL'], exp_filter='0254')
    # plot_episodes(exp_pardir, exp_subdir, 3439, 2, 4)
    plot_weight_evolution(exp_pardir, exp_subdir)