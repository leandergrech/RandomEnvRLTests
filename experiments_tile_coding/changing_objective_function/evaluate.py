import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series


def unpack_stats(arr, key, rolling):
    s = Series([item[key] for item in arr])
    return s.rolling(rolling).mean()

def plot_individual_experiments(experiment_pardir):
    # experiment_pardir = 'sarsa_090222_020057'
    quadratic_stats_file = os.path.join(experiment_pardir,'quadratic-objective/training_stats.pkl')
    rms_stats_file = os.path.join(experiment_pardir,'rms-objective/training_stats.pkl')

    eval_every = 500
    fig, axs = plt.subplots(2, gridspec_kw=dict(height_ratios=[1,3]), figsize=(15, 10))
    for label, pkl_fn, c in zip(('Quadratic', 'RMS'), (quadratic_stats_file, rms_stats_file), ('k', 'b')):
        with open(pkl_fn, 'rb') as f:
            data = pkl.load(f)
            el_stats = data['eval_el_stats']
            iht_counts = data['iht_counts']

            xrange = np.arange(len(iht_counts)) * eval_every

            ax = axs[0]
            ax.plot(xrange, iht_counts, label=label)
            ax.set_title('IHT counts')
            ax.set_ylabel('Nb tiles discovered')

            ax = axs[1]

            ax.plot(xrange, unpack_stats(el_stats, 'mean', 1), ls='dashed', c=c, label=f'{label} Mean')
            ax.plot(xrange, unpack_stats(el_stats, 'median', 1), ls='solid', c=c, label=f'{label} Median')
            ax.plot(xrange, unpack_stats(el_stats, 'min', 1), ls='dotted', c=c, label=f'{label} Min')
            ax.plot(xrange, unpack_stats(el_stats, 'max', 1), ls='solid', lw=0.5, c=c, label=f'{label} Max')
            ax.set_title('Using greedy policy')
            ax.set_ylabel('Episode length')


    for ax in axs:
        ax.legend(loc='best', prop=dict(size=8))
        ax.set_xlabel('Training steps')
    fig.tight_layout()
    plt.savefig(os.path.join(experiment_pardir, 'QO_vs_RO/results.png'))
    # plt.show()

from collections import defaultdict
def plot_all_experiments():
    experiment_pardir = '.'
    # sub_experiments_pardirs = ['quadratic-objective', 'rms-objective']
    sub_experiments_pardirs = ['quadratic-objective', 'quadratic-objective-x5', 'quadratic-objective-x10', 'rms-objective']

    ep_lens = defaultdict(lambda : defaultdict(list))
    iht_counts = defaultdict(list)
    xrange = None
    eval_every = 500

    all_exp = []

    # Iterate over experiment with different environment
    for exp_name in sorted(os.listdir(experiment_pardir)):
        if 'sarsa' not in exp_name:
            continue

        all_exp.append(exp_name)
        experiment_dir = os.path.join(experiment_pardir, exp_name)

        # Iterate over different state initialisation schemes
        for sub_exp in sub_experiments_pardirs:
            pkl_file = os.path.join(experiment_dir, sub_exp, 'training_stats.pkl')
            with open(pkl_file, 'rb') as f:
                data = pkl.load(f)

                el_stats = data['eval_el_stats']
                for k in el_stats[0].keys():
                    ep_lens[sub_exp][k].append(unpack_stats(el_stats, k, 1))

                iht_counts[sub_exp].append(data['iht_counts'])

                if xrange is None:
                    xrange = np.arange(len(data['iht_counts'])) * eval_every

    fig, axs = plt.subplots(2, gridspec_kw=dict(height_ratios=[1, 3]), figsize=(15, 10))
    # for label, sub_exp, c in zip(('Quadratic', 'RMS'), sub_experiments_pardirs, ('b', 'k')):
    for label, sub_exp, c in zip(('QO', 'QOx5', 'QOx10', 'RO'), sub_experiments_pardirs, ('b', 'g', 'r', 'k')):
        els = ep_lens[sub_exp]
        el_mean = np.mean(els['mean'], axis=0)
        el_med = np.mean(els['median'], axis=0)
        el_min = np.mean(els['min'], axis=0)
        el_max = np.mean(els['max'], axis=0)

        ihts= iht_counts[sub_exp]
        iht_mean = np.mean(ihts, axis=0)
        iht_min = np.min(ihts, axis=0)
        iht_max = np.max(ihts, axis=0)

        ax = axs[0]
        ax.plot(xrange, iht_mean, ls='solid', c=c, label=f'{label} Mean')
        # ax.plot(xrange, iht_min, ls='dotted', c=c, label=f'{label} Min')
        # ax.plot(xrange, iht_max, ls='dashed', c=c, label=f'{label} Max')
        ax.set_title('IHT counts')
        ax.set_ylabel('Nb tiles discovered')
        ax.set_yscale('log')
        # ax.set_xscale('log')

        ax = axs[1]
        ax.plot(xrange, el_mean, ls='solid', c=c, label=f'{label} Mean')
        # ax.plot(xrange, el_med, ls='dashed', c=c, label=f'{label} Ave. Med.')
        ax.plot(xrange, el_min, ls='dotted', c=c, label=f'{label} Min')
        ax.plot(xrange, el_max, ls='dashed', c=c, label=f'{label} Max')
        ax.set_title('Using greedy policy')
        ax.set_ylabel('Episode length')
        # ax.set_yscale('log')
        # ax.set_xscale('log')

    for ax in axs:
        ax.legend(loc='best', prop=dict(size=10))
        ax.set_xlabel('Training steps')
    fig.suptitle(f'Objective function experiment with\n{len(all_exp)} different environments')
    fig.tight_layout()
    plt.savefig(os.path.join(experiment_pardir, 'results.png'))

if __name__ == '__main__':
    # plot_individual_experiments('sarsa_090522_233119')
    plot_all_experiments()














