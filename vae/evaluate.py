import os
import numpy as np
import re
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.plotting_utils import make_heatmap


def parse_info_txt(info_path, pattern, val_type):
    ret = None
    with open(info_path, 'r') as f:
        for line in f:
            if pattern in line:
                ret = val_type(line.split(':')[-1])
            break
    return ret


def compare_losses_vs_envsz(exp_dir):
    stats_steps = 50

    print(f'-> Loading losses for {exp_dir} from file')
    losses = defaultdict(lambda: defaultdict(list))
    for p in os.listdir(exp_dir):
        if os.path.splitext(p)[-1] is not '':
            continue

        n_obs, n_act, latent_dim, seed = re.findall(r'\d+', p)[:4]
        env_sz = int(n_obs)
        latent_dim = int(latent_dim)

        exp_loss = np.load(os.path.join(exp_dir, p, 'losses.npy'))

        losses[env_sz][latent_dim].append(exp_loss)

    print(f'-> Computing losses stats')
    losses_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(np.ndarray)))
    for env_sz, v in losses.items():
        for latent_dim, w in v.items():
            max_len = max([len(item) for item in w])
            w = np.array([np.pad(item, (0, max_len-len(item)), mode='constant', constant_values=np.nan) for item in w])
            losses_stats[env_sz][latent_dim]['mean'] = np.nanmean(w, axis=0)
            losses_stats[env_sz][latent_dim]['median'] = np.nanmedian(w, axis=0)
            losses_stats[env_sz][latent_dim]['std'] = np.nanstd(w, axis=0)

    env_szs = sorted(losses.keys())
    latent_szs = np.copy(env_szs)
    n_envs = len(env_szs)
    loss_improvement = np.ones(shape=(n_envs, n_envs)) * np.nan
    loss_improvement_abs = np.ones(shape=(n_envs, n_envs)) * np.nan

    # fig, ax = plt.subplots(figsize=(15, 12))
    # cmap = mpl.cm.jet

    for i, (env_sz, v) in enumerate(sorted(losses_stats.items())):
        print(f'-> Env. size={env_sz}')
        for j, (lat, w) in enumerate(sorted(v.items())):
            print(f' `-> Latent dim = {lat}')

            val_mean = w['mean']
            val_median = w['median']
            val_std = w['std']
            xrange = np.arange(len(val_mean))

            val = val_mean
            impr = -(min(val) - max(val)) / max(val)
            loss_improvement[i, j] = impr * 100.0

            val = losses[env_sz][lat]
            impr = -(np.min(val) - np.max(val)) / np.max(val)
            loss_improvement_abs[i, j] = impr * 100.0

            # c = cmap((n_envs * env_sz + (lat - min(latent_szs))) / (n_envs ** 2))
            # ax.plot(xrange, val_mean, c=c, label=f'env_sz={env_sz}|lat={lat}')
            # # ax.fill_between(xrange, val_mean - val_std, val_mean + val_std, color=c, alpha=0.4)
    # ax.set_xlim((2, 10))
    # ax.set_ylim((2, 10))
    # ax.legend(loc='best', prop=dict(size=8), ncols=2)
    # fig.tight_layout()

    # exit(69)
    fig, ax = plt.subplots()
    # title = f'Env. size vs. Latent dim final loss\nMean loss % improvement\n'
    title = f'Env. size vs. Latent dim final loss\nAbsolute loss % improvement\n'
    # im, cb = make_heatmap(ax, loss_improvement.T, env_szs, latent_szs, title)
    im, cb = make_heatmap(ax, loss_improvement_abs.T, env_szs, latent_szs, title)
    # cb.set_label('Mean loss % improvement')
    cb.set_label('Absolute loss % improvement')
    ax.set_ylabel('Bottleneck size')
    ax.set_xlabel('Environment size M: RE_MobsxMact')
    # ax.set_xlim((2, 5))
    # ax.set_ylim((2, 5))
    fig.tight_layout()
    plt.show()


def evaluate_reconstruction(exp_path):
    import torch as t
    vae = None
    for fn in os.listdir(exp_path):
        if 'VAE' not in fn:
            continue
        vae = t.load(os.path.join(exp_path, fn))
        break

    fig, ax = plt.subplots()
    cmap = mpl.cm.jet
    for i in range(5):

        ax.plot)


if __name__ == '__main__':
    for fn in os.listdir('.'):
        if 'random_trajectory' in fn:
            exp_dir = fn
            print(f'-> Working on {exp_dir}')

            compare_losses_vs_envsz(exp_dir)
            break
