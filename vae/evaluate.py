import os
import numpy as np
import torch as t
import re
from collections import defaultdict
from vae import VAE
from utils.plotting_utils import make_heatmap

import matplotlib.pyplot as plt

def compare_losses_vs_envsz(exp_dirs):
    nb_exps = len(exp_dirs)
    losses = [defaultdict(dict) for _ in range(nb_exps)]

    stats_steps = 50

    for i, exp_dir in enumerate(exp_dirs):
        seed = int(re.findall(r'\d+', exp_dir)[-1])
        for p in os.listdir(exp_dir):
            n_obs, n_act, latent_dim, _ = re.findall(r'\d+', p)[-4:]
            env_sz = int(n_obs)
            if env_sz > 5:
                continue
            latent_dim = int(latent_dim)

            exp_loss = np.load(os.path.join(exp_dir, p, 'losses.npy'))

            losses[i][env_sz][latent_dim] = exp_loss[-stats_steps:]


    fig, ax = plt.subplots()
    x = np.arange(2, 6)
    y = np.arange(2, 6)
    ax.set_ylabel('Latent dim')
    ax.set_xlabel('Env. size')

    z = []
    fig, ax = plt.subplots()

    import matplotlib as mpl
    cmap = mpl.cm.jet

    for env_sz in x:
        print(f'-> Env. size={env_sz}')
        z.append([])
        for lat in y:
            print(f' `-> Latent dim = {lat}')
            loss_set = np.array([l[env_sz].get(lat, np.zeros(stats_steps)) for l in losses])
            loss_set_mean = np.mean(np.mean(loss_set, axis=0))

            ax.scatter(env_sz, loss_set_mean, c=cmap((lat - 2)/(max(x) - 2)))

            z[-1].append(loss_set_mean)

    plt.show()

    title = f'Env. size vs. Latent dim final loss\nLast {stats_steps} samples used for stats'
    im = make_heatmap(ax, z, x, y, title)
    ax.set_xlabel('Env. size')
    ax.set_ylabel('Latent size')
    ax.set_xlim((2, 5))
    ax.set_ylim((2, 5))
    plt.show()

if __name__ == '__main__':
    exp_dirs = []
    for fn in os.listdir('.'):
        if '234' in fn:
            continue
        if 'random_trajectory' in fn:
            exp_dirs.append(fn)

    compare_losses_vs_envsz(exp_dirs)
