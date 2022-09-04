import os
import numpy as np
from datetime import datetime as dt
import yaml
import matplotlib.pyplot as plt
import pickle as pkl

from training_utils import ExponentialDecay, Constant, get_training_utils_yaml_dumper, LinearDecay, StepDecay
from experiments_tile_coding.sarsa import train_instance; algo_name = 'sarsa'

if __name__ == '__main__':
    experiment_dir = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S')
    os.makedirs(experiment_dir)

    # lr_fun = ExponentialDecay(1.5e-1, 1000)
    # exp_fun = ExponentialDecay(1.0, 70000, label='EXP')
    exp_fun = LinearDecay(1.0, 0.0, 80000)
    # lr_fun = StepDecay(init=1e-2, decay_rate=.99, decay_every=1000)
    # lr_fun = CustomFunc()
    lr_fun = Constant(1e-1, label='LR')
    # exp_fun = Constant(0.1)

    nb_tilings, nb_bins = 16, 2
    gamma = 0.9
    train_params = dict(
        n_obs=3,
        n_act=3,
        nb_tilings=nb_tilings,
        nb_bins=nb_bins,
        env_save_path=experiment_dir,
        results_path=experiment_dir,
        lr_fun=lr_fun,
        exp_fun=exp_fun,
        nb_training_steps=100000,
        eval_every=500,
        eval_eps=20,
        save_every=500,
        gamma=gamma
    )
    with open(os.path.join(experiment_dir, "train_params.yml"), "w") as f:
        f.write(yaml.dump(train_params, Dumper=get_training_utils_yaml_dumper()))

    errors, eval_el_stats, iht_counts, lrs, env = train_instance(**train_params)
    with open(os.path.join(experiment_dir, 'training_stats.pkl'), 'wb') as f:
        d = dict(errors=errors, eval_el_stats=eval_el_stats, iht_counts=iht_counts)
        pkl.dump(d, f)

    eval_el_means = [item['mean'] for item in eval_el_stats]
    eval_el_medians = [item['median'] for item in eval_el_stats]
    eval_el_mins = [item['min'] for item in eval_el_stats]
    eval_el_maxs = [item['max'] for item in eval_el_stats]
    nb_evals = len(eval_el_medians)

    fig, axs = plt.subplots(4, 1, gridspec_kw=dict(height_ratios=[3, 5, 3, 3]), figsize=(15, 10))
    fig.suptitle(f'{algo_name.upper()} with Tile Coding\n'
                 f'Env={repr(env)}\n'
                 f'Gamma={gamma}, Tilings={nb_tilings}, Bins={nb_bins}\n'
                 f'{repr(lr_fun)}\n'
                 f'{repr(exp_fun)}', size=8)
    xrange = np.arange(len(errors))
    eval_xrange = np.linspace(0, train_params['nb_training_steps'], nb_evals)
    axs[0].plot(xrange, errors)
    axs[0].set_ylabel('TD error')
    axs[1].plot(eval_xrange, eval_el_means, ls='--', c='k', lw=0.8)
    axs[1].plot(eval_xrange, eval_el_medians, c='k', lw=0.5)
    axs[1].plot(eval_xrange, eval_el_mins, ls=':', c='k', lw=0.5)
    axs[1].plot(eval_xrange, eval_el_maxs, ls=':', c='k', lw=0.5)
    axs[1].set_ylabel('Eval ep lens')
    axs[2].plot(eval_xrange, iht_counts)
    axs[2].set_ylabel('Nb tiles')

    ax = axs[-1]
    ax_lr = ax
    ax_exp = ax.twinx()
    lr_line, = ax_lr.plot(xrange, lrs, c='k', ls=':')
    exp_line, = ax_exp.plot(xrange, [exp_fun(i) for i in xrange], c='r')
    ax_lr.legend(handles=[lr_line, exp_line], labels=['Learning rate', 'Exploration rate'])

    ax_exp.spines['right'].set_color('r')
    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'experiment_overview.png'))

    # plt.show()
