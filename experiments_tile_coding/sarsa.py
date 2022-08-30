import os
import yaml
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from experiments_tile_coding.eval_utils import eval_agent, make_state_violins
from training_utils import *#ExponentialDecay, get_training_utils_yaml_dumper

from tqdm import trange

# sns.set_context('paper')
sns.set_theme(style='whitegrid')

def obs_init_func():
    r = np.random.normal(0.8, 0.1)
    theta = 2 * np.pi * np.random.rand()

    return np.array([r * np.cos(theta), r * np.sin(theta)])
    # return np.random.uniform(-1, 1, 3)


def train_instance(**kwargs):
    """
    REDA training function. Kwargs holds all configurable training parameters and is saved to file
    :param kwargs: Many, many wonderful things in this dict
    :return: Absolutely nothing
    """
    '''
    ENVIRONMENT
    '''
    n_obs = kwargs.get('n_obs')
    n_act = kwargs.get('n_act')

    env_save_path = kwargs.get('env_save_path', None)
    env = REDA(n_obs, n_act)
    eval_env = REDA(n_obs, n_act, model_info=env.model_info)

    if env_save_path:
        env.save_dynamics(env_save_path)

    actions = get_discrete_actions(n_act, 3)
    n_actions = len(actions)
    env_ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]

    '''
    VALUE ESTIMATION
    '''
    nb_tilings = kwargs.get('nb_tilings')
    nb_bins = kwargs.get('nb_bins')
    tilings = Tilings(nb_tilings, nb_bins, env_ranges, 2**20)
    q = QValueFunctionTiles3(tilings, n_actions)

    '''
    TRAINING PARAMETERS
    '''
    lr_fun = kwargs.get('lr_fun')
    exp_fun = kwargs.get('exp_fun')

    gamma = kwargs.get('gamma')
    nb_training_steps = kwargs.get('nb_training_steps')
    eval_every = kwargs.get('eval_every')

    def get_action(t, obs):
        if np.random.rand() < exp_fun(t):
            act = np.random.choice(n_actions)
        else:
            act = q.greedy_action(obs)
        return act


    '''
    TRAINING LOOP
    '''
    td_errors = np.zeros(nb_training_steps)
    eplen_stats = []
    iht_counts = []
    lrs = []

    o = env.reset(obs_init_func())
    a = q.greedy_action(o)

    for T in trange(nb_training_steps):
        otp1, r, d, _ = env.step(actions[a])
        a_ = get_action(T, otp1)

        target = r + gamma * q.value(otp1, a_)

        # td_errors[T] = q.update(o, a, target, lr_fun(T))
        lr = lr_fun(T)
        lrs.append(lr)
        td_errors[T] = q.update(o, a, target, lr)

        if d:
            o = env.reset(obs_init_func())
            a = get_action(T, o)
        else:
            o = otp1.copy()
            a = a_

        if (T + 1) % eval_every == 0:
            init_obses, terminal_obses, el_stats = eval_agent(eval_env, q, kwargs.get('eval_eps'))
            eplen_stats.append(el_stats)
            iht_counts.append(tilings.count())
            # make_state_violins(init_obses, terminal_obses, os.path.join(results_path, 'obses_violins', f'step-{T}.png'))

    return td_errors, eplen_stats, iht_counts, lrs, env


if __name__ == '__main__':
    experiment_dir = dt.now().strftime('%m%d%y_%H%M%S')
    os.makedirs(experiment_dir)

    # lr_fun = ExponentialDecay(1.5e-1, 1000)
    exp_fun = ExponentialDecay(1.0, 150000, label='EXP')
    # lr_fun = StepDecay(init=1e-2, decay_rate=.99, decay_every=1000)
    # exp_fun = StepDecay(init=1.0, decay_rate=.99, decay_every=1000)
    # lr_fun = CustomFunc()
    lr_fun = Constant(1e-1, label='LR')
    # exp_fun = Constant(0.1)

    nb_tilings, nb_bins = 16, 2
    gamma = 0.9
    train_params = dict(
        n_obs=2,
        n_act=2,
        nb_tilings=nb_tilings,
        nb_bins=nb_bins,
        env_save_path=experiment_dir,
        results_path=experiment_dir,
        lr_fun=lr_fun,
        exp_fun=exp_fun,
        nb_training_steps=15000,
        eval_every=500,
        eval_eps=20,
        gamma=gamma
    )
    with open(os.path.join(experiment_dir, "train_params.yml"), "w") as f:
        f.write(yaml.dump(train_params, Dumper=get_training_utils_yaml_dumper()))

    errors, eval_el_stats, iht_counts, lrs, env = train_instance(**train_params)

    eval_el_medians = [item['median'] for item in eval_el_stats]
    eval_el_mins = [item['min'] for item in eval_el_stats]
    eval_el_maxs = [item['max'] for item in eval_el_stats]
    nb_evals = len(eval_el_medians)

    fig, axs = plt.subplots(4, 1, gridspec_kw=dict(height_ratios=[3, 3, 3, 3]))
    fig.suptitle(f'SARSA with Tile Coding\n'
                 f'Env={repr(env)}\n'
                 f'Gamma={gamma}, Tilings={nb_tilings}, Bins={nb_bins}\n'
                 f'{repr(lr_fun)}\n'
                 f'{repr(exp_fun)}', size=8)
    xrange = np.arange(len(errors))
    eval_xrange = np.linspace(0, train_params['nb_training_steps'], nb_evals)
    axs[0].plot(xrange, errors)
    axs[0].set_ylabel('TD error')
    axs[1].plot(eval_xrange, eval_el_medians, c='k')
    axs[1].plot(eval_xrange, eval_el_mins, ls=':', c='k')
    axs[1].plot(eval_xrange, eval_el_maxs, ls=':', c='k')
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

    plt.show()



