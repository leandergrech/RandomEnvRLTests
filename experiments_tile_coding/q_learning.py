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
from training_utils import *#ExponentialDecay, get_training_utils_yaml_dumper

from tqdm import trange

# sns.set_context('paper')
sns.set_theme(style='whitegrid')


def eval_agent(eval_env, q):
    nb_eps = 100
    init_obses = np.empty(shape=(0, eval_env.obs_dimension))
    terminal_obses = np.empty(shape=(0, eval_env.obs_dimension))
    ep_lens = []

    actions = get_discrete_actions(eval_env.act_dimension, 3)
    for _ in range(nb_eps):
        o = eval_env.reset()
        d = False
        init_obses = np.vstack([init_obses, o])
        t = 0
        while not d:
            a = q.greedy_action(o)
            otp1, r, d, _ = eval_env.step(actions[a])

            o = otp1
            t += 1
        terminal_obses = np.vstack([terminal_obses, o])
        ep_lens.append(t)

    return init_obses, terminal_obses, np.mean(ep_lens)


def make_state_violins(init_obses, terminal_obses, path):
    data = pd.DataFrame()
    for obs_type, obses in zip(('init', 'terminal'), (init_obses, terminal_obses)):
        for dim, vals in enumerate(obses.T):
            df = pd.DataFrame(dict(obs_type=obs_type, dim=dim, vals=vals))
            data = pd.concat([data, df], ignore_index=True)

    plot = sns.violinplot(data=data, x='dim', y='vals', hue='obs_type', split=True, inner='quart', lw=1)
    plot.axhline(0.0, color='k')
    fig = plot.get_figure()

    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    plt.savefig(path)
    plt.close(fig)


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

    o = env.reset()
    t = 0

    results_path = kwargs.get('results_path')

    '''
    TRAINING LOOP
    '''
    td_errors = np.zeros(nb_training_steps)
    eval_ep_lens = []
    iht_counts = []
    for T in trange(nb_training_steps):
        if np.random.rand() < exp_fun(T):
            a = np.random.choice(n_actions)
        else:
            a = q.greedy_action(o)
        otp1, r, d, _ = env.step(actions[a])

        target = r + gamma * q.value(otp1, q.greedy_action(otp1))

        td_errors[T] = q.update(o, a, target, lr_fun(T))

        if d:
            t = 0
            o = env.reset()
        else:
            t += 1
            o = otp1

        if (T + 1) % eval_every == 0:
            init_obses, terminal_obses, mean_ep_len = eval_agent(eval_env, q)
            eval_ep_lens.append(mean_ep_len)
            iht_counts.append(tilings.count())
            make_state_violins(init_obses, terminal_obses, os.path.join(results_path, 'obses_violins', f'step-{T}.png'))

    return td_errors, eval_ep_lens, iht_counts


if __name__ == '__main__':
    experiment_dir = dt.now().strftime('%m%d%y_%H%M%S')
    os.makedirs(experiment_dir)

    # lr_fun = ExponentialDecay(1.5e-1, 1000)
    # exp_fun = ExponentialDecay(1.0, 1000)
    lr_fun = StepDecay(init=1.5e-1, decay_rate=.99, decay_every=1000)
    exp_fun = StepDecay(init=1.0, decay_rate=.85, decay_every=1000)

    train_params = dict(
        n_obs=2,
        n_act=2,
        nb_tilings=8,
        nb_bins=2,
        env_save_path=experiment_dir,
        results_path=experiment_dir,
        lr_fun=lr_fun,
        exp_fun=exp_fun,
        nb_training_steps=100000,
        eval_every=500,
        gamma=0.99
    )
    with open(os.path.join(experiment_dir, "train_params.yml"), "w") as f:
        f.write(yaml.dump(train_params, Dumper=get_training_utils_yaml_dumper()))

    errors, eval_ep_lens, iht_counts = train_instance(**train_params)

    fig, axs = plt.subplots(4, 1, gridspec_kw=dict(height_ratios=[3, 3, 3, 3]))
    xrange = np.arange(len(errors))
    eval_xrange = np.linspace(0, train_params['nb_training_steps'], len(eval_ep_lens))
    axs[0].plot(xrange, errors)
    axs[0].set_ylabel('TD error')
    axs[1].plot(eval_xrange, eval_ep_lens)
    axs[1].set_ylabel('Eval ep lens')
    axs[2].plot(eval_xrange, iht_counts)
    axs[2].set_ylabel('Nb tiles')

    ax = axs[-1]
    ax_lr = ax
    ax_exp = ax.twinx()
    lr_line, = ax_lr.plot(xrange, [lr_fun(i) for i in xrange], c='k', ls=':')
    exp_line, = ax_exp.plot(xrange, [exp_fun(i) for i in xrange], c='r')
    ax_lr.legend(handles=[lr_line, exp_line], labels=['Learning rate', 'Exploration rate'])

    ax_exp.spines['right'].set_color('r')
    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'experiment_overview.png'))

    plt.show()



