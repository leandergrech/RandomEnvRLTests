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
    r = np.random.normal(0.9, 0.1)
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
            eval_obses, el_stats = eval_agent(eval_env, q, kwargs.get('eval_eps'))
            eplen_stats.append(el_stats)
            iht_counts.append(tilings.count())
            # make_state_violins(eval_obses['initial_observations'], eval_obses['terminal_observations'], os.path.join(results_path, 'obses_violins', f'step-{T}.png'))
        if (T + 1) % kwargs['save_every'] == 0 or T == 0:
            save_dir = kwargs.get('results_path')
            save_dir = os.path.join(save_dir, 'q_func')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, f'q_step_{T}.pkl')
            q.save(save_path)

    return td_errors, eplen_stats, iht_counts, lrs, env





