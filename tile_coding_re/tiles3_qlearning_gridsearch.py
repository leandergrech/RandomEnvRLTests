import os
from itertools import product
import numpy as np
from collections import deque
from pandas import Series
import matplotlib.pyplot as plt
from tqdm import trange, tqdm as pbar
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions
from tile_coding_re.heatmap_utils import make_heatmap, update_heatmap
from tile_coding_re.buffers import TrajBuffer
import gym
import pickle as pkl
from datetime import datetime as dt

'''
Name of RL method used to train agent
'''
method_name = 'TD Double Q-Learning with tile-coding'

'''
hparams holds all the hyperparameters used in each respective iteration of the gridsearch
'''
hparams = {}

'''
Choose environment and set state limits accordingly
'''
# env_name = 'MountainCar-v0'
# env = gym.make(env_name)
# env_lows, env_highs = env.observation_space.low, env.observation_space.high
# n_obs = env.observation_space.shape[0]
# n_act = 1
# act_dim = 3

# env_name = 'CartPole-v1'
# env = gym.make(env_name)
# env_highs = [2.4, 3, 0.2095, 3]
# env_lows = [-item for item in env_highs]
# n_obs = env.observation_space.shape[0]
# n_act = 1
# act_dim = 2

n_obs = n_act = 2
env = REDA(n_obs, n_act)
env_name = repr(env)
env_lows, env_highs = env.observation_space.low, env.observation_space.high
act_dim = 3


hparams['env_name'] = env_name
hparams['env_lows'] = env_lows
hparams['env_highs'] = env_highs


'''
Initialise the save directory
'''
par_dir = 'gridsearches'
gridsearch_name = 'decaying-rates-sweep-of-lr-and-exp'
experiment_name = f'{env_name}_coarse'

experiment_path = os.path.join(par_dir, gridsearch_name, experiment_name)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

'''
Set all constant parameters throughout the experiment
'''

# Tiling info
hparams['nb_tilings'] = nb_tilings = 8
hparams['nb_bins'] = nb_bins = 2

# Training info
hparams['nb_eps'] = nb_eps = 500
hparams['nb_runs'] = nb_runs = 5

# Hyper parameters
hparams['gamma'] = gamma = 0.99
hparams['init_lr'] = init_lr = 1.5e-1
hparams['lr_decay_every_eps'] = lr_decay_every_eps = 15  # int(nb_eps/10)
hparams['init_exploration'] = init_exploration = 1.0
hparams['exploration_decay_every_eps'] = exploration_decay_every_eps = 15  # int(nb_eps/10)

'''
Both LR and EXP are tested for the values of [0.1, 0.3, ..., 0.7, 0.9] respectively
'''
# Coarse tuning
lr_decay_rate_list = np.arange(0.1, 1.0, 0.2)
exploration_decay_rate_list = np.arange(0.1, 1.0, 0.2)
# Fine tuning
# lr_decay_rate_list = np.arange(0.8, 1.0, 0.05).tolist() + [0.99]
# exploration_decay_rate_list = np.arange(0.8, 1.0, 0.05).tolist() + [0.99]

# Step-wise decaying lr
lr_fun = lambda ep_i, decay_rate: init_lr * decay_rate ** (ep_i // lr_decay_every_eps)
lr_fun_str = f'{init_lr} x LR_DECAY^(ep_index//{lr_decay_every_eps})'
# Step-wise decaying exploration
exploration_fun = lambda ep_i, decay_rate: init_exploration * decay_rate ** (ep_i // exploration_decay_every_eps)
exploration_fun_str = f'{init_exploration} x EXP_DECAY^(ep_index//{exploration_decay_every_eps})'

max_nb_perms = len(lr_decay_rate_list) * len(exploration_decay_rate_list)
progress = None


def train(lr_decay_rate, exploration_decay_rate):
    global progress

    '''
    Set the changing hyperparameters
    '''
    hparams['lr_decay_rate'] = lr_decay_rate
    hparams['exploration_decay_rate'] = exploration_decay_rate

    ranges = [[l, h] for l, h in zip(env_lows, env_highs)]

    max_tiles = 2 ** 20

    # List of all possible discrete actions
    actions = get_discrete_actions(n_act, act_dim)

    def swap_q(q1, q2):
        if np.random.rand() < 0.5:
            return q1, q2
        else:
            return q2, q1

    def get_qvals(state, q):
        return [q.value(state, a_) for a_ in actions]

    def get_total_greedy_action(state, q1, q2):
        val1 = get_qvals(state, q1)
        val2 = get_qvals(state, q2)
        tot_val = [v1 + v2 for v1, v2 in zip(val1, val2)]
        action_idx = np.argmax(tot_val)

        return actions[action_idx]

    # Start training
    returns = np.zeros((nb_runs, nb_eps), dtype=float)
    errors = np.zeros((nb_runs, nb_eps), dtype=float)
    for run in range(nb_runs):
        tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
        q1 = QValueFunctionTiles3(tilings, actions)
        q2 = QValueFunctionTiles3(tilings, actions)

        for ep in range(nb_eps):
            o = env.reset()
            d = False
            ep_step = 0
            while not d:
                exploration = exploration_fun(ep, exploration_decay_rate)
                if np.random.rand() < exploration:
                    a = env.action_space.sample().tolist()
                else:
                    a = get_total_greedy_action(o, q1, q2)

                otp1, r, d, _ = env.step(a)

                returns[run, ep] += r

                qvfa, qvfb = swap_q(q1, q2)
                target = r + gamma * qvfb.value(otp1, qvfa.greedy_action(otp1))

                lr = lr_fun(ep, lr_decay_rate)
                error = qvfa.update(o, a, target, lr)
                errors[run, ep] += error

                o = otp1
                ep_step += 1
            errors[run, ep] /= ep_step
            progress.update(1)

    return returns, errors


def execute_grid():
    global progress
    # Set up grid of parameters to test
    permutation_test_params = [item for item in product(lr_decay_rate_list, exploration_decay_rate_list)]
    progress = pbar(total=max_nb_perms*nb_eps*nb_runs)
    for i, perm in enumerate(permutation_test_params):
        hparams['iteration'] = i
        returns, errors = train(*perm)

        param_result_set = dict(hparams=hparams, returns=returns, errors=errors)
        with open(os.path.join(experiment_path, f'permutation_{i}.pkl'), 'wb') as f:
            pkl.dump(param_result_set, f)
    progress.close()


def eval_plot():
    file_paths = []
    data_dict = {}
    nb_best = 3
    nb_worst = 3
    best_plots = deque(maxlen=nb_best)
    worst_plots = deque(maxlen=nb_worst)
    for i, file in enumerate(os.listdir(experiment_path)):
        fp = os.path.join(experiment_path, file)
        file_paths.append(fp)
        with open(fp, 'rb') as f:
            dat = pkl.load(f)
            mean_returns = np.mean(dat['returns'], axis=0)
            mean_returns_smooth = Series(mean_returns).rolling(5).mean().to_numpy()

            hp = dat['hparams']
            i = hp['iteration']
            lr_decay = hp['lr_decay_rate']
            exp_decay = hp['exploration_decay_rate']

            key = (i, lr_decay, exp_decay)
            # data_dict[key] = mean_returns
            data_dict[key] = mean_returns_smooth
            # data_dict[key] = mean_returns

            max_val = np.nanmax(mean_returns_smooth)
            # max_val = sum(mean_returns)
            if len(best_plots) < nb_best:
                best_plots.append((max_val, i))
            elif np.nanmin([item[0] for item in best_plots]) < max_val:
                j = np.argmin([item[0] for item in best_plots])
                best_plots[j] = (max_val, i)

            if len(worst_plots) < nb_worst:
                worst_plots.append((max_val, i))
            elif np.nanmax([item[0] for item in worst_plots]) >= max_val:
                j = np.argmax([item[0] for item in worst_plots])
                worst_plots[j] = (max_val, i)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'Environment: {env_name}\nMethod: {method_name}\nGAMMA={gamma}, NB_TILINGS={nb_tilings}, NB_BINS={nb_bins}\nLR function: {lr_fun_str}\nEXP function: {exploration_fun_str}\n'
                 f'LR_DECAY={lr_decay_rate_list}\nEXP_DECAY={exploration_decay_rate_list}')
    ax.set_xlabel('Training episodes')
    ax.set_ylabel('Total reward per episode')
    for label, dat in data_dict.items():
        ls = 'solid'
        if label[0] in [item[1] for item in best_plots]:
            actual_label = f'It#{label[0]},LR_DECAY={label[1]:.2f},EXP_DECAY={label[2]:.2f}'
            lw = 2
        elif label[0] in [item[1] for item in worst_plots]:
            actual_label = f'It#{label[0]},LR_DECAY={label[1]:.2f},EXP_DECAY={label[2]:.2f}'
            lw = 2
            ls = 'dashed'
        else:
            actual_label = None
            lw = 0.5
        ax.plot(dat, lw=lw, ls=ls, label=actual_label)
    fig.tight_layout()
    ax.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    execute_grid()
    eval_plot()

