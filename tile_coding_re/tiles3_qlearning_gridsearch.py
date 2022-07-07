import os
from itertools import product
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
from tqdm import trange
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions
from tile_coding_re.heatmap_utils import make_heatmap, update_heatmap
from tile_coding_re.buffers import TrajBuffer
import gym
import pickle as pkl
from datetime import datetime as dt

par_dir = 'gridsearches'
experiment_name = 'decaying-rates-sweep-of-lr-and-exp'
experiment_path = os.path.join(par_dir, experiment_name)
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

hparams = {}

hparams['env_name'] = env_name = 'MountainCar-v0'
env = gym.make(env_name)
env_lows, env_highs = env.observation_space.low, env.observation_space.high
hparams['env_lows'] = env_lows
hparams['env_highs'] = env_highs

# env_name = 'CartPole-v1'
# env = gym.make(env_name)
# env_highs = [2.4, 3, 0.2095, 3]
# env_lows = [-item for item in env_highs]

n_obs = env.observation_space.shape[0]
n_act = 1

# Tiling info
hparams['nb_tilings'] = nb_tilings = 8
hparams['nb_bins'] = nb_bins = 2

# Training info
hparams['nb_eps'] = nb_eps = 10
hparams['nb_runs'] = nb_runs = 2

# Hyper parameters
# # Linear decaying lr
# init_lr = 1e-1
# final_lr = 1e-2
# lr_decay_eps = 1000
# lr_fun = lambda ep_i: final_lr + (init_lr - final_lr) * max(0, (1 - ep_i/lr_decay_eps))
# lr_str = f'Linear decay LR:  {init_lr:.2}->{final_lr} in {lr_decay_eps} episodes'

# Linear decaying exploration
# init_exploration = 1.
# final_exploration = 0.0
# exploration_decay_eps = 100
# exploration_fun =  lambda ep_i: final_exploration + (init_exploration - final_exploration) * max(0, (1. - ep_i/exploration_decay_eps))
# exploration_str = f'Linear decay EPS: {init_exploration}->{final_exploration} in {exploration_decay_eps} episodes'

hparams['gamma'] = gamma = 0.99
hparams['init_lr'] = init_lr = 1.5e-1
hparams['lr_decay_every_eps'] = lr_decay_every_eps = 15  # int(nb_eps/10)
hparams['init_exploration'] = init_exploration = 1.0
hparams['exploration_decay_every_eps'] = exploration_decay_every_eps = 15  # int(nb_eps/10)

lr_decay_rate_list = np.arange(0.1, 1.0, 0.8)
exploration_decay_rate_list = np.arange(0.1, 1.0, 0.8)

def train(lr_decay_rate, exploration_decay_rate):
    hparams['lr_decay_rate'] = lr_decay_rate
    hparams['exploration_decay_rate'] = exploration_decay_rate

    # Step-wise decaying lr
    lr_fun = lambda ep_i: init_lr * lr_decay_rate ** (ep_i // lr_decay_every_eps)

    # Step-wise decaying exploration
    exploration_fun = lambda ep_i: init_exploration * exploration_decay_rate ** (ep_i // exploration_decay_every_eps)

    ranges = [[l, h] for l, h in zip(env_lows, env_highs)]

    act_dim = env.action_space.n
    max_tiles = 2 ** 20

    actions = [item[0] for item in get_discrete_actions(n_act, act_dim)]

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
    for run in trange(nb_runs):
        tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
        q1 = QValueFunctionTiles3(tilings, actions)
        q2 = QValueFunctionTiles3(tilings, actions)

        for ep in range(nb_eps):
            o = env.reset()
            d = False
            ep_step = 0
            while not d:
                exploration = exploration_fun(ep)
                if np.random.rand() < exploration:
                    a = env.action_space.sample()
                else:
                    a = get_total_greedy_action(o, q1, q2)

                otp1, r, d, _ = env.step(a)

                returns[run, ep] += r

                qvfa, qvfb = swap_q(q1, q2)
                target = r + gamma * qvfb.value(otp1, qvfa.greedy_action(otp1))

                lr = lr_fun(ep)
                error = qvfa.update(o, a, target, lr)
                errors[run, ep] += error

                o = otp1
                ep_step += 1
            errors[run, ep] /= ep_step

    return returns, errors

def execute_grid():
    permutation_test_params = [item for item in product(lr_decay_rate_list, exploration_decay_rate_list)]
    for i, perm in enumerate(permutation_test_params):
        hparams['iteration'] = i
        returns, errors = train(*perm)

        results = dict(hparams=hparams, returns=returns, errors=errors)
        with open(os.path.join(experiment_path, f'permutation_{i}.pkl'), 'wb') as f:
            pkl.dump(results, f)


def eval_plot():
    file_paths = []
    for file in os.listdir(experiment_path):
        file_paths.append(os.path.join(experiment_path, file))

    for file in file_paths:
        


if __name__ == '__main__':
    # execute_grid()
    eval_plot()

