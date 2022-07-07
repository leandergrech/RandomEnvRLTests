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

env_name = 'MountainCar-v0'
env = gym.make(env_name)
env_lows, env_highs = env.observation_space.low, env.observation_space.high

# env_name = 'CartPole-v1'
# env = gym.make(env_name)
# env_highs = [2.4, 3, 0.2095, 3]
# env_lows = [-item for item in env_highs]

n_obs = env.observation_space.shape[0]
n_act = 1

def ranges_stats():
    n_eps = 10000
    obses = np.zeros(shape=(n_obs, n_eps))
    for i in trange(n_eps):
        o = env.reset()
        d = False
        while not d:
            for j, o_ in enumerate(o):
                obses[j, i] = o_

            a = env.action_space.sample()
            otp1, r, d, info = env.step(a)

            o=otp1
    for i, obses_ in enumerate(obses):
        print(f'Obs index {i}: Min={min(obses_):.2f}, Max={max(obses_):.2f}, Mean={np.mean(obses_):.2f}, Std={np.std(obses_)}, Median={np.median(obses_):.2f}')


# Tiling info
nb_tilings = 8
nb_bins = 2

# Training info
nb_eps = 200
nb_runs = 5

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

# Step-wise decaying lr
init_lr = 1.5e-1
lr_decay_rate = 0.9
lr_decay_every_eps = 15#int(nb_eps/10)
lr_fun = lambda ep_i: init_lr * lr_decay_rate**(ep_i//lr_decay_every_eps)
lr_str = f'Step decay LR: {init_lr}x{lr_decay_rate}^(ep_idx//{lr_decay_every_eps})'

# Step-wise decaying exploration
init_exploration = 1.0
exploration_decay_rate = 0.3
exploration_decay_every_eps = 15#int(nb_eps/10)
exploration_fun = lambda ep_i: init_exploration * exploration_decay_rate**(ep_i//exploration_decay_every_eps)
exploration_str = f'Step decay LR: {init_exploration}x{exploration_decay_rate}^(ep_idx//{exploration_decay_every_eps})'

gamma = 0.99

suptitle_suffix = f'Tilings: {nb_tilings} - Bins: {nb_bins}\n' \
                  f'Gamma: {gamma}\n' \
                  f'{lr_str}\n' \
                  f'{exploration_str}'


save_name = f'{env_name}_{nb_runs}Runs_{nb_eps}eps_{nb_tilings}Tilings_{nb_bins}Bins_1'

def train():
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
    for run in range(nb_runs):
        tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
        q1 = QValueFunctionTiles3(tilings, actions)
        q2 = QValueFunctionTiles3(tilings, actions)

        for ep in trange(nb_eps):
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

    np.save(os.path.join('results', 'saved_arrays', f'{save_name}_returns.npy'), returns)
    np.save(os.path.join('results', 'saved_arrays', f'{save_name}_errors.npy'), errors)


def eval_plot():
    file_prefix = save_name

    returns = np.load(os.path.join('results', 'saved_arrays', f'{file_prefix}_returns.npy'))
    errors = np.load(os.path.join('results', 'saved_arrays', f'{file_prefix}_errors.npy'))

    nb_runs, nb_eps = returns.shape

    gs_kw = dict(width_ratios=[3, 1],height_ratios=[2, 1])
    fig, axd = plt.subplot_mosaic([['returns', 'returns'],
                                   ['errors', 'others']],
                                  gridspec_kw=gs_kw, figsize=(10, 7))
    fig.suptitle(f'{env_name} - {nb_runs} Runs\n{suptitle_suffix}')
    ax_rets = axd['returns']
    ax_rets.set_ylabel('Total rewards')
    ax_err = axd['errors']
    ax_err.set_ylabel('Average TD error')

    ax_lr = axd['others']
    ax_lr.set_ylabel('Learning rate')
    ax_exp = ax_lr.twinx()
    ax_exp.set_ylabel('Exploration rate')

    ax_lr.plot([lr_fun(i) for i in range(nb_eps)], c='b')
    ax_exp.plot([exploration_fun(i) for i in range(nb_eps)], c='r')
    ax_lr.legend(handles=(plt.Line2D([], [], c='b'), plt.Line2D([],[],c='r')), labels=('LR', 'EXP'), loc='best')

    ax_err.axhline(0.0, c='k')
    fig.tight_layout()

    for dat, ax in zip((returns, errors), (ax_rets, ax_err)):
        quants = np.quantile(dat, [0.0, 0.25, 0.5, 0.75, 1.0], axis=0)

        averaging_window = 20
        for i, q in enumerate(quants):
            quants[i] = Series(q).rolling(averaging_window).mean().to_numpy()

        dat0, dat25, dat50, dat75, dat100 = quants
        xrange = np.arange(nb_eps)

        ax.plot(xrange, dat50, color='b', label='Median')
        ax.fill_between(xrange, dat0, dat25, color='none', edgecolor='b', hatch='//', label='Min-Max')
        ax.fill_between(xrange, dat25, dat75, color='b', alpha=0.5, label='25%-75%')
        ax.fill_between(xrange, dat75, dat100, color='none', edgecolor='b', hatch='//')

        # for datum in dat:
        #     ax.plot(xrange, Series(datum).rolling(averaging_window).mean().to_numpy())

        ax.set_xlabel('Training episodes')

        plt.legend(loc='upper left')

    plt.savefig(os.path.join('results', f'{file_prefix}.png'))

    plt.show()

if __name__ == '__main__':
    # ranges_stats()
    train()
    eval_plot()

