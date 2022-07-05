import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions
from tile_coding_re.heatmap_utils import make_heatmap, update_heatmap
from tile_coding_re.buffers import TrajBuffer
import gym

# n_obs, n_act = 2, 2
# env = REDA(n_obs, n_act)
env_name = 'MountainCar-v0'
env = gym.make(env_name)

n_obs = env.observation_space.shape[0]
n_act = 1

nb_tilings = 8
nb_bins = 2

ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
max_tiles = 2 ** 10

actions = [item[0] for item in get_discrete_actions(n_act)]

# Hyper parameters
init_lr = 1e-1
final_lr = 1e-3
lr_decay_eps = 1000
lr_fun = lambda ep_i: final_lr + (init_lr - final_lr) * max(0, (1 - ep_i/lr_decay_eps))
init_exploration = 1.0
final_exploration = 0.0
exploration_decay_eps = 1000
exploration_fun =  lambda ep_i: final_exploration + (init_exploration - final_exploration) * max(0, (1. - ep_i/exploration_decay_eps))
gamma = 0.99

suptitle_suffix = f'Tilings: {nb_tilings} - Bins: {nb_bins}\n' \
                  f'Gamma: {gamma}\n' \
                  f'Linear decay LR:  {init_lr:.2}->{final_lr} in {lr_decay_eps} episodes\n' \
                  f'Linear decay EPS: {init_exploration}->{final_exploration} in {exploration_decay_eps} episodes'


# Training and evaluation info
nb_eps = 2000

def swap_q():
    if np.random.rand() < 0.5:
        return qvf1, qvf2
    else:
        return qvf2, qvf1


def get_qvals(state, qvf):
    return [qvf.value(state, a_) for a_ in actions]
    

def get_total_greedy_action(state):
    global actions, qvf1, qvf2
    val1 = get_qvals(state, qvf1)
    val2 = get_qvals(state, qvf2)
    tot_val = [v1 + v2 for v1, v2 in zip(val1, val2)]
    action_idx = np.argmax(tot_val)

    return actions[action_idx]

# Training info
nb_eps = 2000
nb_runs = 10


# Start training
returns = []
for _ in range(nb_runs):
    returns.append([])
    
    tilings = Tilings(nb_tilings, nb_bins, ranges, max_tiles)
    qvf1 = QValueFunctionTiles3(tilings, actions)
    qvf2 = QValueFunctionTiles3(tilings, actions)
    
    for ep in trange(nb_eps):
        o = env.reset()
        d = False
        cum_rew = 0.0
        while not d:
            exploration = exploration_fun(ep)
            if np.random.rand() < exploration:
                a = env.action_space.sample()
            else:
                a = get_total_greedy_action(o)

            otp1, r, d, _ = env.step(a)
            
            cum_rew += r

            qvfa, qvfb = swap_q()
            target = r + gamma * qvfb.value(otp1, qvfa.greedy_action(otp1))

            lr = lr_fun(ep)
            qvfa.update(o, a, target, lr)

            o = otp1.copy()
        returns[-1].append(cum_rew)
    
rets0, rets25, rets50, rets75, rets100 = np.quantile(returns, [0.0, 0.25, 0.5, 0.75, 1.0], axis=0)
xrange = np.arange(nb_eps)

    
fig, ax = plt.subplots(figsize=(20, 10))
fig.suptitle(f'{env_name} - {nb_runs} Runs\n{suptitle_suffix}')
ax.plot(xrange, rets50, color='b', label='Median')
ax.fill_between(xrange, rets0, rets25, color='none', edgecolor='b', hatch='//')
ax.fill_between(xrange, rets25, rets75, color='b', alpha=0.5)
ax.fill_between(xrange, rets75, rets100, color='none', edgecolor='b', hatch='//')

ax.set_xlabel('Training episodes')
ax.set_ylabel('Cumulative rewards')

plt.savefig(os.path.join('results', f'{env_name}_{nb_runs}Runs_{nb_tilings}Tilings_{nb_bins}Bins.png'))

plt.show()


