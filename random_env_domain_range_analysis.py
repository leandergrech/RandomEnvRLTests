import os
from datetime import datetime as dt
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import random
from random_env.envs import RandomEnv

N_POINTS = 1000
FIG_SIZE = (10, 7)
MAX_EP_LEN = RandomEnv.EPISODE_LENGTH_LIMIT
RANDOM_SEEDS = (123, 234, 345, 456, 567)#, 345, 456, 567)
N_ENVS = len(RANDOM_SEEDS)
GAMMA = 0.99
gammas = [np.power(GAMMA, i) for i in range(MAX_EP_LEN)]


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calculate_returns(rewards):
    global gammas

    n = len(rewards)
    returns = np.zeros_like(rewards)

    for i in range(n):
        returns[i] = sum(np.multiply(gammas[:n-i], rewards[i:]))

    return returns

# Environment size loop
# for n_obs, n_act in zip((5, 10, 50, 50, 50, 100, 100, 100), (5, 10, 50, 75, 100, 100, 150, 200)):
for n_obs, n_act in zip((3,), (2,)):
    par_dir = 'random_env_domain_range_analysis'
    session = f'session_{dt.strftime(dt.now(), "%m%d%y_%H%M%S")}'
    session_dir = os.path.join(par_dir, session)

    # Environment creation with diff random seeds loop
    old_rm = None
    for random_seed in RANDOM_SEEDS:
        model_name = f'RE{n_obs}x{n_act}_{random_seed}'
        model_dir = os.path.join(session_dir, model_name)
        make_path(model_dir)

        env = RandomEnv(n_obs, n_act, estimate_scaling=True, seed=random_seed)
        env.save_dynamics(model_dir)

        if old_rm is not None:
            assert not np.equal(old_rm, env.rm).all(), 'what the shit?'
        else:
            old_rm = np.copy(env.rm)

    # Create initial states with which to evaluate
    np.random.seed() # dynamics created with seed, now lose control

    sample_actions = np.random.uniform(-1, 1, size=N_POINTS * n_act).reshape((N_POINTS, n_act))

    # Environment probing loop
    for i, model_name in enumerate(os.listdir(session_dir)):
        # Load models and finish a test one env at a time
        model_dir = os.path.join(session_dir, model_name)
        env = RandomEnv.load_from_dir(model_dir)

        sample_trims = np.zeros(shape=(N_POINTS, n_obs))
        for j in range(N_POINTS):
            sample_trims[j] = env.rm.dot(sample_actions[j])

        # Plot the shit
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*sample_trims.T, marker='o', color='b')
        ax.set_xlabel('State 0')
        ax.set_ylabel('State 1')
        ax.set_zlabel('State 2')

        results_dir = os.path.join(par_dir, 'results')
        make_path(results_dir)
        save_path = os.path.join(results_dir, f'{session}_{model_name}')
        fig.suptitle(f'{model_name}\nRandom action trims')
        plt.show()
        fig.savefig(save_path)
