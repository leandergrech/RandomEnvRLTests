import os
from datetime import datetime as dt
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import random
from random_env.envs import RandomEnv

N_INITIAL_STATES = 5
FIG_SIZE = (7, 4)
RANDOM_SEEDS = (123, 234, 345)  # , 345, 456, 567)
N_ENVS = len(RANDOM_SEEDS)
GAMMA = 0.99
gammas = [np.power(GAMMA, i) for i in range(RandomEnv.EPISODE_LENGTH_LIMIT)]


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calculate_returns(rewards):
    global gammas

    n = len(rewards)
    returns = np.zeros_like(rewards)

    for i in range(n):
        returns[i] = sum(np.multiply(gammas[:n - i], rewards[i:]))

    return returns


# for n_obs, n_act in zip((5, 10, 50, 50, 50, 100, 100, 100), (5, 10, 50, 75, 100, 100, 150, 200)):
for n_obs, n_act in zip((10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10), (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)):
    par_dir = 'random_env_characterisation'
    session = f'session_{dt.strftime(dt.now(), "%m%d%y_%H%M%S")}'
    session_dir = os.path.join(par_dir, session)

    # Create models
    for random_seed in RANDOM_SEEDS:
        model_name = f'RE{n_obs}x{n_act}_{random_seed}'
        model_dir = os.path.join(session_dir, model_name)
        make_path(model_dir)

        env = RandomEnv(n_obs, n_act, estimate_scaling=True, seed=random_seed)
        env.save_dynamics(model_dir)

    # Create initial states with which to evaluate
    np.random.seed()  # dynamics created with seed, now lose control
    initial_states = np.random.normal(0, 1, size=N_INITIAL_STATES * n_obs).reshape((N_INITIAL_STATES, n_obs))

    # Create figure
    fig, axs = plt.subplots(N_ENVS, N_INITIAL_STATES, figsize=(10, 7), sharex=True, sharey=True)
    axs2 = []

    # plt.show(block=False)
    # Access and probe models
    for i, init_state in enumerate(initial_states):
        # print(f'-> Initial state #{i}')
        for j, model_name in enumerate(os.listdir(session_dir)):
            model_dir = os.path.join(session_dir, model_name)
            env = RandomEnv.load_from_dir(model_dir)

            # Initialise trajectory with same initial state
            o = np.copy(init_state)
            env.reset(o)
            d = False
            k = 0

            obses = []
            acts = []
            rews = []
            while not d:
                a = env.get_optimal_action(o)
                o, r, d, _ = env.step(a)
                k += 1

                obses.append(np.copy(o))
                acts.append(np.copy(a))
                rews.append(r)

            ax = axs[j, i]

            if j == 0:
                ax.set_title(f'Initial state #{i}')
            if i == 0:
                ax.set_ylabel(f'Env #{j}')

            rets = calculate_returns(rews)

            # s_line and a_line only used for legend
            ax.plot(np.array(obses), color='b', lw=0.25)
            ax.plot(np.array(acts), color='r', lw=0.25)
            ax.axhline(-env.GOAL, color='g', ls='dashed')
            ax.axhline(env.GOAL, color='g', ls='dashed')
            ax.set_ylim((-1, 1))

            ax2 = ax.twinx()
            ax2.plot(rets, color='k', label='Returns')
            ax2.set_yscale('symlog')
            axs2.append(ax2)

    for ax2 in axs2[1:]:
        axs2[0].get_shared_y_axes().join(axs2[0], ax2)

    lines = [plt.Line2D([], [], color='b'),
             plt.Line2D([], [], color='r'),
             plt.Line2D([], [], color='k'),
             plt.Line2D([], [], color='g', ls='dashed')]
    labels = ['States', 'Action', 'Discounted Returns', 'Goal']

    fig.legend(handles=lines, labels=labels, loc='lower left', ncol=5)
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.12, top=0.85, wspace=0.255, hspace=0.475)
    fig.suptitle(
        f'{session}\n{repr(env)} => Action_scale={env.ACTION_SCALE}, Trim_factor={env.TRIM_FACTOR}, K_p={env.K_p}, '
        f'Reward_scale={env.REWARD_SCALE}, Gamma={GAMMA}')

    results_dir = os.path.join(par_dir, 'results')
    make_path(results_dir)
    save_path = os.path.join(results_dir, f'{session}_{repr(env)}')
    fig.savefig(save_path)
# plt.show()
