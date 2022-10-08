import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from random_env.envs import RandomEnv
from re_qfb_size.other_random_env import NoisyClipRE, NoisyRE
from utils.plotting_utils import y_grid_on


def load_agent(exp_name, sub_exp_name, train_step):
    if 'PPO' in exp_name:
        agent_type = PPO
    elif 'TRPO' in exp_name:
        agent_type = TRPO

    model_path = os.path.join(exp_name, sub_exp_name, 'saves', f'rl_model_{train_step}_steps.zip')
    model = agent_type.load(model_path)

    return model


class OptimalAgent:
    def __init__(self, env):
        self.env = env
    def predict(self, state, deterministic=None):
        return [self.env.get_optimal_action(state)]


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def predict(self, state, deterministic=None):
        return [self.env.action_space.sample()]


def plot_episodes(exp_name, sub_exp_name, train_step, env, nrows=2, ncols=4):
    if isinstance(train_step, int):
        model = load_agent(exp_name, sub_exp_name, train_step)
    elif 'optimal' in train_step:
        model = OptimalAgent(env)
    elif 'random' in train_step:
        model = RandomAgent(env)

    n_obs, n_act = env.obs_dimension, env.act_dimension
    init_func = env.reset

    fig, axs = plt.subplots(nrows * 2, ncols, figsize=(30, 15))
    # axs = np.ravel(axs)
    nb_eps = nrows * ncols

    for i in range(nb_eps):
        obses = []
        acts = []
        d = False
        o = env.reset(init_func())
        obses.append(o.copy())
        acts.append(np.zeros(n_act))
        step = 1
        while not d:
            a = model.predict(o, deterministic=True)[0]
            otp1, _, d, _ = env.step(a)
            o = otp1.copy()

            obses.append(o)
            acts.append(a)

            step += 1
        obses = np.array(obses)
        acts = np.array(acts)

        ax_obs = axs[(i // ncols) * 2, i % ncols]
        ax_act = axs[(i // ncols) * 2 + 1, i % ncols]

        ax_obs.set_title(f'Ep {i + 1}', size=15)
        ax_obs.axhline(-env.GOAL, c='g', ls='--', lw=2)
        ax_obs.axhline(env.GOAL, c='g', ls='--', lw=2)
        ax_obs.plot(obses, c='b')
        ax_act.axhline(0.0, c='k', ls='-.', lw=2)
        ax_act.plot(acts, c='r')

        for ax, ylab in zip((ax_obs, ax_act), ('States', 'Actions')):
            y_grid_on(ax)
            ax.set_xticks(np.arange(step))
            ax.set_ylabel(ylab, size=12)

        ax_obs.get_shared_x_axes().join(ax_obs, ax_act)
        ax_act.set_xlabel('Step', size=12)

    fig.suptitle(f'Environment: {repr(env)}\n'
                 f'Experiment:  {exp_name}\n'
                 f'Sub-exp:     {sub_exp_name}\n'
                 f'At step:     {train_step}')
    fig.tight_layout()
    fig.savefig(os.path.join(exp_name, f'{sub_exp_name}_{train_step}_step.png'))
    plt.show()


def plot_episode_ion(exp_name, sub_exp_name, train_step, env):
    model = load_agent(exp_name, sub_exp_name, train_step)
    n_obs, n_act = env.obs_dimension, env.act_dimension
    init_func = env.reset

    fig, axs = plt.subplots(2, figsize=(30, 15))

    o = env.reset(init_func())

    obs_bars = axs[0].bar(range(n_obs), o, facecolor='b')
    axs[0].axhline(-env.GOAL, color='g', ls='--')
    axs[0].axhline(env.GOAL, color='g', ls='--')
    act_bars = axs[1].bar(range(n_act), np.zeros(n_act), facecolor='r')
    for ax in axs:
        ax.set_ylim((-1, 1))
    plt.ion()
    plt.pause(1)

    d = False
    step = 0
    while not d:
        # a = env.get_optimal_action(o)
        # a /= np.max([1, np.max(np.abs(a))])
        # a *= 0.1
        a = model.predict(o, deterministic=True)[0]
        otp1, _, d, _ = env.step(a)

        o = otp1
        step += 1

        fig.suptitle(step)
        for bars, data in zip((obs_bars, act_bars), (o, a)):
            for bar, datum in zip(bars, data):
                bar.set_height(datum)
        plt.pause(1)
    plt.ioff()

    fig.suptitle(f'{repr(env)}\n'
                 f'Optimal policy derived from linear dynamics\n'
                 f'Steps taken = {step}')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    experiment_name = 'PPO_NoisyClipRE_100822_203732'
    # experiment_name = 'TRPO_092922_174343'
    sub_experiment_name = 'seed-780'
    training_step = 290000#'optimal'

    env = NoisyClipRE.load_from_dir(experiment_name)
    # env.action_noise = 0.08
    plot_episodes(experiment_name, sub_experiment_name, training_step, env, nrows=2, ncols=2)
