import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from random_env.envs import RandomEnv
from utils.plotting_utils import y_grid_on

# experiment_name = 'PPO_092722_173143'
# experiment_name = 'PPO_092722_195643'
experiment_name = 'PPO_092722_203511'
sub_experiment_name = 'seed-123'
training_step = 170000

model_path = os.path.join(experiment_name, sub_experiment_name, 'saves', f'rl_model_{training_step}_steps.zip')
env_path = experiment_name
model = PPO.load(model_path)

env = RandomEnv.load_from_dir(env_path)
n_obs, n_act = env.obs_dimension, env.act_dimension

init_func = env.reset
# init_func = InitSolvableState(env)


def plot_episodes():
    nrows = 2
    ncols = 4
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
            ax.set_xlabel('Step', size=12)

    fig.suptitle(f'Environment: {repr(env)}\n'
                 f'Experiment:  {experiment_name}\n'
                 f'Sub-exp:     {sub_experiment_name}\n'
                 f'At step:     {training_step}')
    fig.tight_layout()
    fig.savefig(os.path.join(experiment_name, f'{sub_experiment_name}_{training_step}_step.png'))
    plt.show()


def plot_episode_ion():
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
    plot_episodes()