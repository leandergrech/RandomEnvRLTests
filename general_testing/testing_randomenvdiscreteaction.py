import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from tqdm import trange
from random_env.envs import RandomEnvDiscreteActions as REDA, VREDA, get_discrete_actions, REDAClip, REDAClipCont
from utils.training_utils import InitSolvableState
import yaml


def quick_testing_randomenvdiscreteactions():
    n_obs = 5
    n_act = 5

    env = REDA(n_obs, n_act)

    record = defaultdict(list)

    d = False
    o = env.reset()
    record['o'].append(o)
    while not d:
        # a = env.action_space.sample()
        a = env.get_optimal_action(o)
        o, r, d, _ = env.step(a)
        aa = env.get_actual_actions().copy()

        record['a'].append(a)
        record['aa'].append(aa)
        record['o'].append(o.copy())
        record['r'].append(r)

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.set_title('Observations')
    ax1.plot(record['o'], c='b')
    # ax2.bar(np.repeat([range(len(record['a']))], n_act, axis=0), record['a'], c='r')
    ax2.set_title('Multi-Discrete Actions')
    ax2.plot(record['a'], c='r', ls='solid')
    ax3.set_title('Dynamics Actions')
    ax3.plot(record['aa'], c='r', ls='dashed')
    fig.tight_layout()


def testing_vreda_velocities():
    env = VREDA(2, 2)
    vels = []
    for ep in trange(1000):
        o = env.reset()
        d = False
        while not d:
            a = env.action_space.sample()
            otp1, r, d, _ = env.step(a)
            
            vels.append(otp1[2])
    print('Random actions')
    print(f'Average velocity = {np.mean(vels)}')
    print(f'Min velocity = {np.min(vels)}')
    print(f'Max velocity = {np.max(vels)}')

    vels = []
    for ep in trange(1000):
        o = env.reset()
        d = False
        while not d:
            a = env.get_optimal_action(o)
            otp1, r, d, _ = env.step(a)

            vels.append(otp1[2])
    print('Optimal actions')
    print(f'Average velocity = {np.mean(vels)}')
    print(f'Min velocity = {np.min(vels)}')
    print(f'Max velocity = {np.max(vels)}')


def testing_vreda_eplens():
    env = VREDA(2, 2)
    nb_eps = 1000
    for trim_factor in (1., 5., 10., 15., 20., 25., 30., 35., 40.):
        env.TRIM_FACTOR = trim_factor
        eplens = []
        for ep in trange(nb_eps):
            o = env.reset()
            d = False
            step = 0
            while not d:
                a = np.array(env.get_optimal_action(o), dtype=int)
                otp1, r, d, _ = env.step(a)

                step += 1
            eplens.append(step)
        print('Random actions')
        print(f'Average ep_len = {np.mean(eplens)}')
        print(f'Min ep_len = {np.min(eplens)}')
        print(f'Max ep_len = {np.max(eplens)}')
        eplens = np.array(eplens)

        non_maxed_eps = np.where(eplens==env.EPISODE_LENGTH_LIMIT, np.zeros_like(eplens), eplens)
        maxed_out = sum(np.where(eplens==env.EPISODE_LENGTH_LIMIT, np.ones_like(eplens), np.zeros_like(eplens)))
        mean_ep_len = sum(non_maxed_eps)/(nb_eps - maxed_out)
        print(f'Maxed out eps = {maxed_out}')

        fig = plt.figure(figsize=(4, 3))
        fig.suptitle(f'{repr(env)}\nTRIM_FACTOR={trim_factor:.1f}\n{(maxed_out/nb_eps)*100.0:.2f}% maxed episodes\nAverage ep_len={mean_ep_len:.2f}', size=10)
        plt.axhline(mean_ep_len, c='k')
        plt.scatter(range(len(eplens)), eplens, marker='x')
        plt.xlabel('Episodes')
        plt.ylabel('Episode lengths')
        plt.minorticks_on()
        plt.grid(which='both', axis='y')
        fig.tight_layout()
        plt.savefig(os.path.join('vreda_eplens', f'{trim_factor:.2f}.png'))
    plt.show()


def vreda_diagnostic_plots():
    env = VREDA(2, 2)
    env.TRIM_FACTOR = 4.
    # o = env.reset([-.1,0.1])
    o = env.reset()
    d = False
    obses = [o[:2]]
    acts = [[1,1]]
    vels = [0.0]
    cacts = [[0.,0.]]

    # destiny = [[0,2], [0,2], [1,1], [1,1], [2,1], [2,1], [2,0], [2,0], [2,1], [0,1], [0,1], [1,1], [1,1], [1,1]]
    # for a in destiny:
    step = 0
    while step < 10000:
        # a = env.get_optimal_action(o)
        a = env.action_space.sample()
        otp1, r, d, _ = env.step(a)

        cacts.append(env.cum_action.copy())
        obses.append(otp1[:2])
        acts.append(a)
        vels.append(otp1[2])
        step += 1

    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(5,8), gridspec_kw=dict(height_ratios=[5,3,1]))
    fig.suptitle(repr(env))
    # ax1.add_patch(plt.Circle([0, 0], env.GOAL, facecolor='None', edgecolor='g', ls='dashed'))
    obses = np.array(obses).T
    ax1.add_patch(plt.Circle((0,0), env.GOAL, facecolor='None', edgecolor='g'))
    ax1.plot(obses[0], obses[1], marker='x')
    ax1.scatter(obses[0][0], obses[1][0], marker='o', c='k', label='Start')
    ax1.set_xlabel('State[0]')
    ax1.set_ylabel('State[1]')
    lim = 1.0
    ax1.set_ylim((-lim,lim))
    ax1.set_xlim((-lim,lim))
    ax1.legend(loc='best')

    ax3.plot(vels, c='k', marker='x')
    ax3.set_ylabel('Velocity')
    ax3.set_xlabel('Steps')

    acts = np.array(acts).T
    act_range = range(len(acts[0]))
    ax2.step(act_range, acts[0], c='tab:brown', marker='s')
    ax2.step(act_range, acts[1], c='tab:brown', ls='--', marker='s')
    ax2.set_ylim((-0.1, 2.1))
    ax2.set_ylabel('Actions')
    ax2.set_xlabel('Steps')
    ax2.yaxis.label.set_color('tab:brown')
    ax22 = ax2.twinx()
    cacts = np.array(cacts).T
    ax22.axhline(0.0, c='k', ls=':', alpha=0.5, lw=1)
    ax22.plot(cacts[0], c='r', marker='o')
    ax22.plot(cacts[1], c='r', ls='--', marker='o')
    ax22.set_ylabel('Actual action')
    ax22.yaxis.label.set_color('r')
    cact_absmax = np.max(np.abs(cacts))*1.2
    ax22.set_ylim((-cact_absmax, cact_absmax))

    fig.tight_layout()

    plt.show()


def testing_reda_trims():
    n_obs, n_act = 2, 2
    n_envs = 5

    actions = get_discrete_actions(n_act, 3)
    n_actions = len(actions)

    init_state = np.zeros(n_obs)

    cmap = mpl.cm.get_cmap('jet')
    fig, ax = plt.subplots()
    # ax.set_facecolor('grey')
    for i, c in enumerate(np.linspace(0, 1, n_envs)):
        env = REDA(n_obs, n_act, estimate_scaling=False)
        env.ACTION_EPS = 0.1
        if i == 0:
            thresh = ax.add_patch(plt.Circle((0.,0.), env.GOAL, facecolor='None', edgecolor='g', ls='--'))
            thresh2 = ax.add_patch(plt.Circle((0.,0.), 2*env.GOAL, facecolor='None', edgecolor='g', ls=':'))
        for j, a in enumerate(actions):
            if a == np.ones(n_act).tolist():
                continue
            env.reset(init_state.copy())
            otp1, *_ = env.step(a)
            label = f'Env {i}' if j == 0 else None
            text_pos = otp1 + 0.2*otp1
            text_align = 'top' if otp1[1] > 0 else 'bottom'
            ax.text(*text_pos, f'{a}', size=8, c=cmap(c), verticalalignment=text_align)
            ax.plot(*np.vstack([init_state, otp1]).T, c=cmap(c), marker='x', label=label)
    h, l = ax.get_legend_handles_labels()
    plt.legend(handles=h+[thresh, thresh2], labels=l+['Threshold', 'Threshold x2'], loc='best')

    ax.set_title('REDA unit trims fixed by random linear dynamics')
    ax.set_xlabel('State dimension 0')
    ax.set_ylabel('State dimension 1')

    fig.tight_layout()

    plt.show()


def testing_reda_optimal_policy_2d():
    n_obs, n_act = 3, 2
    env = REDA(n_obs, n_act)
    nb_eps = 30

    init_func = InitSolvableState(env)

    cmap = mpl.cm.get_cmap('hsv')
    fig, ax = plt.subplots()

    ax.add_patch(plt.Circle((0, 0), env.GOAL, edgecolor='g', facecolor='None', ls='--', label='Threshold', zorder=20, lw=1.5))
    for ep, c in enumerate(np.linspace(0, 1, nb_eps)):
        o = env.reset(init_func())
        d = False
        obses = [o.copy()]
        label = 'Initial state' if ep == 0 else None
        ax.scatter(o[0], o[1], marker='o', c='k', label=label, zorder=15)
        while not d:
            a = env.get_optimal_action(o)
            otp1, _, d, _ = env.step(a)
            obses.append(otp1.copy())
            o = otp1
        obses = np.array(obses).T
        label = 'Terminal state' if ep == 0 else None
        ax.scatter(o[0], o[1], marker='*', c='k', label=label, zorder=15)
        ax.plot(obses[0], obses[1], c=cmap(c), marker='x', zorder=10)
    ax.set_title(f'{repr(env)}\n'
                 f'Optimal policy derived from linear dynamics')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()


def testing_reda_optimal_policy_bars():
    # from random_env.envs import RandomEnv
    from random_env.envs import IREDA
    n_obs, n_act = 10,10
    # env = RandomEnv(n_obs, n_act)
    # env.load_dynamics('../common_envs')
    env = IREDA(n_obs, n_act)

    # init_func = env.reset
    init_func = InitSolvableState(env)

    fig, axs = plt.subplots(2)

    o = env.reset(init_func())

    obs_bars = axs[0].bar(range(n_obs), o, facecolor='b')
    axs[0].axhline(-env.GOAL, color='g', ls='--')
    axs[0].axhline(env.GOAL, color='g', ls='--')
    act_bars = axs[1].bar(range(n_act), np.zeros(n_act), facecolor='r')
    axs[0].set_ylim((-1, 1))
    axs[1].set_ylim((-0.15, 0.15))
    plt.ion()
    plt.pause(1)

    d = False
    step = 0
    while not d:
        a = env.get_optimal_action(o)
        # a /= np.max([1, np.max(np.abs(a))])
        # a *= 0.3
        otp1, _, d, _ = env.step(a)

        o = otp1
        step += 1

        fig.suptitle(step)
        for bars, data in zip((obs_bars, act_bars), (o, np.subtract(a, 1)*env.ACTION_EPS)):
            for bar, datum in zip(bars, data):
                bar.set_height(datum)
        plt.pause(.3)
    plt.ioff()

    fig.suptitle(f'{repr(env)}\n'
                 f'Optimal policy derived from linear dynamics\n'
                 f'Steps taken = {step}')
    fig.tight_layout()
    plt.show()


def testing_redaclip_yaml():
    from random_env.envs import RunningStats
    env = REDAClip(2, 2, 1.0)
    test_file = 'save.yml'
    print('Before save:')
    print(env.rm, env.pi, env.trim_stats)
    with open(test_file, 'w') as f:
        yaml.dump({'env': env}, f, default_flow_style=False)

    with open(test_file, 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
        loaded_env = d['env']
    print('After save')
    print(loaded_env.rm, loaded_env.pi, loaded_env.trim_stats)

def why_get_slower_during_training():
    env = REDAClipCont(7, 7, 1.0)
    env.reset()
    for T in trange(500000):
        env.step(env.action_space.sample())

if __name__ == '__main__':
    # testing_vreda_velocities()
    # testing_vreda_eplens()
    # vreda_diagnostic_plots()
    # testing_reda_trims()
    # testing_reda_optimal_policy_bars()
    # testing_redaclip_yaml()
    why_get_slower_during_training()