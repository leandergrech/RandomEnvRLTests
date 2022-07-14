import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
from random_env.envs import RandomEnvDiscreteActions, VREDA


def quick_testing_randomenvdiscreteactions():
    n_obs = 5
    n_act = 5

    env = RandomEnvDiscreteActions(n_obs, n_act)

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
# plt.show()

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
    o = env.reset([-.1,0.1])
    d = False
    obses = [o[:2]]
    acts = [[1,1]]
    vels = [0.0]
    cacts = [[0.,0.]]

    destiny = [[0,2], [0,2], [1,1], [1,1], [2,1], [2,1], [2,0], [2,0], [2,1], [0,1], [0,1], [1,1], [1,1], [1,1]]

    for a in destiny:
        # a = env.get_optimal_action(o)
        otp1, r, d, _ = env.step(a)

        cacts.append(env.cum_action.copy())
        obses.append(otp1[:2])
        acts.append(a)
        vels.append(otp1[2])

    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(5,8), gridspec_kw=dict(height_ratios=[5,3,1]))
    fig.suptitle(repr(env))
    # ax1.add_patch(plt.Circle([0, 0], env.GOAL, facecolor='None', edgecolor='g', ls='dashed'))
    obses = np.array(obses).T
    ax1.plot(obses[0], obses[1], marker='x')
    ax1.scatter(obses[0][0], obses[1][0], marker='o', c='k', label='Start')
    ax1.set_xlabel('State[0]')
    ax1.set_ylabel('State[1]')
    lim = 0.15
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

if __name__ == '__main__':
    # testing_vreda_velocities()
    testing_vreda_eplens()
    # vreda_diagnostic_plots()