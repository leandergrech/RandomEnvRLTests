import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
from random_env.envs import RandomEnvDiscreteActions


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


def testing_reda_velocities():
    env = RandomEnvDiscreteActions(2, 2)
    get_vel = lambda state, next_state: np.sqrt(np.sum(np.square(np.subtract(next_state, state))))
    vels = []
    
    # fig, (ax1, ax2) = plt.subplots(2)
    # o_line, = ax1.plot([], [], marker='x')
    # ax1.set_ylim((-1,1))
    # ax1.set_xlim((-1,1))
    # v_line, = ax2.plot([],[])
    # ax2.set_xlim((0, env.EPISODE_LENGTH_LIMIT))
    # plt.ion()
    
    for ep in trange(1000):
        o = env.reset()
        d = False
        v = []
        obses = [[], []]
        while not d:
            a = env.action_space.sample()
            # a = env.get_optimal_action(o)
            otp1, r, d, _ = env.step(a)
            
            vel = get_vel(otp1, o)
            
            v.append(vel)
            obses[0].append(otp1[0])
            obses[1].append(otp1[1])
            
            vels.append(vel)
            
            o = otp1.copy()
        # o_line.set_data(obses[0], obses[1])
        # ax1.set_xlim((min(obses[0]), max(obses[0])))
        # ax1.set_ylim((min(obses[1]), max(obses[1])))
        # v_line.set_data(range(len(v)), v)
        # ax2.set_ylim((min(v), max(v)))
        # plt.pause(0.1)
        # input()
        
        
    print(f'Average velocity = {np.mean(vels)}')
    print(f'Min velocity = {np.min(vels)}')
    print(f'Max velocity = {np.max(vels)}')
# plt.show()


if __name__ == '__main__':
    # for _ in range(5):
    #     quick_testing_randomenvdiscreteactions()
    # plt.show()
    testing_reda_velocities()
