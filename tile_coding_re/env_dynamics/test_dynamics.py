import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random_env.envs import VREDA, RandomEnvDiscreteActions as REDA, get_discrete_actions

env = REDA(2, 2)
env.TRIM_FACTOR = 3.
# env.load_dynamics('.')
env.K_i = 1.0

def plot_some_episodes():
    obses = []
    acts = []
    nb_eps = 5
    for ep in range(nb_eps):
        o = env.reset()
        d = False
        obses.append([o])
        acts.append([[1, 1]])
        while not d:
            # a = env.get_optimal_action(o)
            a = env.action_space.sample()
            otp1, r, d, _ = env.step(a)

            acts[-1].append(a)
            obses[-1].append(otp1.copy())
            o = otp1.copy()

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.add_patch(plt.Circle((0,0), env.GOAL, edgecolor='g', facecolor='None'))
    cm = plt.cm.get_cmap('jet')(np.linspace(0, 1, nb_eps))
    for obs, act, c in zip(obses, acts, cm):
        obs = np.array(obs).T
        act = np.array(act)

        ax1.plot(obs[0], obs[1], marker='x', c=c)
        ax1.scatter(obs[0][0], obs[1][0], marker='o', c='k')
        ax2.plot(act, c=c)

    plt.show()

def get_eplen_stats():
    eplens = []
    for ep in trange(100):
        o = env.reset()
        d = False
        step = 0
        while not d:
            a = env.get_optimal_action(o)
            otp1, r, d, _ = env.step(a)
            o = otp1.copy()
            step += 1
        eplens.append(step)

    print(min(eplens))
    print(max(eplens))
    print(sum(np.where(eplens==100, np.ones_like(eplens), np.zeros_like(eplens))))

def calculating_eps_vector():
    action_set = get_discrete_actions(2)
    init_state = [0., 0.]
    fig, ax1 = plt.subplots()
    for a in action_set:
        # if a!=[1,1]:
        #     continue
        env.reset(init_state)
        next_state, *_ = env.step(a)

        obs = np.vstack([init_state, next_state]).T
        ax1.plot(obs[0], obs[1], label=a)
    ax1.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    # plot_some_episodes()
    # get_eplen_stats()
    calculating_eps_vector()