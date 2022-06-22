import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from tile_coding_re.mc_method.tile_coding import QValueFunction
from tile_coding_re.mc_method.utils import TrajSimple, TrajBuffer
from tile_coding_re.mc_method.constants import *

from random_env.envs import RandomEnvDiscreteActions

load_path = os.path.join(par_dir, 'saves')


def load_env():
    env = RandomEnvDiscreteActions(2, 2)
    env.load_dynamics(load_path)
    return env


def load_qvf_for_ep(ep):
    qvf_path = os.path.join(load_path, f'{ep}ep')
    qvf = QValueFunction.load(qvf_path)
    return qvf


def play_episode(env, qvf, buffer=None):
    all_actions = qvf.actions

    if buffer is None:
        buffer = TrajSimple()
    else:
        buffer.reset()

    o = env.reset()
    d = False
    while not d:
        vals = [qvf.value(o, a_) for a_ in all_actions]
        a = all_actions[np.argmax(vals)]
        otp1, r, d, _ = env.step(a)

        if isinstance(buffer, TrajSimple):
            buffer.add(r)
        elif isinstance(buffer, TrajBuffer):
            buffer.add(o, a, r)
        o = otp1
    return buffer


def evaluation_episodes():
    start_ep = EVAL_EVERY
    end_ep = NB_TRAINING_EPS
    ep_step = EVAL_EVERY

    env = load_env()

    results_path = os.path.join(par_dir, 'evaluation-episodes')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    buffer = TrajBuffer()
    for ep in np.arange(start_ep, end_ep + ep_step, ep_step):
        print(f'Episode #{ep}/{end_ep}')
        qvf = load_qvf_for_ep(ep)

        for i in range(3):
            buffer = play_episode(env, qvf, buffer)

            fig, (ax1, ax2, ax3)  = plt.subplots(3)
            ax1.plot(buffer.o, color='b')
            ax1.set_title('States')
            ax2.plot(buffer.a, color='r')
            ax2.set_title('Actions')
            ax3.plot(buffer.r, color='g')
            ax3.set_title('Rewards')
            fig.tight_layout()
            plt.savefig(os.path.join(results_path, f'{ep}-training-eps_{i}.png'))
            plt.close(fig)


def episode_length_statistics():
    start_ep = EVAL_EVERY
    end_ep = NB_TRAINING_EPS
    ep_step = EVAL_EVERY

    env = load_env()

    results_path = os.path.join(par_dir, 'episode-length-stats')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    nb_eval_eps = 20

    ep_range = np.arange(start_ep, end_ep + ep_step, ep_step)
    ep_lens = np.zeros(shape=(ep_range.size, nb_eval_eps))

    for i, ep in enumerate(ep_range):
        print(f'Episode #{ep}/{end_ep}')
        qvf = load_qvf_for_ep(ep)

        buffer = TrajSimple()
        for j in trange(nb_eval_eps):
            buffer = play_episode(env, qvf, buffer)
            ep_lens[i, j] = len(buffer)

    el_min, el_25, el_med, el_75, el_max = np.quantile(ep_lens, [0.0, 0.25, 0.5, 0.75, 1.0], axis=1)
    fig, ax = plt.subplots()
    tok = par_dir.split('_')
    fig.suptitle(' '.join(tok[:4]) + '\n' + ' '.join(tok[4:]))
    ax.plot(ep_range, el_med, color='b', label='Median')
    ax.fill_between(ep_range, el_min, el_25, color='none', edgecolor='b', hatch='//')
    ax.fill_between(ep_range, el_25, el_75, color='b', alpha=0.5)
    ax.fill_between(ep_range, el_75, el_max, color='none', edgecolor='b', hatch='//')

    ax.set_xlabel('Training episodes')
    ax.set_ylabel('Episode length')
    plt.legend(loc='best')

    plt.savefig(os.path.join(results_path, f'ep_lens_{nb_eval_eps}eval-eps.png'))
    plt.close(fig)


if __name__ == '__main__':
    episode_length_statistics()
    evaluation_episodes()
