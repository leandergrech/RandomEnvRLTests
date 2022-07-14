import os
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from tile_coding_re.indirect_approach.tile_coding import QValueFunction2, get_tilings_from_env
from tile_coding_re.utils import TrajSimple, TrajBuffer
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions

METHOD = 'mc_method'
# METHOD = 'sarsa'

if METHOD == 'mc_method':
    from tile_coding_re.indirect_approach.mc_method.constants import par_dir
elif METHOD == 'sarsa':
    from tile_coding_re.indirect_approach.sarsa import par_dir


def get_or_ask_par_dir():
    global par_dir
    avails = []
    for file in os.listdir(METHOD):
        if par_dir in file:
            avails.append(file)
    if not avails:
        print(os.path.abspath('..'))
        print(os.listdir('mc_method'))
        raise FileNotFoundError(par_dir)
    elif len(avails) > 1:
        for i, file in enumerate(avails):
            print(f'{i}:\t{file}')
        idx = int(input('Choose which one to evaluate: '))
        par_dir = avails[idx]

    return os.path.join(METHOD, par_dir)


par_dir = get_or_ask_par_dir()
load_path = os.path.join(par_dir, 'saves')

print(f'Loading Q-tables from: {load_path}')


def load_env():
    env = RandomEnvDiscreteActions(2, 2)
    env.load_dynamics(load_path)
    return env


def load_qvf_for_ep(ep):
    qvf_path = os.path.join(load_path, f'{ep}ep')
    qvf = QValueFunction2.load(qvf_path)
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
        a = qvf.greedy_action(o)
        otp1, r, d, _ = env.step(a)

        if isinstance(buffer, TrajSimple):
            buffer.add(r)
        elif isinstance(buffer, TrajBuffer):
            buffer.add(o, a, r)
        o = otp1
    return buffer


def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return (func(*args))

    return wrapper


env = None
qvf = None


def get_an_ep_len(i):
    o = env.reset()
    d = False
    i = 0
    while not d:
        a = qvf.greedy_action(o)
        _, _, d, _ = env.step(a)
        i += 1
    return i


def evaluation_episodes(start_ep=SAVE_EVERY, end_ep=NB_TRAINING_EPS):
    print('CREATE EVALUATION EPISODES')
    # start_ep = NB_TRAINING_EPS - SAVE_EVERY
    # end_ep = NB_TRAINING_EPS
    ep_step = SAVE_EVERY

    env = load_env()

    results_path = os.path.join(par_dir, 'evaluation-episodes')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    buffer = TrajBuffer()
    tilings = get_tilings_from_env(env, NB_TILINGS, NB_BINS)
    for ep in np.arange(start_ep, end_ep + ep_step, ep_step):
        print(f'\rEpisode #{ep}/{end_ep}', end='')
        qvf = load_qvf_for_ep(ep)

        for i in range(3):
            buffer = play_episode(env, qvf, buffer)

            colors = iter(plt.cm.jet(np.linspace(0, 1, NB_TILINGS)))
            line_styles = cycle(('-', '--', ':', '-.'))

            fig, (ax1, ax2, ax3) = plt.subplots(3)
            for tiling, c, ls in zip(tilings, colors, line_styles):
                for t in tiling[0]:  # because of symmetrical offsetting
                    ax1.axhline(t, color=c, ls=ls, alpha=0.5, lw=0.5)
            ax1.plot(buffer.o, color='b')
            ax1.set_title('States')
            ax2.plot(buffer.a, color='r')
            ax2.set_title('Actions')
            ax3.plot(buffer.r, color='g')
            ax3.set_title('Rewards')
            fig.tight_layout()
            plt.savefig(os.path.join(results_path, f'{ep}-training-eps_{i}.png'))
            plt.close(fig)
    print('')


def episode_length_statistics():
    global env, qvf
    print('CALCULATING EPISODE LENGTH STATISTICS')
    start_ep = SAVE_EVERY
    end_ep = NB_TRAINING_EPS
    ep_step = SAVE_EVERY

    env = load_env()

    results_path = os.path.join(par_dir, 'episode-length-stats')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    nb_eval_eps = 20
    save_name = f'ep_lens_{nb_eval_eps}eval-eps'

    ep_range = np.arange(start_ep, end_ep + ep_step, ep_step)
    ep_lens = np.zeros(shape=(ep_range.size, nb_eval_eps))

    for i, ep in enumerate(ep_range):
        print(f'\rEpisode #{ep:4d}/{end_ep}', end='')
        qvf = load_qvf_for_ep(ep)

        # nproc = 4
        # p = Pool(nproc)
        # ep_lens[i] = p.map(get_an_ep_len, np.arange(nb_eval_eps))
        buffer = TrajSimple()
        for j in range(nb_eval_eps):
            buffer = play_episode(env, qvf, buffer)
            ep_lens[i, j] = len(buffer)

    np.save(os.path.join(results_path, f'{save_name}.npy'), ep_lens)

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

    plt.savefig(os.path.join(results_path, f'{save_name}.png'))
    plt.show()
    plt.close(fig)
    print('')


def track_state_values():
    print('TRACK VALUES OF RANDOM STATES THROUGHOUT TRAINING')
    start_ep = SAVE_EVERY
    end_ep = NB_TRAINING_EPS
    ep_step = SAVE_EVERY

    env = load_env()
    n_act = env.act_dimension
    all_actions = get_discrete_actions(n_act)

    results_path = os.path.join(par_dir, 'tracking-q-values')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ep_range = np.arange(start_ep, end_ep + ep_step, ep_step)

    NB_TRACKED_STATES = 10
    env.seed(123)
    track_states = [env.observation_space.sample() for _ in range(NB_TRACKED_STATES)]
    trackes_vals = np.zeros(shape=(NB_TRACKED_STATES, len(ep_range)))
    for i, ep in enumerate(ep_range):
        print(f'\rEpisode #{ep:4d}/{end_ep}', end='')
        qvf = load_qvf_for_ep(ep)
        for j, ts in enumerate(track_states):
            trackes_vals[j, i] = np.mean([qvf.value(ts, a_) for a_ in all_actions])

    fig, ax = plt.subplots()
    tok = par_dir.split('_')
    fig.suptitle(' '.join(tok[:4]) + '\n' + ' '.join(tok[4:]) +
                 '\n' + f'Number of states tracked = {NB_TRACKED_STATES}')
    ax.plot(ep_range, trackes_vals.T, marker='x')
    ax.set_xlabel('Training episodes')
    ax.set_ylabel('Estimated value')

    plt.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', ls=':')

    fig.tight_layout()
    plt.savefig(os.path.join(results_path, f'{NB_TRACKED_STATES}random-states.png'))
    plt.close(fig)
    print('')


if __name__ == '__main__':
    # track_state_values()
    # episode_length_statistics()
    evaluation_episodes(800, 900)
