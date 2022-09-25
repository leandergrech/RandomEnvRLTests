import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random_env.envs import get_discrete_actions


def play_episode(eval_env, q, init_state=None):
    actions = get_discrete_actions(eval_env.act_dimension, 3)
    obses = []
    acts = []
    rews = []
    o = eval_env.reset(init_state)
    obses.append(o.copy())
    d = False
    while not d:
        a = q.greedy_action(o)
        otp1, r, d, _ = eval_env.step(actions[a])
        o = otp1
        obses.append(o.copy())
        acts.append(a)
        rews.append(r)

    return obses, acts, rews


def eval_agent(eval_env, q, nb_eps):
    init_obses = np.empty(shape=(0, eval_env.obs_dimension))
    terminal_obses = np.empty(shape=(0, eval_env.obs_dimension))
    ep_lens = []
    returns= []

    actions = get_discrete_actions(eval_env.act_dimension, 3)
    for _ in range(nb_eps):
        o = eval_env.reset()
        d = False
        init_obses = np.vstack([init_obses, o])
        t = 0
        g = 0
        while not d:
            a = q.greedy_action(o)
            otp1, r, d, _ = eval_env.step(actions[a])

            g += r
            o = otp1
            t += 1
        terminal_obses = np.vstack([terminal_obses, o])
        ep_lens.append(t)
        returns.append(g)

    obses = dict(initial_observations=init_obses, terminal_observations=terminal_obses)
    # ep_len_stats = dict(mean=np.mean(ep_lens), median=np.median(ep_lens), min=min(ep_lens), max=max(ep_lens))
    return obses, returns, ep_lens


def make_state_violins(init_obses, terminal_obses, path):
    data = pd.DataFrame()
    for obs_type, obses in zip(('init', 'terminal'), (init_obses, terminal_obses)):
        for dim, vals in enumerate(obses.T):
            df = pd.DataFrame(dict(obs_type=obs_type, dim=dim, vals=vals))
            data = pd.concat([data, df], ignore_index=True)

    plot = sns.violinplot(data=data, x='dim', y='vals', hue='obs_type', split=True, inner='quart', lw=1)
    plot.axhline(0.0, color='k')
    fig = plot.get_figure()

    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    plt.savefig(path)
    plt.close(fig)


def get_latest_experiment(lab_dir, pattern='sarsa'):
    experiments = []
    for fn in os.listdir(lab_dir):
        if pattern in fn:
            experiments.append(fn)
    experiments = sorted(experiments)

    experiment_name = experiments[-1]

    return os.path.join(lab_dir, experiment_name)


def get_optimal_ep_len(env, nb_eps=200):
    """
    Run optimal policy provided by env.get_optimal_policy for nb_eps episodes
    and return average episode length.
    """
    ep_lens = []
    for _ in range(nb_eps):
        o = env.reset()
        d = False
        t = 0
        while not d:
            a = env.get_optimal_action(o)
            otp1, _, d, _ = env.step(a)
            otp1 = o
            t += 1
        ep_lens.append(t)
    return np.mean(ep_lens)


def get_q_func_step(fn):
    """
        Every experiment contains q_func directory which stores the QValueFunctionTiles3
        instance at a specified training step. Training step X obtained from filename fn
        in the form q_step_X.pkl.
    """
    return int(os.path.split(fn)[-1].split('_')[2].split('.')[0])


def get_q_func_filenames(experiment_dir):
    """
        Every experiment has q_func directory. Get sorted filenames found within.
    """
    q_func_dir = os.path.join(experiment_dir, 'q_func')
    q_func_filenames = [fn for fn in os.listdir(q_func_dir)]

    q_func_filenames = sorted(q_func_filenames, key=get_q_func_step)
    q_func_filenames = [os.path.join(q_func_dir, item) for item in q_func_filenames]

    return q_func_filenames


def get_q_func_xrange(q_func_filenames):
    """
        Given a sorted list of q-table files stored during an experiment, return
        the xrange to be used for the plotting x-axis.
    """
    return np.linspace(get_q_func_step(q_func_filenames[0]), get_q_func_step(q_func_filenames[-1]), len(q_func_filenames))


def get_val(qvf, state, nb_actions):
    """
        Value is defined as the expected return given a state. QValueFunctionTiles3
        only gives us Q-values. I'm assuming the value is the average of all q-values
        obtained from all possible actions.
    """
    return np.mean([qvf.value(state, a_) for a_ in range(nb_actions)])


if __name__ == '__main__':
    from random_env.envs.random_env_discrete_actions import RandomEnvDiscreteActions as REDA
    from tqdm import trange
    ep_lens = []
    for _ in trange(20):
        env = REDA(2,2)
        ep_lens.append(get_optimal_ep_len(env))
    fig, ax = plt.subplots()
    ax.plot(ep_lens)
    plt.show()
