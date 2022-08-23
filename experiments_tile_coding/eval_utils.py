import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random_env.envs import get_discrete_actions


def eval_agent(eval_env, q):
    nb_eps = 100
    init_obses = np.empty(shape=(0, eval_env.obs_dimension))
    terminal_obses = np.empty(shape=(0, eval_env.obs_dimension))
    ep_lens = []

    actions = get_discrete_actions(eval_env.act_dimension, 3)
    for _ in range(nb_eps):
        o = eval_env.reset()
        d = False
        init_obses = np.vstack([init_obses, o])
        t = 0
        while not d:
            a = q.greedy_action(o)
            otp1, r, d, _ = eval_env.step(actions[a])

            o = otp1
            t += 1
        terminal_obses = np.vstack([terminal_obses, o])
        ep_lens.append(t)

    return init_obses, terminal_obses, np.mean(ep_lens)


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