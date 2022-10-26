import os
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.plotting_utils import grid_on
import pickle as pkl


def get_max_training_steps(p):
    saved_steps = []
    for item in os.listdir(os.path.join(p, 'saves')):
        saved_steps.append(int(re.findall('\d+', item)[0]))

    if len(saved_steps) == 0:
        return np.nan
    else:
        return max(saved_steps)


par_dir = 'sb3_identityenv_training'
sess_dir = 'sess_ppo_102422_150542'
exp_pardir = os.path.join(par_dir, sess_dir)

exp_dirs = []
env_sz_to_train_steps = defaultdict(list)
scatter_train_steps = []
scatter_env_szs = []

env_sz_list = []


def get_sorted_sub_exps(p):
    sub_exps = []
    for file in os.listdir(p):
        if os.path.splitext(file)[-1]:
            continue
        else:
            sub_exps.append(file)
    return sorted(sub_exps, key=lambda item: int(re.findall('\d+', item)[0]))


for exp_fn in get_sorted_sub_exps(exp_pardir):
    env_sz, _, seed = (int(item) for item in re.findall('\d+', exp_fn))

    exp_dir = os.path.join(exp_pardir, exp_fn)
    exp_dirs.append(exp_dir)
    train_step = get_max_training_steps(exp_dir)

    env_sz_to_train_steps[env_sz].append(train_step)
    scatter_train_steps.append(train_step)
    scatter_env_szs.append(env_sz)
    if env_sz not in env_sz_list:
        env_sz_list.append(env_sz)



plot_average_train_step = []
plot_average_x = np.arange(min(scatter_env_szs), max(scatter_env_szs) + 1, dtype=int)
for env_sz in plot_average_x:
    plot_average_train_step.append(np.mean(env_sz_to_train_steps[env_sz]))

with open(os.path.join(par_dir, f'{sess_dir}_train_steps.pkl'), 'wb') as f:
    pkl.dump(dict(train_steps=scatter_train_steps,
                  env_szs=scatter_env_szs,
                  env_sz_to_train_steps=env_sz_to_train_steps), f)


fig, ax = plt.subplots()
if 'ppo' in sess_dir.lower():
    algo_name = 'PPO'
elif 'trpo' in sess_dir.lower():
    algo_name = 'TRPO'

fig.suptitle(f'{algo_name} trained on RandomEnv\nNumber of training steps vs. environment size')
ax.scatter(scatter_env_szs, scatter_train_steps, marker='x', zorder=15)
ax.plot(plot_average_x, plot_average_train_step, marker='o', c='k', label='Means', zorder=10)
ax.set_yscale('log')
grid_on(ax, 'y')
grid_on(ax, 'x', minor_grid=False)
ax.set_xticks(plot_average_x)
ax.set_xticklabels(plot_average_x)
ax.set_ylabel('Training steps until convergence')
ax.set_xlabel('Environment size n (n obs x n act)')
fig.savefig(os.path.join(par_dir, f'{sess_dir}.png'))
plt.show()






