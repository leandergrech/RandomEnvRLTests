import os
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from utils.plotting_utils import grid_on

experiment_dir = 'ramp_env_size_102322_201202'

scatter_env_sz_list = []
env_sz_list = []
train_step_list = []

train_info = defaultdict(list)

def get_sorted_experiments(p):
    fns = [item for item in os.listdir(p) if not os.path.splitext(item)[-1]]
    return sorted(fns, key=lambda item: int(re.findall('\d+', item)[0]))

for sub_exp in get_sorted_experiments(experiment_dir):
    env_sz = int(re.findall('\d+', sub_exp)[0])

    train_info_file = os.path.join(experiment_dir, sub_exp, 'train_info.md')
    if not os.path.exists(train_info_file):
        continue

    with open(train_info_file, 'r') as f:
        train_step = int(f.read())

    scatter_env_sz_list.append(env_sz)
    train_step_list.append(train_step)
    train_info[env_sz].append(train_step)

    if env_sz not in env_sz_list:
        env_sz_list.append(env_sz)

average_trace = []
for env_sz in env_sz_list:
    average_trace.append(np.mean(train_info[env_sz]))
print(train_info)
fig, ax = plt.subplots()
ax.plot(env_sz_list, average_trace, c='k', lw=1)
ax.scatter(scatter_env_sz_list, train_step_list, marker='x')
ax.set_xlabel('Environment size n (n obs x n act')
ax.set_ylabel('Training steps until convergence')

axx = ax.twinx()
axx.spines.right.set_color('r')
axx.yaxis.label.set_color('r')
axx.tick_params(axis='y', colors='r')
# nb_actions = lambda item: 3**item
nb_actions = lambda item: 2*item + 1
axx.plot(env_sz_list, [nb_actions(item) for item in env_sz_list], c='r')
axx.set_ylabel('Number of discrete actions')
grid_on(ax, axis='y', major_loc=10000, minor_loc=2500)
fig.tight_layout()
fig.savefig(os.path.join(experiment_dir, 'convergence_analysis.png'))
plt.show()

