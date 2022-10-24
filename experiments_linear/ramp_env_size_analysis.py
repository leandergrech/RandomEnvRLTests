import os
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from utils.plotting_utils import grid_on

experiment_dir = 'ramp_env_size_102022_104654'

env_sz_list = []
train_step_list = []

train_info = defaultdict(list)

for sub_exp in sorted(os.listdir(experiment_dir)):
    if '.png' in sub_exp:
        continue

    env_sz = int(re.findall('\d+', sub_exp)[0])

    train_info_file = os.path.join(experiment_dir, sub_exp, 'train_info.md')
    if not os.path.exists(train_info_file):
        continue

    with open(train_info_file, 'r') as f:
        train_step = int(f.read())

    env_sz_list.append(env_sz)
    train_step_list.append(train_step)
    train_info[env_sz].append(train_step)

average_trace = []
average_trace_x = np.arange(min(env_sz_list), max(env_sz_list)+1)
for env_sz in average_trace_x:
    average_trace.append(np.mean(train_info[env_sz]))
print(train_info)
fig, ax = plt.subplots()
ax.plot(average_trace_x, average_trace, c='k', lw=1)
ax.scatter(env_sz_list, train_step_list, marker='x')
ax.set_xlabel('Environment size n (n obs x n act')
ax.set_ylabel('Training steps until convergence')

axx = ax.twinx()
axx.spines.right.set_color('r')
axx.yaxis.label.set_color('r')
axx.tick_params(axis='y', colors='r')
axx.plot(average_trace_x, [3**item for item in average_trace_x], c='r')
axx.set_ylabel('Number of discrete actions')
grid_on(ax, axis='y', major_loc=10000, minor_loc=2500)
fig.tight_layout()
fig.savefig(os.path.join(experiment_dir, 'convergence_analysis.png'))
plt.show()

