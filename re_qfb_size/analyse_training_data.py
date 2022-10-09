import os
import re

import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt

from utils.plotting_utils import y_grid_on

experiment_name = 'PPO_NoisyClipRE_100822_221745'
# experiment_name = 'TRPO_092922_185111'

tags = ['ep_length', 'reward']

# Get filenames
training_files = defaultdict(list)
for fn in os.listdir(os.path.join(experiment_name, 'training_data')):
    for tag in tags:
        if tag in fn:
            training_files[tag].append(os.path.join(experiment_name, 'training_data', fn))

# Access file training data
training_data = defaultdict(list)
training_steps = defaultdict(list)
for tag in tags:
    for fn in training_files[tag]:
        training_data[tag].append([])
        training_steps[tag].append([])
        with open(fn, 'r') as f:
            data = json.load(f)
        for walltime, training_step, val in data:
            training_data[tag][-1].append(float(val))
            training_steps[tag][-1].append(int(training_step))

# Look for available models which were saved during training
# Look for sub experiment directories
sub_exps = []
for d in os.listdir(experiment_name):
    if os.path.splitext(d)[-1] != '':
        continue
    if 'training_data' in d:
        continue
    sub_exps.append(os.path.join(experiment_name, d))
# # Access sub experiment and get steps available
# available_steps = []
# for d in sub_exps:
#     available_steps.append([])
#     d = os.path.join(d, 'saves')
#     for model_name in os.listdir(d):
#         step = int(re.findall('\d+', model_name)[0])
#         available_steps[-1].append(step)

fig, axs = plt.subplots(2, figsize=(20, 10))
for ax, tag, c in zip(axs, tags, ('b', 'k')):
    data = training_data[tag]
    x = training_steps[tag][0]
    data_med = np.median(data, axis=0)
    data_mea = np.mean(data, axis=0)
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    ax.plot(x, data_med, c=c, label=f'{tag} median', lw=2, zorder=10)
    ax.plot(x, data_mea, c=c, label=f'{tag} mean', lw=2, ls=':', zorder=10)
    for datum, sub_exp in zip(data, sub_exps):
        exp_name = os.path.split(sub_exp)[-1]
        ax.plot(x, datum, lw=0.5, label=exp_name, zorder=15)
    # ax.fill_between(x, data_min, data_max, facecolor='None', edgecolor=c, hatch='//', label=f'{tag} ptp', alpha=0.5)

    xticks = np.arange(0, x[-1], int(1e4))
    ax.set_xticks(xticks, xticks, rotation=20)

    y_grid_on(ax)
    ax.legend(loc='best')
    ax.set_ylabel(tag)
    ax.set_xlabel('Training steps')

# # show available models saved during training
# for sub_exp_steps in available_steps:
#     for step in sub_exp_steps:
#         axs[0].axvline(step, color='c', ls='solid', alpha=0.3, lw=0.5)

fig.suptitle(experiment_name)
fig.tight_layout()
fig.tight_layout()
fig.savefig(os.path.join(experiment_name, 'training_info.png'))
plt.show()


