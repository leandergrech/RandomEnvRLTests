import os
import csv
import re
from collections import defaultdict
from comet_ml import API
import numpy as np
import matplotlib.pyplot as plt

COMET_API_KEY = 'LvCyhW3NX1yaPPqv3LIMb1qDr'
COMET_WORKSPACE_NAME = 'testing-ppo-trpo'
COMET_PROJECT_NAME = 'sess-trpo-050522-131337'
algo_name = 'TRPO'

api = API(api_key=COMET_API_KEY)
experiments = api.get(COMET_WORKSPACE_NAME, COMET_PROJECT_NAME)


def get_experiment_by_name(name):
    for e in experiments:
        if name in e.get_name():
            return e


def get_metric_list(experiment, metric_name):
    raw_data = experiment.get_metrics(metric_name)
    data = []
    steps = []
    for item in raw_data:
        data.append(item['metricValue'])
        steps.append(item['step'])
    return data, steps


data = defaultdict(list)
for filename in os.listdir('csv'):
    env_sz, *_ = [int(item) for item in re.findall(r'\d+', filename)]

    # Get experiment names from this file
    with open(os.path.join('csv', filename), 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            expname = row[0]
            if 'Name' in expname: continue
            e = get_experiment_by_name(expname)
            _, steps = get_metric_list(e, 'eval/success')
            print(f'{expname}\t{env_sz}x{env_sz}\ttotal training steps = {steps[-1]}')

            data[env_sz].append(steps[-1])

fig, ax = plt.subplots()
nb_training_steps_mean_list = {}
for env_sz, nb_training_steps in data.items():
    x = np.repeat(env_sz, len(nb_training_steps))
    ax.scatter(x, nb_training_steps, marker='x', zorder=15)
    nb_training_steps_mean_list[env_sz] = np.mean(nb_training_steps)

sizes = np.arange(min(nb_training_steps_mean_list), max(nb_training_steps_mean_list) + 1)
means = [nb_training_steps_mean_list[item] for item in sizes]
ax.plot(sizes, means, marker='o', color='k', label='Means', zorder=10)

fig.suptitle(f'{algo_name} trained on RandomEnv\nNumber of training steps vs. environment size')
ax.legend(loc='upper left')
ax.set_ylabel('Training steps until convergence')
ax.set_xlabel('Environment size n (n obs x n act)')
ax.grid(which='both')
fig.suptitle('IdentityEnv')
fig.tight_layout()
fig.savefig('results.png')
plt.show()
