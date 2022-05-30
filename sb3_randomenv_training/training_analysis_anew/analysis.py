import os
import csv
import re
from collections import defaultdict
from comet_ml import API
import numpy as np
import matplotlib.pyplot as plt

COMET_API_KEY = 'LvCyhW3NX1yaPPqv3LIMb1qDr'
COMET_WORKSPACE_NAME = 'testing-ppo-trpo'
COMET_PROJECT_NAME = 'sess-trpo-050622-004429'

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

csv_dir = 'csv'
data = defaultdict(list)
for filename in os.listdir(csv_dir):
	env_sz, _, env_idx = [int(item) for item in re.findall(r'\d+', filename)]

	# Get metric data
	data[env_sz].append([])
	with open(os.path.join(csv_dir, filename), 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			expname = row[0]
			if 'Name' in expname: continue

			e = get_experiment_by_name(expname)
			_, steps = get_metric_list(e, 'eval/success')
			print(f'{expname}\t{env_sz}x{env_sz}\ttotal training steps = {steps[-1]}')

			data[env_sz][-1].append(steps[-1])

# plotting
fig, ax = plt.subplots()
nb_training_steps_mean_list = defaultdict(list)
env_seeds = (123, 234, 345, 456, 567)
colors = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd')
markers = ('x', '1', '2', '3', '4')
for i, (env_sz, datum) in enumerate(data.items()):
	for nb_training_steps, c, env_seed, m in zip(datum, colors, env_seeds, markers):
		x = np.repeat(env_sz, len(nb_training_steps))
		if i == 0: label=f'env_seed={env_seed}'
		else: label=None
		ax.scatter(x, nb_training_steps, marker=m, s=60, color=c, label=label, zorder=15)
		nb_training_steps_mean_list[env_sz].append(np.mean(nb_training_steps))

sizes = np.arange(min(nb_training_steps_mean_list), max(nb_training_steps_mean_list) + 1)
means = [[nb_training_steps_mean_list[item][i] for item in sizes] for i in range(len(nb_training_steps_mean_list[sizes[0]]))]
for m, c, env_seed in zip(means, colors, env_seeds):
	ax.plot(sizes, m, color=c, zorder=10, alpha=0.5)

ax.legend(loc='upper left')

ax.set_ylabel('Training steps until convergence')
ax.set_xlabel('Environment size n (n obs x n act)')
ax.grid(which='both')
fig.suptitle('RandomEnv')
fig.tight_layout()
fig.savefig('results_with_mean.png')
plt.show()





