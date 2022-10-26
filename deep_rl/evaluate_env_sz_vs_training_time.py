import os
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.plotting_utils import grid_on
import pickle as pkl
import csv
from comet_ml import API
from tqdm import tqdm


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

trpo_csv = 'testing_ppo_trpo_sess_trpo_050522_131337.csv'

def create_ppo_result_heuristics():
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

    with open(os.path.join(par_dir, f'ppo_train_steps.pkl'), 'wb') as f:
        pkl.dump(dict(train_steps=scatter_train_steps,
                      env_szs=scatter_env_szs,
                      env_sz_to_train_steps=env_sz_to_train_steps), f)


def create_trpo_result_heuristics():
    COMET_API_KEY = 'LvCyhW3NX1yaPPqv3LIMb1qDr'
    COMET_WORKSPACE_NAME = 'testing-ppo-trpo'
    COMET_PROJECT_NAME = 'sess-trpo-050522-131337'

    api = API(api_key=COMET_API_KEY)
    experiments = api.get(COMET_WORKSPACE_NAME, COMET_PROJECT_NAME)

    csv_file = os.path.join(par_dir, trpo_csv)

    env_sz_to_train_steps = defaultdict(list)
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for item in tqdm(reader):
            comet_name = item['Name']
            env_sz = int(re.findall('\d+', item['Tags'])[0])
            for i, e in enumerate(experiments):
                e_name = e.get_name()
                if e_name == comet_name:
                    train_step = e.get_metrics('eval/success')[-1]['step']
                    env_sz_to_train_steps[env_sz].append(train_step)
                    break
                if i == len(experiments) - 1:
                    raise f'Experiment {comet_name} not found in Comet-ML!'

    env_sz_list = sorted(env_sz_to_train_steps.keys())
    scatter_train_steps, scatter_env_szs = [], []
    for env_sz in env_sz_list:
        for train_step in env_sz_to_train_steps[env_sz]:
            scatter_env_szs.append(env_sz)
            scatter_train_steps.append(train_step)

    with open(os.path.join(par_dir, f'trpo_train_steps.pkl'), 'wb') as f:
        pkl.dump(dict(train_steps=scatter_train_steps,
                      env_szs=scatter_env_szs,
                      env_sz_to_train_steps=env_sz_to_train_steps), f)


def load_result_heuristics(algo_name):
    with open(os.path.join(par_dir, f'{algo_name.lower()}_train_steps.pkl'), 'rb') as f:
        d = pkl.load(f)
        scatter_train_steps = d['train_steps']
        scatter_env_szs = d['env_szs']
        env_sz_to_train_steps = d['env_sz_to_train_steps']

    return scatter_train_steps, scatter_env_szs, env_sz_to_train_steps


def calc_average_track(e2ts):
    env_sz_list = sorted(e2ts.keys())
    mean_train_step = []
    for es in env_sz_list:
        mean_train_step.append(np.mean(e2ts[es]))

    return env_sz_list, mean_train_step

def plot_result_heuristics(algo_name, log_scale=False):
    scatter_train_steps, scatter_env_szs, env_sz_to_train_steps = load_result_heuristics(algo_name)

    plot_average_x, plot_average_train_step = calc_average_track(env_sz_to_train_steps)


    fig, ax = plt.subplots()

    fig.suptitle(f'{algo_name.upper()} trained on RandomEnv\nNumber of training steps vs. environment size')
    ax.scatter(scatter_env_szs, scatter_train_steps, marker='x', zorder=15)
    ax.plot(plot_average_x, plot_average_train_step, marker='o', c='k', label='Means', zorder=10)
    if log_scale:
        ax.set_yscale('log')
    grid_on(ax, 'y')
    grid_on(ax, 'x', minor_grid=False)
    ax.set_xticks(plot_average_x)
    ax.set_xticklabels(plot_average_x)
    ax.set_ylabel('Training steps until convergence')
    ax.set_xlabel('Environment size m (m obs x m act)')
    fig.savefig(os.path.join(par_dir, f'{algo_name}({"log" if log_scale else "linear"}).png'))


def plot_compare_ppo_trpo():
    x_ppo, means_ppo = calc_average_track(load_result_heuristics('ppo')[2])
    x_trpo, means_trpo = calc_average_track(load_result_heuristics('trpo')[2])

    fig, ax = plt.subplots()
    ax.plot(x_ppo, means_ppo, marker='x', label='PPO')
    ax.plot(x_trpo, means_trpo, marker='o', label='TRPO')

    grid_on(ax, 'y')
    grid_on(ax, 'x', minor_grid=False)

    ax.set_ylabel('Training steps until convergence')
    ax.set_xlabel('Environment size m (m obs x m act)')

    ax.legend(loc='best')

    fig.savefig(os.path.join(par_dir, 'compare_ppo_vs_trpo.png'))




if __name__ == '__main__':
    # create_trpo_result_heuristics()
    # for algo_name in ('trpo', 'ppo'):
    #     plot_result_heuristics(algo_name, True)
    #     plot_result_heuristics(algo_name, False)
    plot_compare_ppo_trpo()
