import os
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import yaml

from utils.plotting_utils import grid_on, recolor_yaxis
import pickle as pkl

# experiment_dir = 'ramp_env_size_102322_201202'
experiment_dir = 'ramp_env_size_102022_104654'
nb_actions = lambda item: 3**item         # Power action set
# nb_actions = lambda item: 2*item + 1        # Canonical action set


def get_sorted_experiments(p):
    fns = [item for item in os.listdir(p) if not os.path.splitext(item)[-1]]
    return sorted(fns, key=lambda item: int(re.findall('\d+', item)[0]))


def get_params(p):
    params_file = os.path.join(p, 'train_params.yml')
    with open(params_file, 'r') as f:
        d = yaml.load(f, yaml.Loader)
    return d


def convergence_analysis(experiment_dir, nb_actions):
    scatter_env_sz_list = []
    env_sz_list = []
    train_step_list = []

    train_info = defaultdict(list)
    train_params = defaultdict(dict)

    for sub_exp in get_sorted_experiments(experiment_dir):
        env_sz = int(re.findall('\d+', sub_exp)[0])

        d = get_params(os.path.join(experiment_dir, sub_exp))

        train_params[env_sz]['nb_training_steps'] = d['nb_training_steps']
        train_params[env_sz]['lr_fun_init'] = d['lr_fun'].init
        train_params[env_sz]['lr_fun_final'] = d['lr_fun'].final
        train_params[env_sz]['exp_fun_init'] = d['exp_fun'].init
        train_params[env_sz]['exp_fun_final'] = d['exp_fun'].final

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
    train_params_lists = defaultdict(list)
    for env_sz in env_sz_list:
        average_trace.append(np.mean(train_info[env_sz]))
        for k, v in train_params[env_sz].items():
            train_params_lists[k].append(v)

    fig, axs = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [3,1,1,1]},
                            figsize=(7, 7))

    ax = axs[0]
    ax.plot(env_sz_list, average_trace, c='k', lw=1)
    for env_sz, dat in train_info.items():
        ax.scatter(np.repeat(env_sz, len(dat)), dat, marker='x')
    ax.set_xlabel('Environment size n (n obs x n act)')
    ax.set_ylabel('Training steps until convergence')

    axx = ax.twinx()
    recolor_yaxis(axx, 'r')

    axx.plot(env_sz_list, [nb_actions(item) for item in env_sz_list], c='r')
    axx.set_ylabel('Number of discrete actions')

    grid_on(ax, axis='y', major_loc=max(train_step_list)//5, minor_loc=max(train_step_list)//20)

    ax = axs[1]
    for k, v in train_params_lists.items():
        if 'lr' in k:
            ax = axs[2]
            ax.plot(env_sz_list, v, label=k, marker='o')
            ax.set_ylabel('Learning rate')
            ax.set_yscale('log')
        elif 'exp' in k:
            ax = axs[3]
            ax.plot(env_sz_list, np.array(v)*100., label=k, marker='o')
            ax.set_ylabel('Exploration (%)')
        elif 'steps' in k:
            ax = axs[1]
            ax.plot(env_sz_list, v, label=k, marker='o')
            ax.set_ylabel('Nb training steps')
            grid_on(ax, axis='y', major_loc=max(v)//5)
        ax.set_xlabel('Environment size n (n obs x n_act)')
        ax.legend(loc='best')

    for ax in axs:
        grid_on(ax, axis='x', major_loc=1, minor_loc=None, major_grid=True, minor_grid=False)

    with open(os.path.join(experiment_dir, 'convergence_analysis.pkl'), 'wb') as f:
        pkl.dump(dict(train_info=train_info,
                      env_sz_list=env_sz_list,
                      average_trace=average_trace), f)

    fig.tight_layout()
    fig.savefig(os.path.join(experiment_dir, 'convergence_analysis.png'))

def compare_permutation_vs_canonical():
    perm_experiment_dir = 'ramp_env_size_102022_104654'
    cano_experiment_dir = 'ramp_env_size_102322_201202'

    label_fs = 16

    # fig, axs = plt.subplots(3, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig, ax = plt.subplots(1, figsize=(7, 7))
    axs = [ax]

    ##### START PPO & TRPO

    from deep_rl.evaluate_env_sz_vs_training_time import load_result_heuristics, calc_average_track

    ax = axs[0]
    for algo_name, m in zip(('ppo', 'trpo'), ('v', '^')):
        with open(f'/home/leander/code/RandomEnvRLTests/deep_rl/sb3_identityenv_training/{algo_name}_train_steps.pkl',
                  'rb') as f:
            d = pkl.load(f)
            scatter_train_steps = d['train_steps']
            scatter_env_szs = d['env_szs']
            env_sz_to_train_steps = d['env_sz_to_train_steps']

            env_sz_list2, mean_train_step = calc_average_track(env_sz_to_train_steps)

        ax.plot(env_sz_list2, mean_train_step, label=f'{algo_name.upper()}', marker=m)
    # ax.set_yscale('log')
    ##### END PPO & TRPO

    for label, exp_dir, m in zip(('Linear SARSA - All action permutations', 'Linear SARSA - Canonical actions'),
                              (perm_experiment_dir, cano_experiment_dir),
                              ('o', 'x')):
        ax = axs[0]
        with open(os.path.join(exp_dir, 'convergence_analysis.pkl'), 'rb') as f:
            dat = pkl.load(f)
        env_sz_list = dat['env_sz_list']
        average_trace = dat['average_trace']
        ax.plot(env_sz_list, average_trace, label=label, marker=m)
        ax.set_ylabel('Training steps until convergence', size=label_fs)
        # grid_on(ax, 'y', major_loc=max(ax.get_ylim()) // 5)
        grid_on(ax, 'x', major_loc=10, minor_loc=1, major_grid=True, minor_grid=True)

        from matplotlib.ticker import LogLocator

        axis_ = ax.yaxis
        axis_.set_major_locator(LogLocator(base=10.0, numticks=10))
        # axis_.set_minor_locator(LogLocator(10.0, (1.0,), numticks=10))
        ax.minorticks_on()
        ax.grid(which='major', c='gray', axis='y')
        # ax.grid(which='minor', c='gray', ls='--', alpha=0.5, axis='y')



        continue

        ret = load_analysis_ep_lens(exp_dir)
        mean_returns = ret['mean_returns']
        mean_ep_lens = ret['mean_ep_lens']

        unpack = lambda d: [np.mean(d[e]) for e in env_sz_list]
        ax = axs[1]
        mean_returns = unpack(mean_returns)
        ax.plot(env_sz_list, mean_returns, label=label, marker=m)
        ax.set_ylabel('Average\ntest return', size=label_fs)
        grid_on(ax, 'y', major_loc=max(np.abs(mean_returns))//5)

        ax = axs[2]
        mean_ep_lens = unpack(mean_ep_lens)
        ax.plot(env_sz_list, mean_ep_lens, label=label, marker=m)
        ax.set_ylabel('Average test\nepisode length', size=label_fs)
        grid_on(ax, 'y', major_loc=max(mean_ep_lens) // 5)

    for ax in axs:
        ax.legend(loc='best')
        ax.set_xlabel('Environment size n (n obs x n act)', size=label_fs)

    fig.tight_layout()
    # fig.savefig('compare_permutation_vs_canonical.png')
    fig.savefig('compare_permutation_vs_canonical_vs_ppo_vs_trpo.png')




from utils.eval_utils import eval_agent, get_q_func_filenames, get_q_func_xrange
from linear_q_function import QValueFunctionLinear, FeatureExtractor
from random_env.envs import REDAClip, REDAClipCont
from tqdm import tqdm

def get_train_step(p):
    train_info_file = os.path.join(p, 'train_info.md')

    with open(train_info_file, 'r') as f:
        train_step = int(f.read())

    return train_step

def create_subexperiment_training_stats(experiment_dir):
    mean_returns = defaultdict(list)
    mean_ep_lens = defaultdict(list)
    for sub_exp in tqdm(os.listdir(experiment_dir)):
        if 'seed' not in sub_exp:
            continue

        env_sz = int(re.findall('\d+', sub_exp)[0])

        sub_exp_path = os.path.join(experiment_dir, sub_exp)
        qfn = get_q_func_filenames(sub_exp_path)[-1]


        q = QValueFunctionLinear.load(qfn)
        env = REDAClipCont.load_from_dir(sub_exp_path)
        env = REDAClip(env.obs_dimension, env.act_dimension, state_clip=1.0, model_info=env.model_info)
        env.EPISODE_LENGTH_LIMIT = 300

        ret = eval_agent(eval_env=env, q=q, nb_eps=10)

        mean_returns[env_sz].append(np.mean(ret['returns']))
        mean_ep_lens[env_sz].append(np.mean(ret['ep_lens']))

    with open(os.path.join(experiment_dir, 'subexperiment_training_stats.pkl'), 'wb') as f:
        pkl.dump(dict(mean_returns=mean_returns,
                      mean_ep_lens=mean_ep_lens), f)


def load_analysis_ep_lens(experiment_dir):
    with open(os.path.join(experiment_dir, 'subexperiment_training_stats.pkl'), 'rb') as f:
        d = pkl.load(f)
    return d

def convergence_analysis_ep_lens(experiment_dir, nb_actions, nb_eps=10):
    # scatter_env_sz_list = []
    env_sz_list = []
    # train_step_list = []

    train_info = defaultdict(list)
    train_params = defaultdict(dict)

    # Get training steps until convergence
    for sub_exp in get_sorted_experiments(experiment_dir):
        env_sz = int(re.findall('\d+', sub_exp)[0])

        train_step = get_train_step(os.path.join(experiment_dir, sub_exp))

        # scatter_env_sz_list.append(env_sz)
        # train_step_list.append(train_step)
        train_info[env_sz].append(train_step)

        if env_sz not in env_sz_list:
            env_sz_list.append(env_sz)

    average_trace = []
    train_params_lists = defaultdict(list)
    for env_sz in env_sz_list:
        average_trace.append(np.mean(train_info[env_sz]))
        for k, v in train_params[env_sz].items():
            train_params_lists[k].append(v)

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                            figsize=(10, 10))

    ax = axs[0]
    ax.plot(env_sz_list, average_trace, c='k', lw=1, zorder=15)
    for env_sz in env_sz_list:
        dat = train_info[env_sz]
        ax.scatter(np.repeat(env_sz, len(dat)), dat, marker='x')
    ax.set_xlabel('Environment size n (n obs x n act')
    ax.set_ylabel('Training steps until convergence')

    axx = ax.twinx()
    recolor_yaxis(axx, 'r')

    axx.plot(env_sz_list, [nb_actions(item) for item in env_sz_list], c='r')
    axx.set_ylabel('Number of discrete actions')
    grid_on(ax, axis='y', major_loc=10000, minor_loc=2500)

    for ax in axs:
        ax.set_xlabel('Environment size n (n obs x n_act)')
        grid_on(ax, axis='x', major_loc=1, major_grid=True, minor_grid=False)

    ret = load_analysis_ep_lens(experiment_dir)
    mean_returns = ret['mean_returns']
    mean_ep_lens = ret['mean_ep_lens']

    ax = axs[1]
    ax.plot(env_sz_list, [np.mean(mean_ep_lens[es]) for es in env_sz_list], c='k')
    ax.set_ylabel('Average test episode lengths')

    axx = ax.twinx()
    recolor_yaxis(axx, 'b')
    axx.plot(env_sz_list, [np.mean(mean_returns[es]) for es in env_sz_list], c='b')
    axx.set_ylabel('Average test returns')

    with open(os.path.join(experiment_dir, 'convergence_analysis_ep_lens.pkl'), 'wb') as f:
        pkl.dump(dict(train_info=train_info,
                      average_trace=average_trace,
                      env_sz_list=env_sz_list,
                      mean_returns=mean_returns,
                      mean_ep_lens=mean_ep_lens), f)


    fig.tight_layout()
    fig.savefig(os.path.join(experiment_dir, 'convergence_analysis_ep_lens.png'))
    plt.show()

def compare_with_ppo_trpo(exp_dir):
    load_analysis_ep_lens(exp_dir)


if __name__ == '__main__':
    experiment_dir = 'ramp_env_size_102022_104654'
    # experiment_dir = 'ramp_env_size_102322_201202'
    # create_subexperiment_training_stats(experiment_dir)
    # nb_actions = lambda i: 3**i
    # nb_actions = lambda i: 2*i + 1
    # convergence_analysis_ep_lens(experiment_dir=experiment_dir, nb_actions=nb_actions, nb_eps=10)
    # convergence_analysis(experiment_dir, nb_actions)
    compare_permutation_vs_canonical()
