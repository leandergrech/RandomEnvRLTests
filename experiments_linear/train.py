import os
import numpy as np
from datetime import datetime as dt
import yaml
import matplotlib.pyplot as plt
import pickle as pkl
from multiprocessing import Pool

from utils.training_utils import Constant, LinearDecay, boltzmann, eps_greedy
from experiments_linear.sarsa import train_instance; algo_name = 'sarsa'
from random_env.envs import *


def run_experiment(experiment_dir):
    n_obs, n_act = 10, 10

    env = REDAClipCont(n_obs, n_act, 1.0)

    nb_training_steps = 100000
    eval_every = 100000
    save_every = 10000
    explore_until = 1000000
    decay_lr_until = 1000000
    exp_fun = LinearDecay(1.0, 1e-2, explore_until, label='EPS')
    lr_fun = LinearDecay(1e-1, 1e-2, decay_lr_until, label='LR')
    gamma = 0.9

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    train_params = dict(
        n_obs=n_obs,
        n_act=n_act,
        env_save_path=experiment_dir,
        results_path=experiment_dir,
        lr_fun=lr_fun,
        exp_fun=exp_fun,
        nb_training_steps=nb_training_steps,
        eval_every=eval_every,
        eval_eps=10,
        save_every=save_every,
        gamma=gamma
    )
    with open(os.path.join(experiment_dir, "train_params.yml"), "w") as f:
        f.write(yaml.dump(train_params))

    train_params['env'] = env
    if 'EPS' in exp_fun.label:
        train_params['policy'] = eps_greedy
    elif 'TAU' in exp_fun.label:
        train_params['policy'] = boltzmann

    # Training
    ret = train_instance(**train_params)

    # Save training data
    with open(os.path.join(experiment_dir, 'training_stats.pkl'), 'wb') as f:
        pkl.dump(ret, f)


if __name__ == '__main__':
    nb_trials = 1
    exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S_{nb_trials}trials')
    # with Pool(4) as p:
    #     exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S_{nb_trials}trials')
    #     p.map(run_experiment, [os.path.join(exp_prefix, f'trial_{item}') for item in range(nb_trials)])
    for i in range(nb_trials):
        run_experiment(os.path.join(exp_prefix, f'trial_{i}'))
    # run_experiment(dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S'))
