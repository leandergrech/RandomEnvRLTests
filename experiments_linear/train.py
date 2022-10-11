import os
import numpy as np
from datetime import datetime as dt
import yaml
import matplotlib.pyplot as plt
import pickle as pkl
from multiprocessing import Pool

from utils.training_utils import Constant, LinearDecay, boltzmann
from experiments_linear.sarsa import train_instance; algo_name = 'sarsa'
from random_env.envs import REDAClip, IREDA, RandomEnvDiscreteActions as REDA


def run_experiment(exp_name):
    n_obs, n_act = 2, 2
    # state_clip = 1.0
    # env = REDAClip(n_obs, n_act, state_clip)
    env = REDA(n_obs, n_act)
    # env = IREDA(n_obs, n_act)

    nb_training_steps = 1000
    eval_every = 10
    save_every = 10
    explore_until = 800
    exp_fun = LinearDecay(1.0, 1e-2, explore_until, label='TAU')
    # lr_fun = Constant(1e-1, label='LR')
    lr_fun = LinearDecay(1e-1, 1e-2, 1000, label='LR')
    gamma = 0.9

    sub_experiment_names = ['default']

    for experiment_name in sub_experiment_names:
        experiment_dir = os.path.join(exp_name, experiment_name)
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
        train_params['policy'] = boltzmann

        # Training
        ret = train_instance(**train_params)

        # Save training data
        with open(os.path.join(experiment_dir, 'training_stats.pkl'), 'wb') as f:
            pkl.dump(ret, f)


if __name__ == '__main__':
    nb_trials = 1
    # with Pool(4) as p:
    #     exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S')
    #     p.map(run_experiment, [f'{exp_prefix}_{item}' for item in range(nb_trials)])
    run_experiment(dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S'))
