import os
import numpy as np
from datetime import datetime as dt
import yaml
import pickle as pkl
from multiprocessing import Pool

from utils.training_utils import Constant, LinearDecay, InitSolvableState
from experiments_tile_coding.sarsa import train_instance; algo_name = 'sarsa'
from random_env.envs.random_env_discrete_actions import REDAClip, get_discrete_actions
from experiments_tile_coding.policy_types import boltzmann


def run_experiment(exp_name):
    n_obses = [2, 3, 4, 5]
    n_act = 2

    envs = []
    init_funcs = []
    for n_obs in n_obses:
        env = REDAClip(n_obs, n_act, state_clip=1.0)
        envs.append(env)

        init_func = InitSolvableState(env)
        init_funcs.append(init_func)

    nb_training_steps = 80000
    exp_fun = LinearDecay(1.0, 1e-1, 80000, label='TAU')
    lr_fun = Constant(1e-1, label='LR')

    nb_tilings, nb_bins = 16, 2
    gamma = 0.9

    sub_experiments_names = [repr(env) for env in envs]
    for experiment_name, env, init_func in zip(sub_experiments_names, envs, init_funcs):
        experiment_dir = os.path.join(exp_name, experiment_name)
        os.makedirs(experiment_dir)

        train_params = dict(
            n_obs=env.obs_dimension,
            n_act=n_act,
            nb_tilings=nb_tilings,
            nb_bins=nb_bins,
            env_save_path=experiment_dir,
            results_path=experiment_dir,
            lr_fun=lr_fun,
            exp_fun=exp_fun,
            nb_training_steps=nb_training_steps,
            eval_every=500,
            eval_eps=20,
            save_every=500,
            gamma=gamma,
        )

        with open(os.path.join(experiment_dir, "train_params.yml"), "w") as f:
            f.write(yaml.dump(train_params))

        train_params['env'] = env
        train_params['init_state_func'] = init_func
        train_params['objective_func'] = env.objective
        train_params['policy'] = boltzmann

        ret = train_instance(**train_params)

        with open(os.path.join(experiment_dir, 'training_stats.pkl'), 'wb') as f:
            pkl.dump(ret, f)


if __name__ == '__main__':
    nb_trials = 8
    with Pool(4) as p:
        exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S')
        p.map(run_experiment, [f'{exp_prefix}_{item}' for item in range(nb_trials)])

