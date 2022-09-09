import os
import numpy as np
from datetime import datetime as dt
import yaml
import matplotlib.pyplot as plt
import pickle as pkl

from training_utils import ExponentialDecay, Constant, get_training_utils_yaml_dumper, LinearDecay, StepDecay, circular_initial_state_distribution_2d
from experiments_tile_coding.sarsa import train_instance; algo_name = 'sarsa'
from random_env.envs.random_env_discrete_actions import RandomEnvDiscreteActions as REDA, REDAClip


def run_experiment(exp_name):
    n_obs, n_act = 2, 2

    exp_fun = LinearDecay(1.0, 0.0, 80000, label='EXP')
    lr_fun = Constant(1e-1, label='LR')
    nb_tilings, nb_bins = 16, 2
    gamma = 0.9

    for experiment_name, state_clip in zip(('no-clipping', 'clip-1.0', 'clip-1.2', 'clip-1.5'),
                                    (0.0, 1.0, 1.2, 1.5)):
        env = REDAClip(n_obs, n_act, state_clip)
        experiment_dir = os.path.join(exp_name, experiment_name)
        os.makedirs(experiment_dir)

        train_params = dict(
            n_obs=n_obs,
            n_act=n_act,
            env=env,
            nb_tilings=nb_tilings,
            nb_bins=nb_bins,
            env_save_path=experiment_dir,
            results_path=experiment_dir,
            lr_fun=lr_fun,
            exp_fun=exp_fun,
            nb_training_steps=100000,
            eval_every=500,
            eval_eps=20,
            save_every=500,
            gamma=gamma,
        )

        with open(os.path.join(experiment_dir, "train_params.yml"), "w") as f:
            f.write(yaml.dump(train_params, Dumper=get_training_utils_yaml_dumper()))

        train_params['init_state_func'] = env.reset
        train_params['objective_func'] = env.objective

        errors, ep_lens, iht_counts, lrs, env = train_instance(**train_params)

        with open(os.path.join(experiment_dir, 'training_stats.pkl'), 'wb') as f:
            d = dict(errors=errors, ep_lens=ep_lens, iht_counts=iht_counts)
            pkl.dump(d, f)

from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(8) as p:
        exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S')
        p.map(run_experiment, [f'{exp_prefix}_{item}' for item in range(16)])
        run_experiment(exp_prefix)

