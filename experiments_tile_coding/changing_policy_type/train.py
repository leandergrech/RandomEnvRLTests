import os
import numpy as np
from datetime import datetime as dt
import yaml
import pickle as pkl
from multiprocessing import Pool

from utils.training_utils import Constant, LinearDecay
from experiments_tile_coding.sarsa import train_instance as sarsa_train; algo_name = 'sarsa'
from random_env.envs.random_env_discrete_actions import REDAClip, get_discrete_actions
from experiments_tile_coding.policy_types import eps_greedy, boltzmann



def run_experiment(exp_name):
    n_obs, n_act = 2, 2
    state_clip = 1.0
    env = REDAClip(n_obs, n_act, state_clip)

    nb_training_steps = 60000
    explore_until = 50000
    eps_fun = LinearDecay(1.0, 1e-1, explore_until, label='EPS')
    tau_fun1 = LinearDecay(1.0, 1e-1, explore_until, label='TAU')
    # tau_fun2 = LinearDecay(5.0, 1e-2, explore_until, label='TAU')
    # tau_fun3 = LinearDecay(10.0, 1e-2, explore_until, label='TAU')
    lr_fun = Constant(1e-1, label='LR')
    nb_tilings, nb_bins = 16, 2
    gamma = 0.9

    # sub_experiments_names = ['eps-greedy', 'boltz-1', 'boltz-5', 'boltz-10']
    sub_experiments_names = ['eps-greedy', 'boltz-1']
    for experiment_name, policy, exp_fun in zip(sub_experiments_names,
                                                (eps_greedy, boltzmann, boltzmann, boltzmann),
                                                # (eps_fun, tau_fun1, tau_fun2, tau_fun3)):
                                                (eps_fun, tau_fun1)):
        experiment_dir = os.path.join(exp_name, experiment_name)
        os.makedirs(experiment_dir)

        train_params = dict(
            n_obs=n_obs,
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
        train_params['init_state_func'] = env.reset
        train_params['objective_func'] = env.objective
        train_params['policy'] = policy

        # Actual training
        ret = sarsa_train(**train_params)

        # Save training data
        with open(os.path.join(experiment_dir, 'training_stats.pkl'), 'wb') as f:
            pkl.dump(ret, f)


if __name__ == '__main__':
    nb_trials = 1
    with Pool(4) as p:
        exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S')
        p.map(run_experiment, [f'{exp_prefix}_{item}' for item in range(nb_trials)])
    # exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S')
    # run_experiment(exp_prefix)

