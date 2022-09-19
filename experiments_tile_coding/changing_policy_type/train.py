import os
import numpy as np
from datetime import datetime as dt
import yaml
import matplotlib.pyplot as plt
import pickle as pkl

from training_utils import ExponentialDecay, Constant, get_training_utils_yaml_dumper, LinearDecay, StepDecay, circular_initial_state_distribution_2d
from experiments_tile_coding.sarsa import train_instance; algo_name = 'sarsa'
from random_env.envs.random_env_discrete_actions import RandomEnvDiscreteActions as REDA, REDAClip, get_discrete_actions


def run_experiment(exp_name):
    n_obs, n_act = 2, 2
    state_clip = 0.0
    env = REDAClip(n_obs, n_act, state_clip)
    nb_actions = len(get_discrete_actions(n_act, 3))

    eps_fun = LinearDecay(1.0, 0.0, 80000, label='EPS')
    tau_fun1 = LinearDecay(10.0, 1e-2, 80000, label='TAU')
    tau_fun2 = LinearDecay(5.0, 1e-2, 80000, label='TAU')
    tau_fun3 = LinearDecay(1.0, 1e-2, 80000, label='TAU')
    lr_fun = Constant(1e-1, label='LR')
    nb_tilings, nb_bins = 16, 2
    gamma = 0.9

    def eps_greedy(s, q, epsilon):
        nonlocal nb_actions
        if np.random.rand() < epsilon:
            return np.random.choice(nb_actions)
        else:
            return q.greedy_action(s)

    def boltzmann(s, q, tau):
        nonlocal nb_actions
        qvals_exp = np.exp([q.value(s, a_)/tau for a_ in range(nb_actions)])
        qvals_exp_sum = np.sum(qvals_exp)

        cum_probas = np.cumsum(qvals_exp/qvals_exp_sum)
        return np.searchsorted(cum_probas, np.random.rand())

    sub_experiments_names = ['eps-greedy', 'boltz-10', 'boltz-5', 'boltz-1']
    for experiment_name, policy, exp_fun in zip(sub_experiments_names,
                                                (eps_greedy, boltzmann, boltzmann, boltzmann),
                                                (eps_fun, tau_fun1, tau_fun2, tau_fun3)):
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
        train_params['policy'] = policy

        iht_counts, ep_lens, returns = train_instance(**train_params)

        with open(os.path.join(experiment_dir, 'training_stats.pkl'), 'wb') as f:
            d = dict(iht_counts=iht_counts, ep_lens=ep_lens, returns=returns)
            pkl.dump(d, f)


from multiprocessing import Pool
if __name__ == '__main__':
    nb_trials = 1
    with Pool(8) as p:
        exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S')
        p.map(run_experiment, [f'{exp_prefix}_{item}' for item in range(nb_trials)])
    # exp_prefix = dt.now().strftime(f'{algo_name}_%m%d%y_%H%M%S')
    # run_experiment(exp_prefix)
