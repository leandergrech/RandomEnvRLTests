import os
import numpy as np
from datetime import datetime as dt
import yaml

from utils.training_utils import LinearDecay, boltzmann, eps_greedy
from linear_q_function import QValueFunctionLinear, FeatureExtractor
from random_env.envs import REDAClipCont, REDAClip
from sarsa import train_instance_early_termination

# experiment_dir = f"ramp_env_size_{dt.now().strftime('%m%d%y_%H%M%S')}"
# experiment_dir = f"ramp_env_size_102322_183706"
experiment_dir = f"ramp_env_size_102322_201202"

nb_training_steps = 100000
eval_every = 100
save_every = 1000
eval_eps = 2
start_eval = 0

explore_until = decay_lr_until = nb_training_steps

exp_fun = LinearDecay(1.0, 1e-2, explore_until, label='EPS')
lr_fun = LinearDecay(1e-1, 1e-2, decay_lr_until, label='LR')
gamma = 0.9

if 'EPS' in exp_fun.label:
    policy = eps_greedy
else:
    policy = boltzmann

env_sz = 20

n_obs = n_act = env_sz
for seed in (678,):#(123, 234, 345, 456, 567):
    sub_experiment_dir = f"{env_sz}obsx{env_sz}act_{seed}seed"
    experiment_path = os.path.join(experiment_dir, sub_experiment_dir)

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    else:
        raise FileExistsError

    np.random.seed(seed=seed)
    env = REDAClipCont(n_obs=n_obs, n_act=n_act, state_clip=1.0)
    eval_env = REDAClip(n_obs=n_obs, n_act=n_act, state_clip=1.0, model_info=env.model_info)
    eval_env.EPISODE_LENGTH_LIMIT = 500

    env.save_dynamics(experiment_path)

    train_params = dict(
        n_obs=n_obs,
        n_act=n_act,
        gamma=gamma,
        nb_training_steps=nb_training_steps,
        exp_fun=exp_fun,
        lr_fun=lr_fun,
        eval_eps=eval_eps,
        eval_every=eval_every,
        save_path=experiment_path,
        save_every=save_every,
        nb_successes_early_termination=5,
        start_eval=start_eval
    )
    with open(os.path.join(experiment_path, "train_params.yml"), "w") as f:
        f.write(yaml.dump(train_params))

    train_params['policy'] = policy
    train_params['env'] = env
    train_params['eval_env'] = eval_env

    timestep = train_instance_early_termination(**train_params)
    with open(os.path.join(experiment_path, 'train_info.md'), 'w') as f:
        f.write(f'{timestep}')




