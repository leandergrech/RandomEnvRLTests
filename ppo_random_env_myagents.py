import os
import numpy as np
from datetime import datetime as dt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch as t

from my_agents import PPO
from random_env.envs import RandomEnv, RunningStats
from my_agents_utils import EvalCheckpointEarlyStopTrainingCallback, make_path_exist, get_writer, count_parameters

COMET_WORKSPACE = 'testing-ppo-trpo'

COMMON_ENV_DIR = 'common_envs'
DEFAULT_PPO_PARAMS = dict(
	# Network structures
	actor_hidden_layers=[200, 200],
	critic_hidden_layers=[200, 100],
	# Data collection
	timesteps_per_batch=2000,
	max_timesteps_per_episode=RandomEnv.EPISODE_LENGTH_LIMIT,
	# Return
	gamma=0.99,
	# Optimization
	actor_optim_type=t.optim.SGD,
	critic_optim_type=t.optim.Adam,
	n_updates_per_iteration=10,
	clip=0.2,
	lr_actor=1e-2,
	lr_halflife_actor=int(1e5),
	lr_critic=1e-3)

# '''
# HYPERPARAMETER GRID SEARCH PPO ON 5X5
# '''
# hparam_search_dict = dict(ent_coef=(0.01, 0.0),
#                           vf_coef=(0.5, 1.0),
#                           learning_rate=(1e-4, 1e-5),
#                           n_steps=(100, 500),
#                           batch_size=(64, 256))
# keys, values = [], []
# for k, v in hparam_search_dict.items():
# 	keys.append(k)
# 	values.append(v)
# hparam_set = [{k: v for k, v in zip(keys, htuple)} for htuple in product(*values)]

# for hparam_i, hparam_tuple in enumerate(hparam_set):
# ppo_params.update(hparam_tuple)

NB_STEPS = int(5e5)
EVAL_FREQ = 100
CHKPT_FREQ = 10000

env_sz = 10
N_OBS = env_sz
N_ACT = env_sz

ppo_params = DEFAULT_PPO_PARAMS.copy()
session_name = dt.strftime(dt.now(), 'training_session_%m%d%y_%H%M%S')
par_dir = os.path.join('myagents_randomenv_training', session_name)
save_dir = os.path.join(par_dir, 'saves')
log_dir = os.path.join(par_dir, 'logs')
for d in (save_dir, log_dir):
	make_path_exist(d)

for RANDOM_SEED in (123, 234, 345, 456, 567):
	'''Set seed to random generators'''
	t.manual_seed(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)

	'''Create environments with same dynamics, duh'''
	env = RandomEnv(N_OBS, N_ACT, estimate_scaling=False)
	eval_env = RandomEnv(N_OBS, N_ACT, model_info=env.model_info)

	'''Reload dynamics already created for this size RE or save new RE dynamics for this size (size=obs_spacexact_space)'''
	try:
		env.load_dynamics(COMMON_ENV_DIR)
	except:
		env.save_dynamics(COMMON_ENV_DIR)

	'''Name agent (model) and create sub dir required for saving'''
	model_name = f'PPO_' + repr(env) + f'_seed{RANDOM_SEED}'

	'''PPO agent'''
	model = PPO(env, **ppo_params)
	print(f'-> Policy nb. of parameters: {count_parameters(model.actor)}')

	'''Callback evaluated agent every EVAL_FREQ steps and saved best model, and auto-saves every CHKPT_FREQ steps'''
	eval_callback = EvalCheckpointEarlyStopTrainingCallback(env=eval_env, save_dir=save_dir,
															EVAL_FREQ=EVAL_FREQ, CHKPT_FREQ=CHKPT_FREQ)

	'''Log some more info and save it in the same directory as the agent'''
	with open(os.path.join(save_dir, 'info.txt'), 'w') as f:
		'''Log PPO hyperparameters used in this experiment'''
		f.write('-> PPO parameters\n')
		for k, v in ppo_params.items():
			f.write(f'\t-> {k} = {v}\n')

	writer = get_writer(model_name, session_name, COMET_WORKSPACE)
	'''Start training'''
	model.learn(total_timesteps=NB_STEPS, callback=eval_callback, writer=writer)
