import os
import numpy as np
from datetime import datetime as dt
import comet_ml
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch as t

from stable_baselines3 import PPO
from sb3_contrib import TRPO
from random_env.envs import RandomEnv, RunningStats
from my_agents_utils import EvalCheckpointEarlyStopTrainingCallback, make_path_exist, get_writer, count_parameters

COMET_WORKSPACE = 'testing-ppo-trpo'
COMMON_ENV_DIR = 'common_envs'
algo = 'TRPO'

if 'PPO' in algo:
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
	DEFAULT_PARAMS = dict(
		policy='MlpPolicy',
		learning_rate=3e-4,
		n_steps=2048,
		batch_size=64,
		n_epochs=10,
		gamma=0.99,
		gae_lambda=0.95,
		clip_range=0.2,
		clip_range_vf=None,
		normalize_advantage=True,
		ent_coef=0.0,
		vf_coef=0.5,
		max_grad_norm=0.5,
		use_sde=False,
		sde_sample_freq=- 1,
		target_kl=None,
		tensorboard_log=None,
		create_eval_env=False,
		policy_kwargs=None,
		verbose=0,
		seed=None,
		device='auto',
		_init_setup_model=True)
elif 'TRPO' in algo:
	DEFAULT_PARAMS = dict(
		policy='MlpPolicy',
		learning_rate=0.001,
		n_steps=2048,
		batch_size=128,
		gamma=0.99,
		cg_max_steps=15,
		cg_damping=0.1,
		line_search_shrinking_factor=0.8,
		line_search_max_iter=10,
		n_critic_updates=10,
		gae_lambda=0.95,
		use_sde=False,
		sde_sample_freq=- 1,
		normalize_advantage=True,
		target_kl=0.01,
		sub_sampling_factor=1,
		tensorboard_log=None,
		create_eval_env=False,
		policy_kwargs=None,
		verbose=0,
		seed=None,
		device='auto',
		_init_setup_model=True
	)


NB_STEPS = int(5e5)
EVAL_FREQ = 100
CHKPT_FREQ = 10000

env_sz = 10
N_OBS = env_sz
N_ACT = env_sz

params = DEFAULT_PARAMS.copy()
session_name = dt.strftime(dt.now(), 'training_session_%m%d%y_%H%M%S')
par_dir = os.path.join('sb3_randomenv_training', session_name)
save_dir = os.path.join(par_dir, 'saves')
log_dir = os.path.join(par_dir, 'logs')
for d in (save_dir, log_dir):
	make_path_exist(d)

for RANDOM_SEED in (123,):#, 234, 345, 456, 567):
	'''Set seed to random generators'''
	t.manual_seed(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)
	params['seed'] = RANDOM_SEED

	'''Create environments with same dynamics, duh'''
	env = RandomEnv(N_OBS, N_ACT, estimate_scaling=False)

	'''Reload dynamics already created for this size RE or save new RE dynamics for this size (size=obs_spacexact_space)'''
	try:
		env.load_dynamics(COMMON_ENV_DIR)
	except:
		env.save_dynamics(COMMON_ENV_DIR)
	eval_env = RandomEnv(N_OBS, N_ACT, model_info=env.model_info)
	params['env'] = env

	'''Name agent (model) and create sub dir required for saving'''
	model_name = f'{algo}_' + repr(env) + f'_seed{RANDOM_SEED}'

	'''Agent'''
	if 'PPO' in algo:
		model = PPO(**params)
	elif 'TRPO' in algo:
		model = TRPO(**params)
	# print(f'-> Policy nb. of parameters: {count_parameters(model.actor)}')

	'''Callback evaluated agent every EVAL_FREQ steps and saved best model, and auto-saves every CHKPT_FREQ steps'''
	eval_callback = EvalCheckpointEarlyStopTrainingCallback(env=eval_env, save_dir=save_dir,
															EVAL_FREQ=EVAL_FREQ, CHKPT_FREQ=CHKPT_FREQ)
	writer = get_writer(model_name, session_name, COMET_WORKSPACE)
	eval_callback.init_callback(model, writer)

	writer.log_parameters(params)

	'''Log some more info and save it in the same directory as the agent'''
	with open(os.path.join(save_dir, 'info.txt'), 'w') as f:
		'''Log hyperparameters used in this experiment'''
		f.write(f'-> {algo} parameters\n')
		for k, v in params.items():
			f.write(f'\t-> {k} = {v}\n')

	'''Start training'''
	model.learn(total_timesteps=NB_STEPS, callback=eval_callback)


