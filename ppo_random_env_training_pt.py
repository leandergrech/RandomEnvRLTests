import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from itertools import product
import numpy as np
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime as dt
import torch as t

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from random_env.envs import RandomEnv, RunningStats


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EvalCheckptEarlyStopTrainingCallback(BaseCallback):
	MAX_EPS = 20 # Number of evaluation episodes to run the latest policy
	END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES = 5 # self-explanatory
	SUCCESS_THRESHOLD = 100.0 # Any mean success rate during evaluation greater than this is considered total success

	def __init__(self, env, save_dir, EVAL_FREQ=100, CHKPT_FREQ=1000):
		"""
		This callback automatically ends training after
		:param env:
		:param save_dir:
		:param EVAL_FREQ:
		:param CHKPT_FREQ:
		"""
		self.env = env
		self.save_dir = save_dir
		self.model_name = os.path.split(save_dir)[-1]

		self.last_call_time = None
		self.last_call_step = None

		self.EVAL_FREQ = EVAL_FREQ
		self.CHKPT_FREQ = CHKPT_FREQ

		self.current_best_model_ep_len = self.env.EPISODE_LENGTH_LIMIT
		self.current_best_model_save_dir = ''

		self.gamma = 0.99
		self.discounts = [np.power(self.gamma, i) for i in range(self.env.EPISODE_LENGTH_LIMIT)]
		self.successes = deque(maxlen=EvalCheckptEarlyStopTrainingCallback.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES)
		super(EvalCheckptEarlyStopTrainingCallback, self).__init__()

		self.steps_since_last_animation = 0

	def quick_save(self, suffix=None):
		if suffix is None:
			save_path = os.path.join(self.save_dir, f'{self.model_name}_{self.num_timesteps}_steps')
		else:
			save_path = os.path.join(self.save_dir, f'{self.model_name}_{suffix}')

		if self.verbose > 1:
			print("Saving model checkpoint to {}".format(save_path))

		self.model.save(save_path)
		if self.verbose > 0:
			print(f'Model saved to: {save_path}')

	def on_step(self) -> bool:
		self.num_timesteps = self.model.num_timesteps
		return self._on_step()

	def _on_step(self):
		if self.num_timesteps % self.EVAL_FREQ == 0:
			verbose = True
			if verbose:
				print(f'-> Training step: {self.num_timesteps}')
			returns = []
			ep_lens = []
			success = []

			observations = []
			actions = []
			trims = []

			### START OF EPISODE LOOP ###
			for ep in range(self.MAX_EPS):
				o = self.env.reset()
				observations.append(o)

				ep_return = 0.0
				step = 0
				d = False
				while not d:
					a = self.model.predict(o, deterministic=True)[0]
					step += 1

					o2, r, d, _ = self.env.step(a)

					observations.append(np.copy(o2))
					actions.append(a)
					trims.append(o - o2)

					o = o2
					ep_return += r

				ep_lens.append(step)
				returns.append(ep_return)
				if step < self.env.max_steps:
					success.append(1.0)
				else:
					success.append(0.0)
			### END OF EPISODE LOOP ###

			returns = np.mean(returns)
			ep_lens = np.mean(ep_lens)
			success = np.mean(success) * 100.0
			obs_mean = np.mean(observations)
			obs_std = np.mean(np.std(observations, axis=0))
			act_mean = np.mean(actions)
			act_std = np.mean(np.std(actions, axis=0))
			trim_mean = np.mean(trims)
			trim_std = np.mean(np.std(trims, axis=0))

			if self.last_call_time is None:
				self.last_call_time = dt.now()
				self.last_call_step = 0
				fps = 0
			else:
				this_call_time = dt.now()
				time_bet_calls = this_call_time - self.last_call_time
				steps_bet_calls = self.num_timesteps - self.last_call_step
				fps = steps_bet_calls / time_bet_calls.total_seconds()

				self.last_call_time = this_call_time
				self.last_call_step = self.num_timesteps


			if verbose:
				print(f'\t-> Returns: {returns:.2f}')
				print(f'\t-> Episode length: {ep_lens:.2f}')
				print(f'\t-> Success rate: {success:.2f}')
				print(f'\t-> Obs.  \u03BC = {obs_mean:.4f}, \u03C3 = {obs_std:.4f}')
				print(f'\t-> Act.  \u03BC = {act_mean:.4f}, \u03C3 = {act_std:.4f}')
				print(f'\t-> Trim. \u03BC = {trim_mean:.4f}, \u03C3 = {trim_std:.4f}')
				print(f'\t-> FPS = {fps:.2f}')

			for tag, val in zip(('eval/episode_return', 'eval/episode_length', 'eval/success',
			                     'spaces/obs_mean', 'spaces/obs_std', 'spaces/act_mean', 'spaces/act_std',
			                     'spaces/trim_mean', 'spaces/trim_std', 'fps'),
			                    (returns, ep_lens, success, obs_mean, obs_std, act_mean, act_std,
			                     trim_mean, trim_std, fps)):
				self.logger.record(tag, val)
			self.logger.dump(self.num_timesteps)
			### SAVE SUCCESSFUL AGENTS ###
			if success > 0:
				self.quick_save()
				if success >= EvalCheckptEarlyStopTrainingCallback.SUCCESS_THRESHOLD:
					self.successes.append(1)
				else:
					self.successes.clear()
					
				if len(self.successes) >= EvalCheckptEarlyStopTrainingCallback.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES:
					return False # End training

		if self.num_timesteps % self.CHKPT_FREQ == 0:
			self.quick_save()

		# self.steps_since_last_animation += 1
		# if self.steps_since_last_animation > 1000:
		# 	self.steps_since_last_animation = 0
		# 	self.animation()

		return True

	def animation(self):
		fig, (ax1, ax2) = plt.subplots(2)
		fig.suptitle(f'Evaluation at training step: {self.num_timesteps}')
		plt.ion()

		o1 = self.env.reset()

		for ax in fig.axes:
			ax.axhline(0.0, color='k', ls='--')
			ax.grid(which='both', color='gray')

		oline, = ax1.plot(o1)
		ax1.set_title('Observation')

		aline, = ax2.plot(np.zeros(self.env.act_dimension))
		ax2.set_title('Action')

		fig.tight_layout()

		d = False
		new_ylim = lambda ax, ydat: (np.min(np.concatenate([ax.get_ylim(), ydat])),
		                             np.max(np.concatenate([ax.get_ylim(), ydat])))
		while not d:
			a = self.model.predict(o1, deterministic=True)[0]
			o2, r, d, _ = self.env.step(a)

			oline.set_ydata(np.copy(o2))
			aline.set_ydata(np.copy(a))

			ax1.set_ylim(new_ylim(ax1, o2))
			ax2.set_ylim(new_ylim(ax2, a))
			plt.pause(0.01)
			o1 = o2
		plt.pause(2)
		plt.close()


DEFAULT_PPO_PARAMS = dict(
			learning_rate = 1e-5,#2.5e-4,
			n_steps = 500,
			batch_size=64,#150,
			n_epochs=10,
			gamma = 0.99,
			gae_lambda = 0.95,
			clip_range = 0.1,
			clip_range_vf = None,
			ent_coef = 0.01,
			vf_coef = 0.5,
			max_grad_norm = 0.5,
			policy_kwargs = {'net_arch': [100, 100]},
			_init_setup_model=True,
			verbose=0
		)

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
ppo_params = DEFAULT_PPO_PARAMS.copy()
# ppo_params.update(hparam_tuple)

par_dir = os.path.join('ppo_random_env_results', dt.strftime(dt.now(), 'training_session_pt_%m%d%y_%H%M%S'))
if not os.path.exists(par_dir):
	os.makedirs(par_dir)

NB_STEPS = int(3e5)
EVAL_FREQ = 100
CHKPT_FREQ = 1000

# for env_sz in (5,):# 15, 20):
env_sz = 10
N_OBS = env_sz
N_ACT = env_sz

for RANDOM_SEED in (123, 234, 345, 456, 567):
	# RANDOM_SEED = 123
	t.manual_seed(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)
	ppo_params['seed'] = RANDOM_SEED

	'''Create environments with same dynamics, duh'''
	# env = RandomEnv(n_obs=N_OBS, n_act=N_ACT, estimate_scaling=True, seed=RANDOM_SEED)
	# eval_env = RandomEnv(n_obs=N_OBS, n_act=N_ACT, estimate_scaling=False, model_info=env.model_info)
	env = RandomEnv(N_OBS, N_ACT, estimate_scaling=False)
	env.load_dynamics('H:\Code\RandomEnvRLTests')
	eval_env = RandomEnv(N_OBS, N_ACT, model_info=env.model_info)


	'''Name agent (model) and create sub dirs required for logging and saving'''
	model_name = f'PPO_' + repr(env) + f'_seed{RANDOM_SEED}'
	log_dir = os.path.join(par_dir, 'logs')
	save_dir = os.path.join(par_dir,  model_name)
	for path in (log_dir, save_dir):
		if not os.path.exists(path):
			os.makedirs(path)

	'''Save env dynamics'''
	env.save_dynamics(save_dir)


	'''PPO agent'''
	model = PPO('MlpPolicy', env, **ppo_params,
				 tensorboard_log=log_dir)
	print(f'-> Policy nb. of parameters: {count_parameters(model.policy)}')

	'''Callback evaluated agent every EVAL_FREQ steps and saved best model, ant auto-saves every CHKPT_FREQ steps'''
	eval_callback = EvalCheckptEarlyStopTrainingCallback(eval_env, save_dir=save_dir,
	                                                     EVAL_FREQ=EVAL_FREQ, CHKPT_FREQ=CHKPT_FREQ)

	'''Log some more info and save it in the same directory as the agent'''
	with open(os.path.join(save_dir, 'info.txt'), 'w') as f:
		'''Log PPO hyperparameters used in this experiment'''
		f.write('-> PPO parameters\n')
		for k, v in ppo_params.items():
			f.write(f'\t-> {k} = {v}\n')
		'''Log some more info about env. machanisms'''
		f.write(f'-> Info: Environment estimated scaling:\n\t{eval_env.trim_stats}\n')
		# f.write(f'-> Info: Environment model output normalised & scaled by:\n\t{env.TRIM_FACTOR}\n')
		f.write(f'-> Info: Environment model output standardised & scaled by:\n\t{env.TRIM_FACTOR}\n')
		f.write(f'-> Info: Environment changed objective from -mean(square()) to -sqrt(mean(square))) [RMS]\n')
		f.write(f'-> Info: Environment added reward scaling by: \n\t{env.REWARD_SCALE}\n')
		f.write(f'-> Info: Policy network nb. of trainable parameters: \n\t{count_parameters(model.policy)}')
		# f.write(f'-> Hyperparameter search Run #{hparam_i}\n')
		# f.write(f'-> Environment dynamics changed to an eye matrix, action=state')

	'''Start training'''
	model.learn(total_timesteps=NB_STEPS, log_interval=300, reset_num_timesteps=True,
	            tb_log_name=model_name, callback=eval_callback, eval_freq=1000)


