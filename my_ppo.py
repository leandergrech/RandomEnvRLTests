import os
import shutil
from collections import deque

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime as dt

import numpy as np

from random_env.envs import RandomEnv


class MLP(nn.Module):
	def __init__(self, in_dim, out_dim, hidden_layers):
		super(MLP, self).__init__()

		layer_input_sizes = np.concatenate([[in_dim], hidden_layers])
		layer_output_sizes = np.concatenate([hidden_layers, [out_dim]])
		self.layers = []
		for in_sz, out_sz in zip(layer_input_sizes, layer_output_sizes):
			self.layers.append(nn.Linear(in_sz, out_sz))
		self.layers = t.nn.ModuleList(self.layers)

	def forward(self, obs):
		if not isinstance(obs, t.Tensor):
			obs = t.tensor(obs, dtype=t.float)

		x = obs
		for l in self.layers[:-1]:
			x = F.relu(l(x))

		return self.layers[-1](x)

class PPO:
	def __init__(self, env):
		self._init_hparams()

		# Extract env info
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		# Initialize actor critic
		actor_hidden_layers = [200, 200]
		self.actor = MLP(self.obs_dim, self.act_dim, actor_hidden_layers)
		self.critic = MLP(self.obs_dim, 1, [200, 100])
		# self.critic2 = MLP(self.obs_dim, 1, [64, 64])

		# Create actor covariance matrix
		self.cov_var = t.full(size=(self.act_dim,), fill_value=0.1) # 0.5 arbitrary
		self.cov_mat = t.diag(self.cov_var)

		# Initialize optimizers
		self.actor_optim = self.actor_optim_type(self.actor.parameters(), lr=self.lr_actor)
		self.critic_optim = self.critic_optim_type(self.critic.parameters(), lr=self.lr_critic)
		# self.critic_optim = self.critic_optim_type(list(self.critic.parameters()) + list(self.critic2.parameters()), lr=self.lr_critic)

		self._callback = None
		self.num_timesteps = 0
		self.writer = None

	def learn(self, total_timesteps, callback, log_dir=None):
		total_timesteps = int(total_timesteps) # just in case

		# Initialize tensorboard writer
		self.writer = SummaryWriter(log_dir)

		# Initialize callback
		self._callback = callback
		self._callback.init_callback(self, self.writer)

		# Logging variables
		actor_losses, critic_losses = [], []

		# Training loop
		t_so_far = 0
		while t_so_far < total_timesteps:
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# V_{phi, k}
			V, _ = self.evaluate(batch_obs, batch_acts)

			# Calculate advantage
			A_k = batch_rtgs - V.detach()
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # standardization

			# Optimization loop
			for _ in range(self.n_updates_per_iteration):
				# Calculate pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate policy ratios
				ratios = t.exp(curr_log_probs - batch_log_probs)

				# Calulate surrogate losses
				surr1 = ratios * A_k
				surr2 = t.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor critic losses
				# Note: minimizing neg objective is equivalent maximizing objective
				actor_loss = (-t.min(surr1, surr2)).mean()
				critic_loss = t.nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backprop for actor
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True) # so we can call backprop twice
				self.actor_optim.step()


				# Calculate gradients and perform backprop for critic
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Collect data for logging
				actor_losses.append(actor_loss.item())
				critic_losses.append(critic_loss.item())


			actor_loss = np.mean(actor_losses)
			critic_loss = np.mean(critic_losses)
			actor_losses, critic_losses = [], []
			self.writer.add_scalar('train/actor_loss', actor_loss, self.num_timesteps)
			self.writer.add_scalar('train/critic_loss', critic_loss, self.num_timesteps)
			self.writer.add_scalar('train/learning_rate_critic', self.lr_critic, self.num_timesteps)
			self.writer.add_scalar('train/learning_rate_actor', self.lr_actor, self.num_timesteps)

			self.update_learning_rate()

	def rollout(self):
		# Batch data
		batch_obs = []  # batch observations
		batch_acts = []  # batch actions
		batch_log_probs = []  # log probs of each action
		batch_rews = []  # batch rewards
		batch_rtgs = []  # batch rewards-to-go
		batch_lens = []  # episodic lengths in batch

		t_so_far = 0
		while t_so_far < self.timesteps_per_batch:
			o = self.env.reset()
			d = False
			ep_rews = []
			for ep_t in range(self.max_timesteps_per_episode):
				self.num_timesteps += 1
				t_so_far += 1
				batch_obs.append(o)

				a, log_prob = self.get_action(o)
				o, r, d, _ = self.env.step(a)

				self._callback.on_step()

				ep_rews.append(r)
				batch_acts.append(a)
				batch_log_probs.append(log_prob)

				if d: break

			# collect episodic data
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		batch_obs = t.tensor(batch_obs, dtype=t.float)
		batch_acts = t.tensor(batch_acts, dtype=t.float)
		batch_log_probs = t.tensor(batch_log_probs, dtype=t.float)

		batch_rtgs = self.compute_rtgs(batch_rews)

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def get_action(self, o):
		mean = self.actor(o)
		dist = MultivariateNormal(mean, self.cov_mat)

		a = dist.sample()
		log_prob = dist.log_prob(a)

		return a.detach().numpy(), log_prob.detach()

	def predict(self,o, deterministic=True):
		if deterministic:
			return self.actor(o).detach().numpy()
		else:
			return self.get_action(o)

	def compute_rtgs(self, batch_rews):
		batch_rtgs = []
		for ep_rews in reversed(batch_rews): # reverse maintains order on append
			discounted_r = 0
			for r in reversed(ep_rews):
				discounted_r = r + discounted_r * self.gamma
				batch_rtgs.insert(0, discounted_r)

		batch_rtgs = t.tensor(batch_rtgs, dtype=t.float)
		return batch_rtgs

	def evaluate(self, batch_obs, batch_acts):
		V = self.critic(batch_obs).squeeze()
		# V1 = self.critic(batch_obs).squeeze()
		# V2 = self.critic2(batch_obs).squeeze()
		# V = t.minimum(V1, V2)

		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		return V, log_probs

	def _init_hparams(self):
		# Data collection
		self.timesteps_per_batch = 2000
		self.max_timesteps_per_episode = 100

		# Return
		self.gamma = 0.99

		# Optimization
		self.actor_optim_type = SGD
		self.critic_optim_type = Adam
		self.n_updates_per_iteration = 10
		self.clip = 0.1
		self.lr_actor = self.lr_init_actor = 2.5e-4
		self.lr_halflife_actor = int(200e3)
		self.lr_critic = 2.5e-4

		# Logging
		self.logging_freq = 10

	def update_learning_rate(self):
		# Update exponential decay of actor lr
		self.lr_actor = self.lr_init_actor * np.exp(-(1/self.lr_halflife_actor) * self.num_timesteps)

		for optim, lr in zip((self.actor_optim, self.critic_optim), (self.lr_actor, self.lr_critic)):
			for g in self.actor_optim.param_groups:
				g['lr'] = lr

class EvalCheckptEarlyStopTrainingCallback():
	MAX_EPS = 20  # Number of evaluation episodes to run the latest policy
	END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES = 5  # self-explanatory
	SUCCESS_THRESHOLD = 100.0  # Any mean success rate during evaluation greater than this is considered total success

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
		self.verbose = True

	def quick_save(self, suffix=None):
		if suffix is None:
			save_path = os.path.join(self.save_dir, f'{self.model_name}_{self.num_timesteps}_steps')
		else:
			save_path = os.path.join(self.save_dir, f'{self.model_name}_{suffix}')

		if self.verbose > 1:
			print("Saving model checkpoint to {}".format(save_path))

		# self.model.save(save_path)
		if self.verbose > 0:
			print(f'Model NOT saved to: {save_path}')

	def init_callback(self, model, writer):
		self.model = model
		self.writer = writer

	def on_step(self):
		self.num_timesteps = self.model.num_timesteps

		if self.num_timesteps % self.EVAL_FREQ == 0:
			if self.verbose:
				print(f'-> Training step: {self.num_timesteps}')
			returns = []
			ep_lens = []
			success = []
			rew_final_neg_init = []
			expected_rew_per_step = []

			observations = []
			actions = []
			trims = []

			### START OF EPISODE LOOP ###
			for ep in range(self.MAX_EPS):
				ep_rewards = []

				o = self.env.reset()
				observations.append(o)

				ep_return = 0.0
				step = 0
				d = False
				while not d:
					a = self.model.predict(o, deterministic=True)
					step += 1

					o2, r, d, _ = self.env.step(a)

					observations.append(np.copy(o2))
					actions.append(a)
					trims.append(o - o2)

					o = o2
					ep_return += r

					ep_rewards.append(r)

				ep_lens.append(step)
				returns.append(ep_return)
				rew_final_neg_init.append(ep_rewards[-1] - ep_rewards[0])
				expected_rew_per_step.append(ep_return / step)
				if step < self.env.max_steps:
					success.append(1.0)
				else:
					success.append(0.0)
			### END OF EPISODE LOOP ###

			returns = np.mean(returns)
			ep_lens = np.mean(ep_lens)
			success = np.mean(success) * 100.0
			rew_final_neg_init = np.mean(rew_final_neg_init)
			expected_rew_per_step = np.mean(expected_rew_per_step)

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

			if self.verbose:
				print(f'\t-> Returns: {returns:.2f}')
				print(f'\t-> Episode length: {ep_lens:.2f}')
				print(f'\t-> Success rate: {success:.2f}')
				print(f'\t-> Rew. final - init: {rew_final_neg_init:.5f}')
				print(f'\t-> Expected rew. per step" {expected_rew_per_step}')
				print(f'\t-> Obs.  \u03BC = {obs_mean:.4f}, \u03C3 = {obs_std:.4f}')
				print(f'\t-> Act.  \u03BC = {act_mean:.4f}, \u03C3 = {act_std:.4f}')
				print(f'\t-> Trim. \u03BC = {trim_mean:.4f}, \u03C3 = {trim_std:.4f}')
				print(f'\t-> FPS = {fps:.2f}')

			for tag, val in zip(('eval/episode_return', 'eval/episode_length', 'eval/success', 'eval/rew_final_neg_init',
			                     'eval/expected_rew_per_step',
			                     'spaces/obs_mean', 'spaces/obs_std', 'spaces/act_mean', 'spaces/act_std',
			                     'spaces/trim_mean', 'spaces/trim_std',
			                     'train/fps'),
			                    (returns, ep_lens, success, rew_final_neg_init, expected_rew_per_step,
			                     obs_mean, obs_std, act_mean, act_std, trim_mean, trim_std, fps)):
				self.writer.add_scalar(tag, val, self.num_timesteps)
			# self.logger.dump(self.num_timesteps)
			### SAVE SUCCESSFUL AGENTS ###
			if success > 0:
				self.quick_save()
				if success >= EvalCheckptEarlyStopTrainingCallback.SUCCESS_THRESHOLD:
					self.successes.append(1)
				else:
					self.successes.clear()

				if len(self.successes) >= EvalCheckptEarlyStopTrainingCallback.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES:
					return False  # End training

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

def train_my_ppo():
	nb_training_steps = 5e5

	env_sz = 5
	env = RandomEnv(env_sz, env_sz, estimate_scaling=True)
	eval_env = RandomEnv(env_sz, env_sz, model_info=env.model_info)

	par_dir = 'testing_my_ppo'
	save_name = dt.strftime(dt.now(), '%m%d%y_%H%M%S')
	save_dir = os.path.join(par_dir, 'saves', save_name)
	log_dir = os.path.join(par_dir, 'logs', save_name)

	for path in (save_dir, log_dir):
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)

	callback = EvalCheckptEarlyStopTrainingCallback(eval_env, save_dir)
	agent = PPO(env)
	agent.learn(nb_training_steps, callback, log_dir)

if __name__ == '__main__':
    train_my_ppo()