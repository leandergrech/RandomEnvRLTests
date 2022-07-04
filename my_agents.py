import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime as dt
import numpy as np

from random_env.envs import RandomEnv
import my_agents_utils as utils

import torch as t
from torch.optim import Adam, SGD
from torch.distributions import MultivariateNormal
from torch.autograd import grad

from collections import namedtuple

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', ])

COMET_WORKSPACE = 'testing_ppo_trpo'


class OnPolicy(ABC):
    def __init__(self, env, **kwargs):
        self.env = env
        self._init_hparams(**kwargs)

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self._callback = None
        self.num_timesteps = 0
        self.writer = None

    @abstractmethod
    def _init_hparams(self, **kwargs):
        pass

    @abstractmethod
    def get_hparams(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback, writer=None):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def predict(self, o, deterministic=True):
        pass

    def rollout(self, *args, **kwargs):
        """
        Return a lists of s, a, r, s', log_probs,
        :param args:
        :param kwargs:
        :return:
        """
        # Batch data
        batch_size = args[0]
        ep_max_size = args[1]

        K = ep_max_size
        B = batch_size // K

        # batch_obs = t.zeros(B * K * self.obs_dim)  # batch observations
        # batch_acts = t.zeros(B * K * self.obs_dim)  # batch actions
        # batch_log_probs = t.zeros(B * K)  # log probs of each action
        # batch_rews = t.zeros_like(batch_log_probs)  # batch rewards
        # batch_lens = t.zeros(B)  # episodic lengths in batch
        # batch_rtgs
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_rtgs = []

        for b in range(B):
            o = self.env.reset()
            for k in range(K):
                a, log_prob = self.predict(o, deterministic=False)
                a = a.detach().numpy().tolist()
                otp1, r, d, _ = self.env.step(a)
                if not self._callback.on_step():
                    return None
                self.num_timesteps += 1
                batch_obs.extend(o.tolist())
                batch_acts.extend(a)
                batch_rews.append(r)
                batch_log_probs.append(log_prob)
                o = otp1
                if d:
                    break
            batch_lens.append(k + 1)
            batch_rtgs.extend(self.compute_rtg_traj(batch_rews[-(k + 1):]))
        batch_obs = t.tensor(batch_obs)
        batch_acts = t.tensor(batch_acts)
        batch_rews = t.tensor(batch_rews)
        batch_log_probs = t.tensor(batch_log_probs)
        batch_rtgs = t.tensor(batch_rtgs)
        batch_lens = t.tensor(batch_lens)
        return batch_obs, batch_acts, batch_rews, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtg_traj(self, r_traj, discount=None):
        if discount is None: discount = self.gamma
        k = len(r_traj)
        pows = t.pow(discount, t.arange(k))
        discounted_rews = pows * t.cat([t.tensor(r_traj[1:]), t.tensor([0.0])], dim=-1)
        return t.cumsum(discounted_rews.flip(-1), dim=-1).flip(-1) / pows

    def compute_rtgs(self, batch_rews, discount=None):
        if discount is None: discount = self.gamma
        b, k = batch_rews.shape
        pows = t.pow(discount, t.arange(k))
        discounted_rews = pows * t.cat([batch_rews[:, 1:], t.zeros(b, 1)], dim=-1)
        batch_rtgs = t.cumsum(discounted_rews.flip(-1), dim=-1).flip(-1) / pows
        return batch_rtgs


class PPO(OnPolicy):
    def __init__(self, env, **kwargs):
        self.__name = None
        super(PPO, self).__init__(env, **kwargs)
        self.env = env
        # Initialize actor critic
        self.actor = utils.MLP(self.obs_dim, self.act_dim, self.actor_hidden_layers)
        self.critic = utils.MLP(self.obs_dim, 1, self.critic_hidden_layers)

        # Create actor covariance matrix
        self.cov_var = t.full(size=(self.act_dim,), fill_value=0.5)  # 0.5 arbitrary
        self.cov_mat = t.diag(self.cov_var)

        # Initialize optimizers
        self.actor_optim = self.actor_optim_type(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = self.critic_optim_type(self.critic.parameters(), lr=self.lr_critic)

    def __repr__(self):
        if self.__name is None:
            self.__name = f'PPO_{repr(self.env)}'
        return self.__name

    def _init_hparams(self, **kw):
        # Network structures
        self.actor_hidden_layers = kw.get('actor_hidden_layers', [200, 200])
        self.critic_hidden_layers = kw.get('critic_hidden_layers', [200, 100])

        # Data collection
        self.timesteps_per_batch = kw.get('timesteps_per_batch', 2000)
        self.max_timesteps_per_episode = kw.get('max_timesteps_per_episode', 100)

        # Return
        self.gamma = kw.get('gamma', 0.99)

        # Optimization
        self.actor_optim_type = kw.get('actor_optim_type', SGD)
        self.critic_optim_type = kw.get('critic_optim_type', Adam)
        self.n_updates_per_iteration = kw.get('n_updates_per_iteration', 10)
        self.clip = kw.get('clip', 0.2)
        self.lr_actor = kw.get('lr_actor', 1e-2)
        self.lr_init_actor = self.lr_actor
        self.lr_halflife_actor = kw.get('lr_halflife_actor', int(100e3))
        self.lr_critic = kw.get('lr_critic', 1e-3)

        # Evaluation
        self.save_agent_every = kw.get('save_agent_every', 1000)
        self.eval_agent_every = kw.get('eval_agent_every', 500)

    def get_hparams(self):
        return dict(
            actor_hidden_layers=self.actor_hidden_layers,
            critic_hidden_layers=self.critic_hidden_layers,
            timesteps_per_batch=self.timesteps_per_batch,
            max_timesteps_per_episode=self.max_timesteps_per_episode,
            gamma=self.gamma,
            actor_optim_type=self.actor_optim_type,
            critic_optim_type=self.critic_optim_type,
            n_updates_per_iteration=self.n_updates_per_iteration,
            clip=self.clip,
            lr_actor=self.lr_actor,
            lr_init_actor=self.lr_init_actor,
            lr_halflife_actor=self.lr_halflife_actor,
            lr_critic=self.lr_critic
        )

    def learn(self, total_timesteps, callback, writer):
        total_timesteps = int(total_timesteps)  # just in case

        # Initialize tensorboard writer
        self.writer = writer
        self.writer.log_parameters(self.get_hparams())
        # Initialize callback
        self._callback = callback
        self._callback.init_callback(self, self.writer)

        # Logging variables
        actor_losses, critic_losses = [], []

        # Training loop
        t_so_far = 0
        while t_so_far < total_timesteps:
            batch = self.rollout(self.timesteps_per_batch, self.max_timesteps_per_episode)
            if batch is None:  # end training
                return
            batch_obs, batch_acts, _, batch_log_probs, batch_rtgs, batch_lens = batch

            # Calculate how many timesteps we collected this batch
            t_so_far += sum(batch_lens)

            # V_{phi, k} and advantage
            V = self.get_values(batch_obs).detach()
            A_k = batch_rtgs - V
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  # standardization

            # Optimization loop
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V = self.get_values(batch_obs)
                curr_log_probs = self.get_action_dist(self.get_action(batch_obs)).log_prob(batch_acts)

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
                actor_loss.backward(retain_graph=True)  # so we can call backprop twice
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
            self.writer.log_metrics({'train/actor_loss': actor_loss,
                                     'train/critic_loss': critic_loss,
                                     'train/learning_rate_critic': self.lr_critic,
                                     'train/learning_rate_actor': self.lr_actor}, step=self.num_timesteps)

            self.update_learning_rate()

    def get_action(self, o):
        return self.actor(o)

    def predict(self, o, deterministic=True):
        action_mean = self.get_action(o)
        if deterministic:
            return action_mean.detach().numpy()
        else:
            a_dist = self.get_action_dist(action_mean)
            a_sample = a_dist.sample()
            log_prob = a_dist.log_prob(a_sample)

            return action_mean, log_prob

    def get_action_dist(self, a_mean):
        return MultivariateNormal(a_mean, self.cov_mat)

    def get_values(self, o):
        return self.critic(o).squeeze()

    def update_learning_rate(self):
        # Update exponential decay of actor lr
        self.lr_actor = self.lr_init_actor * np.exp(-(1 / self.lr_halflife_actor) * self.num_timesteps)

        for optim, lr in zip((self.actor_optim, self.critic_optim), (self.lr_actor, self.lr_critic)):
            for g in self.actor_optim.param_groups:
                g['lr'] = lr


class TRPO(OnPolicy):
    def __init__(self, env, **kwargs):
        """
        Based on https://github.com/ajlangley/trpo-pytorch

        :param env: OpenAI environment to learn on
        """
        self.__name = None
        super(TRPO, self).__init__(env, **kwargs)

        actor_hidden_layers = [200, 200]
        critic_hidden_layers = [200, 100]
        self.policy = utils.MLP(self.obs_dim, self.act_dim, actor_hidden_layers)
        self.value_fun = utils.MLP(self.obs_dim, 1, critic_hidden_layers)

        # Create actor covariance matrix
        self.cov_var = t.full(size=(self.act_dim,), fill_value=0.1)  # 0.5 arbitrary
        self.cov_mat = t.diag(self.cov_var)

        # Initialize optimizers
        # self.policy_optim = self.policy_optim_type(self.policy.parameters(), lr=self.lr_actor)
        self.value_optim = self.value_optim_type(self.value_fun.parameters(), lr=self.value_lr)

    def __repr__(self):
        return f'PPO_{repr(self.env)}_{dt.strftime(dt.now(), "%m%d%y_%H%M%S")}'

    def _init_hparams(self, **kw):
        self.timesteps_per_batch = kw.get('timesteps_per_batch', 2000)
        self.max_timesteps_per_episode = kw.get('max_timesteps_per_episode', 100)

        self.value_optim_type = kw.get('value_optim_type', Adam)
        self.value_lr = kw.get('value_lr', 1e-3)

        self.max_kl_div = kw.get('max_kl_div', 0.01)  # max kl before and after each step
        self.max_value_step = kw.get('max_value_step', 0.01)  # lr for value function
        self.vf_iters = kw.get('vf_iters',
                               1)  # nb of times to optimize value func over each set of training experiences
        self.vf_l2_reg_coef = kw.get('vf_l2_reg_coef', 1e-3)  # regularization term when calc. L2 loss of value function
        self.discount = kw.get('discount', 0.995)  # discount for future rewards
        self.lam = kw.get('lam', 0.98)  # bias reduction parameter when calculing advantages using GAE

        self.cg_damping = kw.get('cg_damping',
                                 1e-3, )  # identity matrix multiple to add to Hessian when calc. Hessian-vector prod
        self.cg_max_iters = kw.get('cg_max_iters', 10)  # max nb of iterations when solving for optimal search direction

        self.line_search_coef = kw.get('line_search_coef',
                                       0.9)  # proportion by which to reduce step length on each line search iteration
        self.line_search_max_iter = kw.get('line_search_max_iter',
                                           10)  # max nb of line search iterations before returning 0.0 as step length
        self.line_search_accept_ratio = kw.get('line_search_accept_ratio',
                                               0.1)  # min proportion of error to accept from line search linear extrapolation
        self.model_name = kw.get('model_name', None)  # filepaths identifier

    def learn(self, total_timesteps, callback, log_dir=None):
        total_timesteps = int(total_timesteps)  # just in case

        # Initialize tensorboard writer
        if log_dir is None:
            log_dir = 'trpo_training'
        self.writer = utils.get_writer(log_dir, project_name=repr(self), workspace=COMET_WORKSPACE)

        # Initialize callback
        self._callback = callback
        self._callback.init_callback(self, self.writer)

        # Logging variables
        actor_losses, critic_losses = [], []

        # Training loop
        t_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_rews, batch_log_probs, batch_rtgs, batch_lens = \
                self.rollout(self.timesteps_per_batch, self.max_timesteps_per_episode)
            t_so_far += sum(batch_lens)

            # Get advantages from current value function and actual returns
            advantages = self.get_advantages(batch_obs, batch_rews)

    # Update policy

    # Update value function

    def get_advantages(self, states, rewards):
        values = self.value_fun(states)
        values_next = t.cat([values[:, 1:], t.zeros(values.shape[0]).unsqueeze(1)])
        td_residuals = rewards + self.discount * values_next - values
        discount_powers = t.pow(self.discount * self.lam,
                                t.arange(0, rewards.size(0)).float())
        discounted_residuals = td_residuals * discount_powers
        advantages = t.cumsum(discounted_residuals.flip(-1), dim=-1).flip(-1) / discount_powers
        return advantages

    def get_action(self, o):
        mean = self.policy(o)
        dist = MultivariateNormal(mean, self.cov_mat)

        a = dist.sample()
        log_prob = dist.log_prob(a)

        return a.detach().numpy(), log_prob.detach()

    def predict(self, o, deterministic=True):
        pass

    @staticmethod
    def conjugate_gradient_solver(Avp_fun, b, max_iter=10):
        """
        Finds an approximate solution to a set of linear equations Ax = b
        :param Avp_fun: function that right multiplies matrix A by vector x
        :param b: rhs of equation
        :param max_iter: max nb of iterations. default is 10
        :return: vector x corresponding to the approximate solution to the system of equations made up of A and b
        """
        x = t.zeros_like(b)
        r = b.clone()
        p = b.clone()

        for i in range(max_iter):
            Avp = Avp_fun(p, retain_graph=True)

            alpha = t.matmul(r, r) / t.matmul(p, Avp)
            x += alpha * p

            if i == max_iter - 1:
                return x

            r_new = r - alpha * Avp
            beta = t.matmul(r_new, r_new) / t.matmul(r, r)
            r = r_new
            p = r + beta * p

    @staticmethod
    def line_search(s, max_steps_length, constraints_satisfied, line_search_coef=0.9, max_iter=10):
        """
        Backtracking line-search that terminates when constraints_satisfied returns True and returns the calculated step
        length. Returns zero if no step length could be found for which constraints_satisfied returns True.
        :param s: search direction along which line search is performed
        :param max_steps_length: maximum step length to consider in line search
        :param constraints_satisfied: function that returns bool whether constraints were met by argument step length
        :param line_search_coef: proportion by which to reduce the step length after each iteration
        :param max_iter: max nb of backtracks to do before returning zero by default
        :return: the maximum step length for which costraints_satisfied returned True
        """
        x = max_steps_length
        for i in range(max_iter):
            if constraints_satisfied(x * s, x):
                return x
            x *= line_search_coef
        return t.tensor(0.0)

    @staticmethod
    def get_Hvp_fun(functional_output, inputs, damping_coef=0.0):
        """
        Returns function that calculates Hessian-vector product with the Hessian of functional_output wrt inputs
        :param functional_output:
        :param inputs:
        :param damping_coef: multiple of identitiy matrix to be added to Hessian
        :return:
        """
        flatten = lambda outs, ins: t.cat([v.view(-1) for v in grad_f])
        inputs = list(inputs)
        grad_f = flatten(grad(functional_output, inputs, retain_graph=True, create_graph=True))

        def Hvp_fun(v, retain_graph=True):
            gvp = t.matmul(grad_f, v)
            Hvp = flatten(grad(gvp, inputs, retain_graph=True))
            Hvp += damping_coef * v
            return Hvp

        return Hvp_fun


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

    callback = utils.EvalCheckpointEarlyStopTrainingCallback(eval_env, save_dir, EVAL_FREQ=500, CHKPT_FREQ=1000)
    agent = PPO(env)
    agent.learn(nb_training_steps, callback, log_dir)


if __name__ == '__main__':
    train_my_ppo()
