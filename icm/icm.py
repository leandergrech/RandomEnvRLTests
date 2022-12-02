import os
import numpy as np
from datetime import datetime as dt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt
from stable_baselines3.ppo import PPO

from random_env.envs import RandomEnv
from utils.training_utils import BestSaveCheckpointCallBack
from utils.plotting_utils import grid_on


class StateFeatures(nn.Module):
    def __init__(self, env: RandomEnv, h: list, feature_size=10):
        super(StateFeatures, self).__init__()
        self.n_obs = env.obs_dimension
        self.n_act = env.act_dimension
        self.feature_size = feature_size

        if len(h) != 2:
            raise NotImplementedError("2 FC layers allowed only")

        self.fc1 = nn.Linear(self.n_obs, h[0])
        self.fc2 = nn.Linear(h[0], h[1])
        self.fc3 = nn.Linear(h[1], self.feature_size)

    def forward(self, state):
        '''
        TODO: Check other activation functions, e.g. LRELU, ELU, TANH
        Note: ELU was used in the ICM paper actually
        '''
        x = t.Tensor(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ForwardModel(nn.Module):
    def __init__(self, env: RandomEnv, h: list, feature_size=10):
        super(ForwardModel, self).__init__()
        self.n_obs = env.obs_dimension
        self.n_act = env.act_dimension
        self.feature_size = feature_size

        if len(h) != 2:
            raise NotImplementedError("2 FC layers allowed only")

        self.fc1 = nn.Linear(feature_size + self.n_act, h[0])
        self.fc2 = nn.Linear(h[0], h[1])
        self.fc3 = nn.Linear(h[1], self.feature_size)

    def forward(self, phi, action):
        phi = t.Tensor(phi)
        action = t.Tensor(action)
        x = t.Tensor(t.hstack((phi, action)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class InverseModel(nn.Module):
    def __init__(self, env: RandomEnv, h: list, feature_size=10):
        super(InverseModel, self).__init__()
        self.n_obs = env.obs_dimension
        self.n_act = env.act_dimension
        self.feature_size = feature_size

        if len(h) != 2:
            raise NotImplementedError("2 FC layers allowed only")

        self.fc1 = nn.Linear(2 * self.feature_size, h[0])
        self.fc2 = nn.Linear(h[0], h[1])
        self.fc3 = nn.Linear(h[1], self.n_act)

    def forward(self, phi1, phi2):
        phi1 = t.Tensor(phi1)
        phi2 = t.Tensor(phi2)
        x = t.concat([phi1, phi2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ICM(nn.Module):
    def __init__(self, env: RandomEnv, fm_kw: {}, im_kw: {}, sf_kw: {}):
        super(ICM, self).__init__()
        self.n_obs = env.obs_dimension
        self.n_act = env.act_dimension

        self.state_features = StateFeatures(env, **sf_kw)
        self.forward_model = ForwardModel(env, **fm_kw)
        self.inverse_model = InverseModel(env, **im_kw)

        self.env = env

    def forward(self, state, action, next_state):
        phi1 = self.state_features(state)
        phi2 = self.state_features(next_state)

        phi2_estimate = self.forward_model(phi1, action)
        action_estimate = self.inverse_model(phi1, phi2)

        return dict(phi1=phi1,
                    phi2=phi2,
                    phi2_estimate=phi2_estimate,
                    action_estimate=action_estimate)


class ICM_RandomEnv(RandomEnv):
    def __init__(self, n_obs, n_act, fm_kw=None, im_kw=None, sf_kw=None, **kwargs):
        super(ICM_RandomEnv, self).__init__(n_obs, n_act, **kwargs)

        if fm_kw is None:
            fm_kw = dict(h=[10, 10], feature_size=10)
        if im_kw is None:
            im_kw = dict(h=[10, 10], feature_size=10)
        if sf_kw is None:
            sf_kw = dict(h=[10, 10], feature_size=10)

        self.icm = ICM(self, fm_kw=fm_kw, im_kw=im_kw, sf_kw=sf_kw)

        self.lr = 1e-3
        self.beta = 0.2
        self.neta = 1.0
        self.optim = t.optim.SGD(self.icm.parameters(), lr=self.lr)
        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = nn.MSELoss()

    def __repr__(self):
        """Don't change this anymore, need it for static_method load_from_dir"""
        return f'ICM_RandomEnv_{self.obs_dimension}obsx{self.act_dimension}act'

    def reset(self, init_state=None):
        return super(ICM_RandomEnv, self).reset(init_state)

    def get_intrinsic_reward(self, state, action, next_state, full_return=False):
        ret = self.icm(state, action, next_state)

        phi2_estimate = ret['phi2_estimate']
        phi2 = ret['phi2']
        action_estimate = ret['action_estimate']

        intrinsic_reward = (self.neta / 2.) * t.sqrt(t.sum(t.square(phi2_estimate - phi2)))

        if full_return:
            return intrinsic_reward, phi2_estimate, phi2, action_estimate
        else:
            return intrinsic_reward

    def gradient_update(self, state, action, next_state):
        self.optim.zero_grad()

        intrinsic_reward, phi2_estimate, phi2, action_estimate = \
            self.get_intrinsic_reward(state, action, next_state, full_return=True)

        loss1 = self.loss_fn1(phi2_estimate, phi2)
        loss2 = self.loss_fn2(action_estimate, t.Tensor(action))

        loss = self.beta * loss1 + (1 - self.beta) * loss2

        loss.backward()

        self.optim.step()

        return intrinsic_reward.detach().numpy(), loss1.item(), loss2.item()

    def step(self, action, curious=True):
        o = self.current_state
        otp1, r, d, info = super(ICM_RandomEnv, self).step(action)

        if curious:
            ri, l1, l2 = self.gradient_update(o, action, otp1)
            # ri = self.get_intrinsic_reward(o, action, otp1)[0].detach().numpy()
            info['ri'] = ri
            info['l1'] = l1
            info['l2'] = l2
            r += ri
        return otp1, r, d, info


def train_icm_and_compare_to_re_random_trajectories():
    n_obs, n_act = 2, 2

    max_steps = 100

    nb_runs = 2
    nb_eps = 10
    seed = 123

    envs = []
    envcs = []

    normal_rewards = np.zeros(shape=(nb_runs, nb_eps, max_steps), dtype=float)
    icm_rewards = np.zeros(shape=(nb_runs, nb_eps, max_steps), dtype=float)
    intr_rewards = np.zeros(shape=(nb_runs, nb_eps, max_steps), dtype=float)
    extr_rewards = np.zeros(shape=(nb_runs, nb_eps, max_steps), dtype=float)

    # normal_steps = np.repeat([np.arange(max_steps * nb_eps)], nb_runs, axis=0)
    # icm_steps = np.repeat([np.arange(max_steps * nb_eps)], nb_runs, axis=0)
    normal_steps = np.arange(max_steps * nb_eps)
    icm_steps = np.arange(max_steps * nb_eps)

    ep_start_normal = []
    ep_start_icm = []

    # Initialize initial state distribution from one of the environments

    print('-> Running on RandomEnv')
    # Run random trajectories with the NORMAL RandomEnv and observe the rewards
    np.random.seed(seed)

    init_o = np.zeros((nb_runs, nb_eps, n_obs))
    for run in range(nb_runs):
        it = 0

        env = RandomEnv(n_obs, n_act)
        env.EPISODE_LENGTH_LIMIT = max_steps
        envs.append(env)
        for i in range(nb_eps):
            init_o[run, i] = env.reset()

        for ep in trange(nb_eps):
            ep_start_normal.append(it)
            o = init_o[run, ep].copy()
            env.reset(o)
            d = False
            step = 0
            while not d:
                a = env.action_space.sample()
                otp1, r, d, info = env.step(a)
                normal_rewards[run, ep, step] = r
                o = otp1
                it += 1
                step += 1
            ep_nb_steps = step
            normal_rewards[run, ep, ep_nb_steps:] = normal_rewards[run, ep, ep_nb_steps - 1]

    print('-> Running on ICM_RandomEnv')
    # Run random trajectories with the ICM environment and observe the extrinsic and instrinsic rewards
    np.random.seed(seed)

    for run in range(nb_runs):
        it = 0

        envc = ICM_RandomEnv(n_obs, n_act, model_info=envs[-1].model_info)
        envc.EPISODE_LENGTH_LIMIT = max_steps

        envcs.append(envc)

        for ep in trange(nb_eps):
            ep_start_icm.append(it)
            o = init_o[run, ep].copy()
            envc.reset(o)
            d = False
            step = 0
            while not d:
                a = envc.action_space.sample()
                otp1, e_r, d, info = envc.step(a)
                i_r, *_ = envc.gradient_update(o, a, otp1)
                i_r = i_r.detach().numpy()
                intr_rewards[run, ep, step] = i_r
                extr_rewards[run, ep, step] = e_r
                r = e_r + i_r
                icm_rewards[run, ep, step] = r
                o = otp1.copy()
                it += 1
                step += 1
            ep_nb_steps = step
            intr_rewards[run, ep, ep_nb_steps:] = intr_rewards[run, ep, ep_nb_steps - 1]
            extr_rewards[run, ep, ep_nb_steps:] = extr_rewards[run, ep, ep_nb_steps - 1]
            icm_rewards[run, ep, ep_nb_steps:] = icm_rewards[run, ep, ep_nb_steps - 1]

    print('-> Plotting')

    fig, (ax1, ax2) = plt.subplots(2)
    # fig, ax1 = plt.subplots()
    fig.suptitle(f'Random Agent on {repr(env)} vs.\n'
                 f'Random Agent on {repr(envc)}')
    ave_nor_rew = np.mean(normal_rewards.reshape((nb_runs, -1)), axis=0)
    std_nor_rew = np.std(normal_rewards.reshape((nb_runs, -1)), axis=0)
    # ave_nor_rew = np.mean(normal_rewards, axis=1).reshape((nb_runs, -1))
    # std_nor_rew = np.std(normal_rewards, axis=1).reshape((nb_runs, -1))
    # xrange = normal_steps[0]
    xrange = normal_steps

    # for y, ys in zip(ave_nor_rew, std_nor_rew):
    y, ys = ave_nor_rew, std_nor_rew
    ax1.plot(xrange, y, label='Random Agent', c='b')
    ax1.fill_between(xrange, y-ys, y+ys, alpha=0.5, color='b')

    ave_icm_rew = np.mean(icm_rewards.reshape((nb_runs, -1)), axis=0)
    std_icm_rew = np.std(icm_rewards.reshape((nb_runs, -1)), axis=0)
    # ave_icm_rew = np.mean(icm_rewards, axis=1).reshape((nb_runs, -1))
    # std_icm_rew = np.std(icm_rewards, axis=1).reshape((nb_runs, -1))
    # xrange = icm_steps[0]
    xrange = icm_steps

    # for y, ys in zip(ave_icm_rew, std_icm_rew):
    y, ys = ave_icm_rew, std_icm_rew
    ax1.plot(xrange, y, label='Random Agent + ICM', c='tab:orange')
    ax1.fill_between(xrange, y - ys, y + ys, alpha=0.5, color='tab:orange')

    for start, startc in zip(ep_start_normal, ep_start_icm):
        ax1.axvline(start, color='b', alpha=0.8, ls='--')
        ax1.axvline(startc, color='tab:orange', alpha=0.8, ls=':')

    grid_on(ax=ax1, axis='y', minor_loc=1e-1, major_loc=1, major_grid=True, minor_grid=True)
    # ave_extr_rew = np.mean(extr_rewards, axis=1).reshape((nb_runs, -1))
    # std_extr_rew = np.std(extr_rewards, axis=1).reshape((nb_runs, -1))
    #
    # ax2.plot(xrange, extr_rewards, label='Extrinsic Reward')
    #
    # ave_intr_rew = np.mean(intr_rewards, axis=1)
    # std_intr_rew = np.std(intr_rewards, axis=1)
    # ax2.plot(xrange, intr_rewards, label='Intrinsic Reward')

    # xrange = normal_steps
    # y = np.cumsum(ave_nor_rew)
    y = np.mean(np.sum(normal_rewards, axis=2), axis=0)
    ys = np.std(np.sum(normal_rewards, axis=2), axis=0)
    xrange = np.range(len(y))
    ax2.plot(xrange, y)
    ax2.fill_between(xrange, y - ys, y + ys, color='b', alpha=0.5)

    # xrange = icm_steps
    # y = np.cumsum(ave_icm_rew)
    # y = np.sum(icm_rewards, axis=2)
    # ax2.plot(y, ls=':')
    y = np.mean(np.sum(icm_rewards, axis=2), axis=0)
    ys = np.std(np.sum(icm_rewards, axis=2), axis=0)
    xrange = np.range(len(y))
    ax2.plot(xrange, y)
    ax2.fill_between(xrange, y - ys, y + ys, color='tab:orange', alpha=0.5)

    for a in fig.axes:
        a.legend(loc='best')
        a.set_xlabel('Iterations')
        a.set_ylabel('Rewards')

    fig.tight_layout()
    plt.show()


def train_ppo_compare_icmre_vs_re():
    n_obs, n_act = 2, 2
    max_steps = 100

    env_seed = 123
    feature_size = 2
    icm_hidden_layers = [5, 5]
    icm_kwargs = dict(fm_kw=dict(h=icm_hidden_layers, feature_size=feature_size),
                      im_kw=dict(h=icm_hidden_layers, feature_size=feature_size),
                      sf_kw=dict(h=icm_hidden_layers, feature_size=feature_size))

    envs = []
    envs.append(ICM_RandomEnv(n_obs=n_obs, n_act=n_act, estimate_scaling=True, **icm_kwargs))
    envs.append(RandomEnv(n_obs=n_obs, n_act=n_act, model_info=envs[0].model_info))

    for env in envs:
        env.EPISODE_LENGTH_LIMIT = max_steps
        env.max_steps = max_steps
        env.seed(env_seed)

    model_prefix = lambda seed: f'{PPO}_{repr(envs[0])}_{seed}seed'

    policy_kwargs = dict(net_arch=[dict(vf=[16, 16], pi=[16, 16])],
                         activation_fn=t.nn.ReLU)

    ppo_kwargs = dict(policy='MlpPolicy',
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
                      sde_sample_freq=-1,
                      target_kl=None,
                      tensorboard_log=None,
                      create_eval_env=False,
                      policy_kwargs=policy_kwargs,
                      verbose=1,
                      # seed=seed,  SET EXTERNALLY
                      device="auto",
                      _init_setup_model=True,)

    total_train_timesteps = int(5e5)
    save_freq = 1000
    eval_freq = 500
    n_eval_episodes = 10

    def train(env, ppo_kwargs):
        nonlocal save_freq, eval_freq, n_eval_episodes, total_train_timesteps

        seed = ppo_kwargs['seed']

        par_dir = 'ppo_icm_experiments'
        model_name = f'PPO_{repr(env)}_{dt.now().strftime("%m%d%y_%H%M%S")}'

        model_dir = os.path.join(par_dir, model_name)
        tensorboard_dir = os.path.join(model_dir, 'log')
        save_dir = os.path.join(model_dir, 'saves')

        callbacks = [BestSaveCheckpointCallBack(save_freq=save_freq, save_dir=save_dir, name_prefix=model_prefix(seed))]

        ppo_kwargs['tensorboard_log'] = tensorboard_dir
        ppo_kwargs['env'] = env
        agent = PPO(**ppo_kwargs)

        eval_env = RandomEnv(n_obs=n_obs, n_act=n_act, model_info=env.model_info)
        agent.learn(total_timesteps=total_train_timesteps, callback=callbacks, log_interval=eval_freq//5,
                    eval_env=eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)

    for seed in (123, 234, 345, 456, 567):
        ppo_kwargs['seed'] = seed
        for env in envs:
            train(env, ppo_kwargs)


if __name__ == '__main__':
    train_ppo_compare_icmre_vs_re()
    # env = RandomEnv(2, 2)
    # layer_kw = dict(h=[5, 5], feature_size=2)
    # icm = ICM(env, fm_kw=layer_kw, im_kw=layer_kw, sf_kw=layer_kw)
    # # for parameter in icm.named_parameters():
    # #     print(parameter)
    # print(len([item for item in icm.named_parameters()]))
    # print(len([item for item in icm.parameters()]))
    #
    # print(sum([len(item) for item in icm.named_parameters()]))
    # print(sum([len(item) for item in icm.parameters()]))
    #
    # print([len(item) for item in icm.named_parameters()])
    # print([len(item) for item in icm.parameters()])
    #
    # print([item for item in icm.named_parameters()])
    # print([item for item in icm.parameters()])
    #
    #
