import os.path
import warnings
from collections import deque
import re  # regular expressions: string formatting, etc.
import matplotlib.pyplot as plt
import numpy as np
from gym import Env
from gym.spaces import Box
from scipy import stats
import pickle as pkl


class RandomEnv(Env):
    REWARD_DEQUE_SIZE = 1  # 5
    ACTION_SCALE = 1.0  # set to one during dynamic output scale adjustment test

    _UPDATE_SCALING = False

    RESET_RANDOM_WALK_STEPS = 50  # Reset func starts from optimal state, and walks randomly for these amount of steps
    SAVED_MODEL_SUFFIX = '_dynamics.pkl'
    K_p = 1.0
    K_i = 0.0

    def __init__(self, n_obs, n_act, estimate_scaling=True, model_info=None):
        """
        An OpenAI Gym environment with random transition dynamics.
        :param n_obs: 1D observations space size
        :param n_act: action space size
        :param estimate_scaling: Boolean, choose whether to
        :param model_info:
        :param seed:
        """
        super(RandomEnv, self).__init__()
        self.REWARD_SCALE = 0.05
        # Have many times smaller should the average state trim be than an state space bounds
        self.TRIM_FACTOR = 5.
        self.EPISODE_LENGTH_LIMIT = 100
        self.GOAL = 0.1  # state threshold boundary

        self.obs_dimension, self.act_dimension = n_obs, n_act

        ''' State and action space'''
        self.observation_space = Box(low=-1.0,
                                     high=1.0,
                                     shape=(self.obs_dimension,),
                                     dtype=float)
        self.action_space = Box(low=-np.ones(self.act_dimension),
                                high=np.ones(self.act_dimension),
                                dtype=float)

        ''' RL related parameters'''
        self.current_state = None
        self._reward = None
        self.reward_thresh = self.objective([self.GOAL] * self.obs_dimension)
        self.reward_deque = deque(maxlen=RandomEnv.REWARD_DEQUE_SIZE)
        self._it = 0
        self.max_steps = self.EPISODE_LENGTH_LIMIT

        ''' Create model dynamics and adjust scaling'''
        if model_info is None:
            self.create_model()
            self.trim_stats = RunningStats(n_obs)  # Used to standardise the model trim outputs
            if estimate_scaling:
                self._estimate_output_scaling()
        else:
            self.rm, self.pi = model_info['rm'], model_info['pi']
            self.trim_stats = model_info['trim_stats']

        self.verbose = False

    def __repr__(self):
        """Don't change this anymore, need it for static_method load_from_dir"""
        return f'RandomEnv_{self.obs_dimension}obsx{self.act_dimension}act'

    def reset(self, init_state=None):
        # QFBEnv method to was easier since state was 2D
        if init_state is None:
            init_state = self.observation_space.sample()

        # Large Envs require a solvable initial state - Solution, reach one through a random walk
        # init_state = np.zeros(self.obs_dimension)
        # for _ in range(RandomEnv.RESET_RANDOM_WALK_STEPS):
        # 	a = self.action_space.sample()
        # 	trim = self.normalise_trim(self.rm.dot(a))
        # 	init_state += trim

        self.current_state = init_state
        self.reward_deque.clear()
        self._it = 0

        self.prev_error = np.zeros(self.obs_dimension)

        return np.copy(init_state)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        action = np.multiply(action, RandomEnv.ACTION_SCALE)
        trim_state = self.rm.dot(action)

        # Standardise trims by stats in trim_stats
        trim_state = self.standardise_trim(trim_state)

        self.current_state += trim_state
        r = self.objective(self.current_state)
        self.reward = r
        done, success = self._is_done()

        self._it += 1

        return self.current_state, r, done, dict(success=success)

    # @staticmethod
    def objective(self, state):
        state_reward = -np.sum(np.square(state)) / self.obs_dimension
        # state_reward = -np.sqrt(np.mean(np.square(state)))
        return state_reward * self.REWARD_SCALE

    def _is_done(self):
        done, success = False, False
        # Reach goal
        if len(self.reward_deque) >= RandomEnv.REWARD_DEQUE_SIZE and \
                np.max(np.abs(self.current_state)) <= self.GOAL:
            done, success = True, True
        elif self._it >= self.EPISODE_LENGTH_LIMIT - 1:
            done = True
        return done, success

    def get_optimal_action(self, state, state_clip=None):
        if state_clip:
            state = np.clip(state, -state_clip, state_clip)
        self.prev_error = np.copy(state)  # in case integral controller is used as well

        action = -self.pi.dot(RandomEnv.K_p * state + RandomEnv.K_i * self.prev_error)
        return action / max([1.0, max(abs(action))])  # linearly scaled response to fit within [-1, 1]

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, r):
        self._reward = r
        self.reward_deque.append(r)

    def seed(self, seed=None):
        np.random.seed(seed)

    def create_model(self):
        n_obs, n_act = self.obs_dimension, self.act_dimension

        if n_obs == 1 and n_act == 1:
            self.rm = np.array([np.random.rand() + 1.0])
            self.pi = 1/self.rm

            return


        # Instantiate left & right singular vectors, and singular value matrices
        u = stats.ortho_group.rvs(n_obs)
        # s = np.diag(sorted(np.random.uniform(0, 1, min(n_obs, n_act)), reverse=True))
        s = np.diag(np.ones(min(n_obs, n_act)))
        vh = stats.ortho_group.rvs(n_act).T

        # Padding logic for s
        if n_obs > n_act:
            first_pad = n_obs - n_act
            second_pad = 0
        elif n_act > n_obs:
            first_pad = 0
            second_pad = n_act - n_obs
        else:
            first_pad, second_pad = 0, 0

        # Pad s to match sizes of actions and states
        s = np.pad(s, ((0, first_pad), (0, second_pad)))

        # Get inverse components
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sinv = np.where(1 / s == np.inf, 0, 1 / s)
        sinv = sinv.T
        uh = u.T
        v = vh.T

        # Get Response Matrix and its Pseudo-Inverse
        self.rm = u.dot(s.dot(vh))
        self.pi = v.dot(sinv.dot(uh))

    def normalise_trim(self, trim):
        trim_normed = np.divide(trim - self.trim_stats.min, self.trim_stats.ptp) * 2.0 - 1.0
        return trim_normed / self.TRIM_FACTOR

    def standardise_trim(self, trim):
        trim_stded = np.divide(trim - self.trim_stats.mean, self.trim_stats.std)
        return trim_stded / self.TRIM_FACTOR

    @property
    def model_info(self):
        return dict(rm=self.rm, pi=self.pi, trim_stats=self.trim_stats)

    def _estimate_output_scaling(self):
        RandomEnv._UPDATE_SCALING = True
        for _ in range(10000):
            a = self.action_space.sample()
            self.trim_stats.add(self.rm.dot(a))

        RandomEnv._UPDATE_SCALING = False

    def save_dynamics(self, save_dir):
        save_path = os.path.join(save_dir, self.__repr__() + RandomEnv.SAVED_MODEL_SUFFIX)
        if not os.path.exists(save_path):
            with open(save_path, 'wb') as f:
                pkl.dump(dict(rm=self.rm, pi=self.pi, trim_stats=self.trim_stats), f)
                if self.verbose:
                    print(f'Saved model dynamics of {self.__repr__()} to: {save_path}')
        else:
            raise FileExistsError(
                f'Directory passed: {save_dir}, already contains dynamics for a {self.__repr__()} environment.')

    def load_dynamics(self, load_dir):
        load_path = os.path.join(load_dir, self.__repr__() + RandomEnv.SAVED_MODEL_SUFFIX)
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                dynamics = pkl.load(f)
                self.rm = dynamics['rm']
                self.pi = dynamics['pi']
                self.trim_stats = dynamics['trim_stats']
        else:
            raise FileNotFoundError(
                f'Directory passed: {load_dir}, does not contain dynamics for a {self.__repr__()} envirnment.')

    @staticmethod
    def load_from_dir(load_dir):
        for file in os.listdir(load_dir):
            if RandomEnv.SAVED_MODEL_SUFFIX in file:
                n_obs, n_act = re.findall(r'\d+', file)
                n_obs, n_act = int(n_obs), int(n_act)
                self = RandomEnv(n_obs, n_act, estimate_scaling=False)
                self.load_dynamics(load_dir)

                return self

        return None


# 01/03/2022
def get_model_output_bounds():
    env = RandomEnv(10, 10)
    env.seed(123)
    import tensorflow as tf
    import random
    tf.random.set_random_seed(123)
    random.seed(123)
    o_list = []

    N_EPS = 1000
    for ep in range(N_EPS):
        o = env.reset()
        o_list.append(o)
        for step in range(env.max_steps):
            a = env.action_space.sample()
            o, *_ = env.step(a)
            o_list.append(o)
        o_mean = np.mean(o_list, axis=0)
        o_std = np.std(o_list, axis=0)

    print(f'Observation mean = {o_mean}, std = {o_std}')


# 02/02/2022
def quick_testing_randomenv():
    n_obs = 10
    n_act = 10

    # env = RandomEnv(n_obs, n_act, estimate_scaling=True)
    env = RandomEnv.load_from_dir("C:\\Users\\Leander\\Code\\RandomEnvRLTests\\common_envs")

    d = False
    o1 = env.reset()
    o_list = [o1]
    a_list = []

    # print(np.linalg.det(env.pi),np.linalg.det(env.rm))

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('Observations')
    ax2.set_title('Actions')

    lobs, = ax1.plot(np.zeros(n_obs))
    lact, = ax2.plot(np.zeros(n_act))

    ax1.axhline(-env.GOAL, color='g', ls='--', lw=1.5)
    ax1.axhline(env.GOAL, color='g', ls='--', lw=1.5)

    plt.show(block=False)
    for ax in fig.axes:
        ax.set_ylim((-1, 1))
        ax.axhline(0.0, color='k', ls='--', lw=0.7)

    cur_step = 0
    while not d:
        a = env.get_optimal_action(o1)
        o2, r, d, _ = env.step(a)

        lobs.set_ydata(o2)
        lact.set_ydata(a)
        fig.suptitle(f'Step {cur_step}, Done = {d}\n' +
                     f'std(o1 - o2))) = {np.std(o1 - o2):.4f}\treward = {r:.4f}')
        plt.pause(0.1)

        o_list.append(o2.copy())
        a_list.append(a)
        o1 = np.copy(o2)

        cur_step += 1
    plt.pause(2)

    o_list = np.array(o_list)
    a_list = np.array(a_list)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(o_list, label='States')
    ax2.plot(a_list, label='Actions')
    ax1.axhline(-env.GOAL, color='k', ls='--')
    ax1.axhline(env.GOAL, color='k', ls='--')

    # for a in fig.axes:
    # 	a.legend(loc='best')
    plt.show()


# 03/02/2022
class RunningStats():
    def __init__(self, size, axis=0):
        self.M = np.zeros(size)
        self.S = np.zeros(size)
        self.Min = np.repeat(np.inf, size)
        self.Max = np.repeat(-np.inf, size)
        self.n = 0

    def __repr__(self):
        return f"Mean: min={min(self.mean):.2f} max={max(self.mean):.2f}\n" \
               f"Std:  min={min(self.std):.2f}  max={max(self.std):.2f}\n" \
               f"Min = {self.Min}\n" \
               f"Max = {self.Max}"

    def add(self, x):
        self.n += 1

        # Welford algorithm
        oldM = np.copy(self.M)
        self.M = self.M + (x - self.M) / self.n
        self.S = self.S + (x - self.M) * (x - oldM)

        self.Min = np.minimum(self.Min, x)
        self.Max = np.maximum(self.Max, x)

    @property
    def mean(self):
        return np.copy(self.M)

    @property
    def std(self):
        if self.n <= 1:
            return np.ones_like(
                self.M)  # set to ones not zeros to avoid logic outside this class when dividing by std to standardise the observations
        else:
            return np.sqrt(self.S / (self.n - 1))

    @property
    def min(self):
        if self.n <= 1:
            return -np.ones_like(self.Min)
        else:
            return self.Min

    @property
    def max(self):
        return self.Max

    @property
    def ptp(self):
        if self.n <= 1:
            return 2 * np.ones_like(self.Min)
        else:
            return self.Max - self.Min


def dynamic_scale_adjustment_test():
    n_obs = 100
    n_act = 100
    env = RandomEnv(n_obs, n_act, estimate_scaling=False)

    # K variables
    obs_stats = RunningStats(n_obs)
    augment_state = lambda x: (x - obs_stats.mean) / obs_stats.std

    NB_EPS = 50
    total_o_list = np.zeros((NB_EPS * env.max_steps + NB_EPS, n_obs))
    total_o_means = np.zeros((NB_EPS * env.max_steps + NB_EPS, n_obs))
    total_o_stds = np.zeros((NB_EPS * env.max_steps + NB_EPS, n_obs))
    ep_start_at_steps = []
    cur_step = 0
    for ep in range(NB_EPS):
        o1 = env.reset()
        ep_start_at_steps.append(cur_step)

        # Save initial state stats too
        obs_stats.add(o1)
        total_o_means[cur_step] = obs_stats.mean
        total_o_stds[cur_step] = obs_stats.std
        total_o_list[cur_step] = o1

        d = False
        while not d:
            cur_step += 1
            # Apply optimal dynamics
            # a = env.get_optimal_action(o1) * 0.1 + np.random.normal(0, 0.05, n_act)
            a = env.action_space.sample()
            o2, r, d, _ = env.step(a)

            # Save stats
            obs_stats.add(o2)
            total_o_means[cur_step] = obs_stats.mean
            total_o_stds[cur_step] = obs_stats.std

            # Apply scaling
            total_o_list[cur_step] = np.copy(o2)
            o1 = o2

        cur_step += 1

    total_o_list = np.array(total_o_list)
    total_o_means = np.array(total_o_means)
    total_o_stds = np.array(total_o_stds)

    print(f"Mean min={total_o_means.min()}, max={total_o_means.max()}")
    print(f"Std min={total_o_stds.min()}, max={total_o_stds.max()}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(18, 8))
    fig.suptitle(
        f'Random walks on RandomEnv{n_obs}x{n_act}\nNb episodes = {NB_EPS}\nMaximum episode length = {env.max_steps}')

    ax1.plot(total_o_list, zorder=5)
    ax1.grid(which='both')
    ax1.set_title('Observations')
    ax1.set_ylabel('Un-standardised\nstate space')

    # ax1.set_xlim((0, cur_step))
    for ep_start in ep_start_at_steps:
        if ep_start == 0:
            ax1.axvline(ep_start, color='k', label='Episode starts')
        else:
            ax1.axvspan(ep_start - 1, ep_start, color='k', zorder=10, alpha=1.0)  # Hide episode end-start transition

    ax1.legend(loc='lower right')
    ax2.plot(total_o_means, alpha=1, lw=0.8, ls='-')
    ax2.axhline(0.0, color='k', ls='dashed')
    ax2.set_title('Obs Running Mean (state per line)')
    ax3.plot(total_o_stds, alpha=1, lw=0.8, ls='-')
    ax3.axhline(0.0, color='k', ls='dashed')
    ax3.set_title('Obs Running Std (state per line)')

    def forward(x):
        return (1 + 3e-4) ** x

    def inverse(x):
        # if sum(x) == 0:
        return x

    # else:
    # 	return x ** (1/2)

    cm = plt.get_cmap('prism')

    def update_colors(ax):
        lines = ax.lines
        colors = cm(np.linspace(0, 1, len(lines)))
        for line, c in zip(lines, colors):
            line.set_color(c)

    for ax in (ax2, ax3):
        ax.set_xscale('function', functions=(forward, inverse))
        ax.set_xlim((0, cur_step))
        ax.set_yscale('symlog')
        update_colors(ax)

    for ax in fig.axes:
        ax.set_xlabel('Steps')
        ax.grid(which='both', color='gray')

    fig.subplots_adjust(top=0.87,
                        bottom=0.085,
                        left=0.065,
                        right=0.96,
                        hspace=0.445,
                        wspace=0.2)
    plt.show()


# 04/02/2022
def scaling_issue_test():
    n_obs = 5
    n_act = 5
    env = RandomEnv(n_obs, n_act, estimate_scaling=True)
    stats = RunningStats(n_obs)
    norm_stats = RunningStats(n_obs)

    NB_STEPS = 1000

    trims = np.zeros((NB_STEPS, n_obs))
    norm_trims = np.zeros_like(trims)
    for i in range(NB_STEPS):
        a = env.action_space.sample()
        trim = env.rm.dot(a)
        norm_trim = env.normalise_trim(trim)

        stats.add(trim)
        norm_stats.add(norm_trim)

        trims[i] = trim
        norm_trims[i] = norm_trim

    fig, ax = plt.subplots()
    # ax.plot(trims.T, ls=':')
    ax.plot(stats.mean, color='b', label='Mean')
    ax.plot(norm_stats.mean, color='k', label='Norm Mean')
    ax.fill_between(range(n_obs), stats.mean - stats.std, stats.mean + stats.std, color='b', alpha=0.4, label='Std')
    ax.fill_between(range(n_obs), norm_stats.mean - norm_stats.std, norm_stats.mean + norm_stats.std, color='k',
                    alpha=0.4, label='Norm Std')

    ax.legend(loc='best')
    plt.show()


# 07/02/2022
def test_saving_and_loading_dynamincs():
    test_dir = 'H:/Code/RandomEnvRLTests/testing_re_save_load'
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)

    os.makedirs(test_dir)

    sz = 5

    env = RandomEnv(sz, sz, estimate_scaling=True, model_info=None, seed=123)
    env.save_dynamics(test_dir)

    env2 = RandomEnv(sz, sz, estimate_scaling=False, model_info=None)
    # env2.load_dynamics(test_dir)

    print(env.rm)
    print(env2.rm)
    print('')
    print(env.pi)
    print(env2.pi)
    print('')
    print(env.rm.dot(env2.pi))
    print(env2.rm.dot(env.pi))
    print('')
    print(env.trim_stats)
    print(env2.trim_stats)


def save_stupid_env():
    n_obs = n_act = 20
    rm = np.diag(np.ones(n_obs))
    pi = np.diag(np.ones(n_obs))
    trim_stats = RunningStats(n_obs)

    model_info = dict(rm=rm,
                      pi=pi,
                      trim_stats=trim_stats)

    env = RandomEnv(n_obs, n_act, estimate_scaling=False, model_info=model_info)

    env.save_dynamics('H:/Code/RandomEnvRLTests')


def get_average_ep_len():
    SZ = 10
    env = RandomEnv(SZ, SZ, True)

    ep_lens = []
    for ep in range(1000):
        o = env.reset()
        for step in range(env.max_steps):
            a = env.get_optimal_action(o)
            o, r, d, _ = env.step(a)
            if d: break
        ep_lens.append(step)

    mean_ep_len = np.mean(ep_lens)
    std_ep_len = np.std(ep_lens)
    print(f'Episode length = {mean_ep_len} +- {std_ep_len}')


if __name__ == '__main__':
    # get_average_ep_len()
    quick_testing_randomenv()
# get_model_output_bounds()
# dynamic_scale_adjustment_test()
# scaling_issue_test()
# test_saving_and_loading_dynamincs()
# save_stupid_env()
