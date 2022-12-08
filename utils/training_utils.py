import os.path
from abc import ABC
import numpy as np
import yaml
from datetime import datetime as dt
from random_env.envs.random_env_discrete_actions import get_discrete_actions

def get_random_alpha_numeral(sz=10):
    chars = ''.join([str(item) for item in range(10)]) + 'abcdefghij'
    ret = []
    for _ in range(sz):
        ret.append(chars[np.random.choice(len(chars))])
    return ''.join(ret)

def argmax(arr):
    return max((x, i) for i, x in enumerate(arr))[1]

def init_label(label):
    if label:
        return f'{label}_'
    else:
        return ''


class CustomFunc(yaml.YAMLObject):
    yaml_tag = '!CustomFunc'

    def __init__(self, pow=0.75):
        self.pow = pow

    def __call__(self, ep_idx):
        return 1/((ep_idx + 1) ** self.pow)

    def __repr__(self):
        return f'1/((ep_idx + 1) ** {self.pow}'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, {
            'func': repr(data),
            'pow': data.pow
        })

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node)
        return CustomFunc(pow=d['pow'])


class Constant(yaml.YAMLObject):
    yaml_tag = '!Constant'

    def __init__(self, val, label=''):
        self.val = val
        self.label = init_label(label)

    def __call__(self, *args, **kwargs):
        return self.val

    def __repr__(self):
        return f'{self.label}Constant_{self.val}'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, {
            'val': data.val,
            'label': data.label
        })

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node)
        return Constant(val=d['val'], label=d['label'])


class ExponentialDecay(yaml.YAMLObject):
    yaml_tag = '!ExponentialDecay'

    def __init__(self, init, halflife, label=None):
        self.init = init
        self.halflife = halflife
        self.label = init_label(label)

    def __call__(self, t):
        return self.init / (1. + (t / self.halflife))

    def __repr__(self):
        return f'{self.label}ExponentialDecay_init-{self.init}_halflife-{self.halflife}'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, {
            'init': data.init,
            'halflife': data.halflife,
            'label': data.label
        })

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node)
        return ExponentialDecay(init=d['init'], halflife=d['halflife'],
                                label=d['label'])


class LinearDecay(yaml.YAMLObject):
    yaml_tag = '!LinearDecay'

    def __init__(self, init, final, decay_steps, label=None):
        self.init = init
        self.final = final
        self.decay_steps = decay_steps
        self.label = init_label(label)

    def __call__(self, t):
        return self.final + (self.init - self.final) * max(0, 1 - t/self.decay_steps)

    def __repr__(self):
        return f'{self.label}LinearDecay_init-{self.init}_final-{self.final}_decaysteps-{self.decay_steps}'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, {
            'init': data.init,
            'final': data.final,
            'decay_steps': data.decay_steps,
            'label': data.label
        })

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node)
        return LinearDecay(init=d['init'], final=d['final'],
                           decay_steps=d['decay_steps'], label=d['label'])


class StepDecay(yaml.YAMLObject):
    yaml_tag = '!StepDecay'

    def __init__(self, init, decay_rate, decay_every, label=None):
        self.init = init
        self.decay_rate = decay_rate
        self.decay_every = decay_every
        self.label = init_label(label)

    def __call__(self, t):
        return self.init * self.decay_rate**(t//self.decay_every)

    def __repr__(self):
        return f'{self.label}StepDecay_init-{self.init}_decayrate-{self.decay_rate}_decayevery-{self.decay_every}'

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, {
            'init': data.init,
            'decay_rate': data.decay_rate,
            'decay_every': data.decay_every,
            'label': data.label
        })

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node)
        return StepDecay(init=d['init'], decay_rate=d['decay_rate'],
                         decay_every=d['decay_every'], label=d['label'])


for cls in (CustomFunc, Constant, ExponentialDecay, LinearDecay, StepDecay):
    yaml.add_representer(cls, cls.to_yaml)
    yaml.add_constructor(cls.yaml_tag, cls.from_yaml)


def circular_initial_state_distribution_2d():
    r = np.random.normal(0.9, 0.1)
    theta = 2 * np.pi * np.random.rand()

    return np.array([r * np.cos(theta), r * np.sin(theta)])
    # return np.random.uniform(-1, 1, 3)


def nball_uniform_sample(N, rlow, rhigh):
    X = np.random.uniform(-1, 1, N)  # Sample point from N-dimensional space
    R = np.sqrt(np.sum(np.square(X)))  # Sampled point lies on N-ball with radius R

    A = np.random.uniform(rlow, rhigh)  # Magnitude sampled from uni. dist. to get rlow-rhigh band
    return (X * A) / R  # Normalise and multiply by sampled magnitude


class InitSolvableState:
    """
        Initialised by passing the REDA-type environment.
        Calling its instance will return a solvable initial state.
        Ensures that if n_obs > n_act, the initial state will lie on a solvable hyperplane with the same
        dimensions as the action space.
    """
    def __init__(self, env, init_thresh=0.5):
        self.env = env
        self.actions = get_discrete_actions(env.act_dimension, 3)
        self.nb_actions = len(self.actions)
        self.init_state_mag_thresh = init_thresh  # Initial state guaranteed to have magnitude larger than this

    def __call__(self, *args, **kwargs):
        # Initialise state within threshold n-ball - more variety in final state since actions are discrete
        init_state = nball_uniform_sample(self.env.obs_dimension, 0.0, self.env.GOAL / 2)

        # Move away from the optimal state with a random agent
        self.env.reset(init_state)
        while True:
            a = self.env.action_space.sample()
            otp1, *_ = self.env.step(a)
            if np.sqrt(np.mean(np.square(otp1))) > self.init_state_mag_thresh:
                return otp1

class TrajSimple:
    def __init__(self):
        self.data = None
        self.reset()

    def reset(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        if self.data:
            return True
        else:
            return False

    def __iter__(self):
        return self.data

    def add(self, datum):
        # if isinstance(datum, np.ndarray):
        #     datum = datum.tolist()
        # if isinstance(datum, list):
        #     self.data.append(datum.copy())
        # else:
        self.data.append(datum)

    def get_returns(self):
        return np.flip(np.cumsum(np.flip(self.data))).tolist()

    def pop(self, idx):
        return self.data.pop(idx)


class TrajBuffer:
    def __init__(self, discount=1.0):
        self.o_ = None
        self.a_ = None
        self.r_ = None
        self.g_ = None
        self.discount = discount
        self.reset()

    def reset(self):
        self.o_ = []
        self.a_ = []
        self.otp1_ = []
        self.r_ = []
        self.g_ = []

    def __len__(self):
        return len(self.o_)

    def __bool__(self):
        if self.o_:
            return True
        else:
            return False

    def add(self, state, action, reward, next_state=None):
        self.o_.append(state.copy())
        self.a_.append(action.copy())
        self.r_.append(reward)
        if next_state is not None:
            self.otp1_.append(next_state.copy())

    def get_returns(self):
        return np.flip(np.cumsum(np.flip(self.r_))).tolist()

    def calculate_returns(self, gamma=1.0):
        r = self.r_
        N = len(r)
        gammas = [gamma**item for item in range(N)]

        self.g_ = [np.sum(np.multiply(r[i:], gammas[:N-i])) for i in range(N)]

    def sample_batch(self, batch_size):
        if batch_size > len(self):
            return None
        temp = np.random.choice(len(self), batch_size).astype(int)
        return np.array(self.o_)[temp], np.array(self.a_)[temp], np.array(self.r_)[temp], np.array(self.otp1_)[temp]

    def pop_target_tuple(self):
        if not self.g_:
            self.calculate_returns(self.discount)

        if not self.o_:
            raise ValueError('No items in buffer')

        o = self.o_.pop(0)
        a = self.a_.pop(0)
        r = self.r_.pop(0)
        g = self.g_.pop(0)

        return o, a, g

    @property
    def o(self):
        return self.o_

    @property
    def otp1(self):
        return self.otp1_

    @property
    def a(self):
        return self.a_

    @property
    def r(self):
        return self.r_

    @property
    def g(self):
        return self.g_


class QFuncBaseClass(ABC):
    def __init__(self, *args, **kwargs):
        self.n_discrete_actions = None

    def value(self, state, action_idx: int):
        raise NotImplementedError

    def update(self, state, action_idx, target, lr):
        raise NotImplementedError

    def greedy_action(self, state):
        return argmax([self.value(state, a_) for a_ in range(self.n_discrete_actions)])

    def save(self, save_path):
        raise NotImplementedError

    @staticmethod
    def load(load_path):
        raise NotImplementedError


def eps_greedy(state: np.ndarray, qfunc: QFuncBaseClass, epsilon: float) -> int:
    nb_actions = qfunc.n_discrete_actions
    if np.random.rand() < epsilon:
        return np.random.choice(nb_actions)
    else:
        return qfunc.greedy_action(state)


def boltzmann(state: np.ndarray, qfunc: QFuncBaseClass, tau: float) -> int:
    nb_actions = qfunc.n_discrete_actions
    qvals_exp = np.exp([qfunc.value(state, a_) / tau for a_ in range(nb_actions)])
    qvals_exp_sum = np.sum(qvals_exp)

    cum_probas = np.cumsum(qvals_exp / qvals_exp_sum)
    return np.searchsorted(cum_probas, np.random.rand())


from stable_baselines3.common.callbacks import CheckpointCallback


class BestSaveCheckpointCallBack(CheckpointCallback):
    def __init__(self, save_freq, save_dir, **kwargs):
        super(BestSaveCheckpointCallBack, self).__init__(save_freq, save_dir, **kwargs)
        self.save_freq = save_freq
        self.save_dir = save_dir

        self.best_average_return = -np.inf

    def _on_step(self) -> bool:
        super(BestSaveCheckpointCallBack, self)._on_step()

        if not self.n_calls % self.locals['eval_freq'] == 0:
            return True

        _env = self.locals['eval_env']
        total_reward = 0
        total_steps = 0
        successes = []

        nb_eps = self.locals['n_eval_episodes']
        for ep in range(nb_eps):
            o = _env.reset()
            _d = False
            _step = 0
            while not _d:
                _a = self.model.predict(observation=o, deterministic=True)[0]
                _otp1, _r, _d, _info = _env.step(_a)

                _step += 1
                _o = _otp1.copy()

                total_reward += _r
            successes.append(_info.get('success', False))
            total_steps += _step

        average_reward = total_reward / total_steps
        average_return = total_reward / nb_eps
        average_ep_len = total_steps / nb_eps
        average_success = np.mean(successes) * 100.0

        self.logger.record('eval2/total_reward', total_reward)
        self.logger.record('eval2/average_reward', average_reward)
        self.logger.record('eval2/average_return', average_return)
        self.logger.record('eval2/average_ep_len', average_ep_len)
        self.logger.record('eval2/average_success(%)', average_success)


        average_return = average_reward * average_ep_len
        if self.best_average_return <= average_return:
            self.best_average_return = average_return
            self.model.save(os.path.join(self.save_dir, f'{self.name_prefix}_{self.num_timesteps}_steps.zip'))