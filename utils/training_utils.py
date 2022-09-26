import numpy as np
import yaml
from random_env.envs.random_env_discrete_actions import get_discrete_actions


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
        self.r_ = []
        self.g_ = []

    def __len__(self):
        return len(self.o_)

    def __bool__(self):
        if self.o_:
            return True
        else:
            return False

    def add(self, state, action, reward):
        self.o_.append(state.copy())
        self.a_.append(action.copy())
        self.r_.append(reward)

    def get_returns(self):
        return np.flip(np.cumsum(np.flip(self.r_))).tolist()

    def calculate_returns(self, gamma=1.0):
        r = self.r_
        N = len(r)
        gammas = [gamma**item for item in range(N)]

        self.g_ = [np.sum(np.multiply(r[i:], gammas[:N-i])) for i in range(N)]


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
    def a(self):
        return self.a_

    @property
    def r(self):
        return self.r_

    @property
    def g(self):
        return self.g_



