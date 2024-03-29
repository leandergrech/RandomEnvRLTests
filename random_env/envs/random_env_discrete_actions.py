import os
import re
from itertools import product
import yaml
import numpy as np
import gym

from random_env.envs import RandomEnv


def get_discrete_actions(n_act, act_dim=3):
    all_actions = [list(item) for item in product(*np.repeat([[item for item in range(act_dim)]], n_act, axis=0))]
    if n_act == 1:
        all_actions = [item[0] for item in all_actions]
    return all_actions


def get_reduced_discrete_actions(n_act, act_dim=3):
    all_actions = np.vstack([np.zeros(n_act),
                      np.diag(-np.ones(n_act)),
                      np.diag(np.ones(n_act))]).astype(int) + 1
    return all_actions.tolist()


class RandomEnvDiscreteActions(RandomEnv, yaml.YAMLObject):
    """
    Model dynamics creation follows that of RandomEnv exactly.
    This environment only assigns 3 discrete actions per one continuous action dimension.
    Discrete actions are -eps/0/+eps on an internal cumulative action.
    The action accumulated at timestep t, is applied to the forward dynamics of the parent
        RandomEnv.step(.)
    The action is reset to zero vector at the start of every episode
    """
    ACTION_EPS = 0.05
    # ACTION_EPS = 0.1
    AVAIL_MOM = [-ACTION_EPS, 0., ACTION_EPS]
    yaml_tag = "!REDA"

    def __init__(self, n_obs, n_act, **kwargs):
        super(RandomEnvDiscreteActions, self).__init__(n_obs, n_act, **kwargs)
        self.REWARD_SCALE = 1.
        self.TRIM_FACTOR = 20.
        self.EPISODE_LENGTH_LIMIT = 100
        self.action_space = gym.spaces.MultiDiscrete(np.repeat(3, n_act))

        self.cum_action = None
        self.action_counter = None
        self.prev_centered_action = None

    def __repr__(self):
        return f'REDA_{self.obs_dimension}obsx{self.act_dimension}act'

    def reset(self, init_state=None):
        self.cum_action = np.zeros(self.act_dimension)
        self.action_counter = np.zeros(self.act_dimension, dtype=int)
        self.prev_centered_action = np.zeros(self.act_dimension, dtype=int)
        return super(RandomEnvDiscreteActions, self).reset(init_state)

    def step(self, action):
        """
        :param action: Array of size self.act_dimension. Each action can be one of (0,1,2) - to index self.AVAIL_MOM
        """
        # Convert from [0, 1, 2] --> [-1, 0, 1]
        centered_action = np.array(action) - 1
        # '''
        # increment self.cum_action
        # '''
        # delta_action = centered_action * self.ACTION_EPS
        # self.cum_action += delta_action
        # return super(RandomEnvDiscreteActions, self).step(self.cum_action)

        # '''
        # self-centering mechanism
        # '''
        # temp = np.clip(self.action_counter, -1, 1)
        # correct_action = centered_action + np.multiply(temp, np.abs(centered_action) - 1)
        # self.action_counter += correct_action
        # delta_action = correct_action * self.ACTION_EPS
        # self.cum_action += delta_action
        # return super(RandomEnvDiscreteActions, self).step(self.cum_action)

        '''
        apply small action as is
        '''
        delta_action = centered_action * self.ACTION_EPS
        return super(RandomEnvDiscreteActions, self).step(delta_action)

    def objective(self, state):
        return -np.sqrt(np.mean(np.square(state)))

    def get_optimal_action(self, state, state_clip=None):
        opt_action = super(RandomEnvDiscreteActions, self).get_optimal_action(state, state_clip)
        # delta_action = opt_action - self.get_actual_actions()
        delta_action = opt_action
        discrete_action = np.sign(np.where(abs(delta_action) < self.ACTION_EPS,
                                np.zeros_like(delta_action), delta_action)) + 1
        return discrete_action.astype(int).tolist()

    def get_actual_actions(self):
        return self.cum_action

    @classmethod
    def to_yaml(cls, dumper, env_instance):
        return dumper.represent_mapping(cls.yaml_tag, {
            'n_obs': env_instance.obs_dimension,
            'n_act': env_instance.act_dimension,
            'rm': env_instance.rm.tolist(),
            'pi': env_instance.pi.tolist(),
            'trim_stats': env_instance.trim_stats
        })

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node)
        env = cls(d['n_obs'], d['n_act'], estimate_scaling=False)
        env.rm = np.array(d['rm'])
        env.pi = np.array(d['pi'])
        env.trim_stats = d['trim_stats']
        return env


class IREDA(RandomEnvDiscreteActions):
    def __init__(self, n_obs, n_act, **kwargs):
        assert n_obs == n_act, f"For now, restricted to n_obs==n_act. n_obs={n_obs}, n_act={n_act} not valid"
        super(IREDA, self).__init__(n_obs, n_act, **kwargs)
        self.rm = np.diag(np.ones(n_obs))
        self.pi = np.diag(np.ones(n_obs))

    def __repr__(self):
        return f'IREDA_{self.obs_dimension}obsx{self.act_dimension}act'


class REDACont(RandomEnvDiscreteActions):
    def _is_done(self):
        return False, False

    def __repr__(self):
        return f'REDACont_{self.obs_dimension}obsx{self.act_dimension}act'


class REDAClip(RandomEnvDiscreteActions):
    """
    Idea behind this environment is that we bind the states to be within state_clip l2 distance away. The reward
    returned when a state tries to go beyond state_clip is guaranteed to be lower than a reward given within bounds
    of the state.
    Note that this environment is episodic. See self.EPISODE_LENGTH_LIMIT to control the maximum allowable length
    in an episode.
    """
    yaml_tag = '!REDAClip'
    def __init__(self, n_obs, n_act, state_clip=0.0, **kwargs):
        super(REDAClip, self).__init__(n_obs, n_act, **kwargs)
        self.state_clip = state_clip

    def step(self, action):
        o_old = self.current_state.copy()

        otp1, r, d, info = super(REDAClip, self).step(action)

        R = np.sqrt(np.mean(np.square(otp1)))
        if R > self.state_clip > 0.0:
            # d = True
            # info['success'] = False
            # self.current_state = np.clip(otp1, -self.state_clip, self.state_clip)
            self.current_state = o_old
            # self.reward = r = -10.0

        return self.current_state, r, d, info

    def __repr__(self):
        return f'REDAClip_{self.state_clip}clip_{self.obs_dimension}obsx{self.act_dimension}act'

    @classmethod
    def load_from_dir(cls, load_dir):
        for file in os.listdir(load_dir):
            if RandomEnv.SAVED_MODEL_SUFFIX in file:
                state_clip, n_obs, n_act = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', file)
                self = cls(int(n_obs), int(n_act), state_clip=float(state_clip), estimate_scaling=False)
                self.load_dynamics(load_dir)

                return self

        return None

    @classmethod
    def to_yaml(cls, dumper, env_instance):
        return dumper.represent_mapping(cls.yaml_tag, {
            'n_obs': env_instance.obs_dimension,
            'n_act': env_instance.act_dimension,
            'state_clip': env_instance.state_clip,
            'rm': env_instance.rm.tolist(),
            'pi': env_instance.pi.tolist(),
            'trim_stats': env_instance.trim_stats
        })

    @classmethod
    def from_yaml(cls, loader, node):
        d = loader.construct_mapping(node)
        env = cls(d['n_obs'], d['n_act'], estimate_scaling=False)
        env.state_clip = d['state_clip']
        env.rm = np.array(d['rm'])
        env.pi = np.array(d['pi'])
        env.trim_stats = d['trim_stats']
        return env



class REDAClipCont(REDAClip):
    yaml_tag = '!REDAClipCont'
    def _is_done(self):
        _, info = super(REDAClipCont, self)._is_done()
        return info, info

    def __repr__(self):
        return f'REDAClipCont_{self.state_clip}clip_{self.obs_dimension}obsx{self.act_dimension}act'


class REDAX(RandomEnvDiscreteActions):
    def __init__(self, n_obs, n_act, **kwargs):
        super(REDAX, self).__init__(n_obs, n_act, **kwargs)
        self.actions = get_discrete_actions(n_act, 3)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def step(self, action):
        return super(REDAX, self).step(self.actions[action])


class VREDA(RandomEnvDiscreteActions):
    def __init__(self, *args, **kwargs):
        super(VREDA, self).__init__(*args, **kwargs)
        # self.observation_space = gym.spaces.Box(low=np.array([-1.0 ,-1.0 ,0.0]),
        #                                         high=np.array([1.0, 1.0, 0.15]),
        #                                         dtype=float)
        # self.velocity = 0.0
        self.observation_space.low = np.concatenate([self.observation_space.low, [0.0]])
        self.observation_space.high = np.concatenate([self.observation_space.high, [0.15]])

    def __repr__(self):
        return f'VREDA_{self.obs_dimension}obsx{self.act_dimension}act'

    def reset(self, init_state=None):
        # self.velocity = 0.0
        if init_state is not None:
            init_state = init_state[:2]
        else:
            init_state = np.random.uniform(-1.0, 1.0, self.obs_dimension)

        init_state = super(VREDA, self).reset(init_state)

        return_val = np.concatenate([init_state, [0.0]])
        return return_val

    def step(self, action):
        o = self.current_state.copy()
        otp1, r, d, info = super(VREDA, self).step(action)

        velocity = np.sqrt(np.sum(np.square(np.subtract(o, otp1))))

        otp1v = np.concatenate([otp1, [velocity]])

        # if sum(abs(otp1) > 2.0) > 0:
        #     d = True

        return otp1v, r, d, info

    def get_optimal_action(self, state, state_clip=None):
        if self.obs_dimension == 1:
            working_state = [state[0]]
        else:
            working_state = state[:self.obs_dimension]
        return super(VREDA, self).get_optimal_action(working_state, state_clip)

