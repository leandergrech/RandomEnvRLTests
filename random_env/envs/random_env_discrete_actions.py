from itertools import product
import gym
import numpy as np
from random_env.envs import RandomEnv


def get_discrete_actions(n_act, act_dim=3):
    all_actions = [list(item) for item in product(*np.repeat([[item for item in range(act_dim)]], n_act, axis=0))]
    if n_act == 1:
        all_actions = [item[0] for item in all_actions]
    return all_actions


class RandomEnvDiscreteActions(RandomEnv):
    """
    Model dynamics creation follows that of RandomEnv exactly.
    This environment only assigns 3 discrete actions per one continuous action dimension.
    Discrete actions are -eps/0/+eps on an internal cumulative action.
    The action accumulated at timestep t, is applied to the forward dynamics of the parent
        RandomEnv.step(.)
    The action is reset to zero vector at the start of every episode
    """
    ACTION_EPS = 0.1
    AVAIL_MOM = [-ACTION_EPS, 0., ACTION_EPS]

    def __init__(self, *args, **kwargs):
        super(RandomEnvDiscreteActions, self).__init__(*args, **kwargs)
        self.action_space = gym.spaces.MultiDiscrete(np.repeat(3, self.act_dimension))
        self.cum_action = None
        self.REWARD_SCALE = 1.
        self.TRIM_FACTOR = 2.
        self.EPISODE_LENGTH_LIMIT = 30

    def __repr__(self):
        return f'REDA_{self.obs_dimension}obsx{self.act_dimension}act'

    def reset(self, init_state=None):
        self.cum_action = np.zeros(self.act_dimension)
        return super(RandomEnvDiscreteActions, self).reset(init_state)

    def step(self, action):
        """
        :param action: Array of size self.act_dimension. Each action can be one of (0,1,2) - to index self.AVAIL_MOM
        """
        for i, a in enumerate(action):
            assert a in (0, 1, 2), f"Invalid action at index {i}: {a}"
        delta_action = (np.array(action) - 1) * self.ACTION_EPS
        self.cum_action += delta_action
        return super(RandomEnvDiscreteActions, self).step(self.cum_action)

    def get_optimal_action(self, *args, **kwargs):
        opt_action = super(RandomEnvDiscreteActions, self).get_optimal_action(*args, **kwargs)
        delta_action = opt_action - self.get_actual_actions()
        return np.sign(np.where(abs(delta_action) < self.ACTION_EPS,
                                np.zeros_like(delta_action), delta_action)) + 1

    def get_actual_actions(self):
        return self.cum_action
