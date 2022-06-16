import numpy as np
from random_env.envs import RandomEnv


class RandomEnvDiscreteActions(RandomEnv):
    """
    Model dynamics creation follows that of RandomEnv exactly.
    This environment only assigns 3 discrete actions per one continuous action dimension.
    Discrete actions are -eps/0/+eps on an internal cumulative action.
    The action accumulated at timestep t, is applied to the forward dynamics of the parent
        RandomEnv.step(.)
    The action is reset to zero vector at the start of every episode
    """
    def __int__(self, *args, **kwargs):
        super(RandomEnvDiscreteActions, self).__int__(*args, **kwargs)
        self.cum_action = np.zeros(self.act_dimension)