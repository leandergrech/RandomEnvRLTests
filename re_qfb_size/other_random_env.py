import os
import numpy as np
import pickle as pkl
import re
from random_env.envs import RandomEnv


class NoisyRE(RandomEnv):
    def __init__(self, n_obs, n_act, action_noise=0.01, estimate_scaling=True, **kwargs):
        super(NoisyRE, self).__init__(n_obs, n_act, estimate_scaling=estimate_scaling, **kwargs)
        self.action_noise = float(action_noise)

    def step(self, action, deterministic=False):
        if not deterministic:
            action += np.random.normal(0, self.action_noise, self.act_dimension)

        return super(NoisyRE, self).step(action)

    def __repr__(self):
        return f'NoisyRE_{self.obs_dimension}obsx{self.act_dimension}act_{self.action_noise}noise'

    @classmethod
    def load_from_dir(cls, load_dir):
        for file in os.listdir(load_dir):
            if RandomEnv.SAVED_MODEL_SUFFIX in file:
                n_obs, n_act, action_noise = re.findall(r'[-+]?(?:\d*\.\d+|\d+)', file)
                n_obs, n_act, action_noise = int(n_obs), int(n_act), float(action_noise)
                self = cls(n_obs, n_act, estimate_scaling=False, action_noise=action_noise)
                self.load_dynamics(load_dir)

                return self

        return None


class NoisyClipRE(NoisyRE):
    def step(self, action, deterministic=False):
        old_o = self.current_state.copy()
        otp1, r, d, info = super(NoisyClipRE, self).step(action, deterministic)
        if max(abs(otp1)) > 1:
            r = -10.0
            self.current_state = otp1 = old_o

        return otp1, r, d, info

    def __repr__(self):
        return f'NoisyClipRE_{self.obs_dimension}obsx{self.act_dimension}act_{self.action_noise}noise'

if __name__ == '__main__':
    env = NoisyClipRE(10, 10, 0.08)
    env.save_dynamics('/home/leander/code/RandomEnvRLTests/common_envs')