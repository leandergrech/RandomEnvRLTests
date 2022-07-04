import numpy as np
from random_env.envs import RandomEnv

for env_sz in np.arange(5, 15):
    env = RandomEnv(env_sz, env_sz, False)
    env.rm = np.diag(np.ones(env_sz))
    env.pi = np.diag(np.ones(env_sz))
    env.save_dynamics('.')
