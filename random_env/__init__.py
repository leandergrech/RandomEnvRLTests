from gym.envs.registration import register

register(id='randomenv-v1', entry_point="random_env.envs:RandomEnv")
register(id='reda-v1', entry_point="random_env.envs:RandomEnvDiscreteActions")
