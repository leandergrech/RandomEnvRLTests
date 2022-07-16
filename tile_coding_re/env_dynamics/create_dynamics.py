from random_env.envs import VREDA, RandomEnvDiscreteActions as REDA

env = REDA(2, 2)
for k, v in env.model_info.items():
    print(f'{k}:\n{v}\n')
print(env.rm.dot(env.pi))
env.save_dynamics('.')
