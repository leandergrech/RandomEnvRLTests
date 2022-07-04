import numpy as np
import matplotlib.pyplot as plt
from random_env.envs import RandomEnv

env_sz = 10

env = RandomEnv(env_sz, env_sz, False)
env.load_dynamics('.')

obses = []
acts = []
o = env.reset()
d = False
while not d:
    obses.append(np.copy(o))
    a = env.get_optimal_action(o)
    acts.append(np.copy(a))
    o, _, d, _ = env.step(a)

fig, ax = plt.subplots()
ax.plot(obses, color='b')
ax.plot(acts, color='r')
plt.show()
