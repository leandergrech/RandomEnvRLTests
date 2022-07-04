import os
import numpy as np
import matplotlib.pyplot as plt

from tile_coding_re.tile_coding import get_tilings_from_env, QValueFunction2
from tile_coding_re.utils import TrajBuffer
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions

env = RandomEnvDiscreteActions(2, 2)
tilings = get_tilings_from_env(env, 4, 4)
actions = get_discrete_actions(2)

buffer = TrajBuffer()

for ep in range(5):
    buffer.reset()
    o = env.reset()
    d = False
    while not d:
        a = env.get_optimal_action(o)
        otp1, r, d, _ = env.step(a)

        buffer.add(o, a, r)
        o = otp1

    fig, ax1 = plt.subplots()
    ax1.plot(buffer.o)
plt.show()
