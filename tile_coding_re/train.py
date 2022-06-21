import os
import shutil

from itertools import product

import numpy as np
from random_env.envs import RandomEnvDiscreteActions
from tile_coding_re.tile_coding import create_tilings, QValueFunction, TrajBuffer
from constants import *

'''
Create RE to train on
'''
env = RandomEnvDiscreteActions(n_obs=N_OBS, n_act=N_ACT)

'''
Create tilings
'''
ranges = [[lo, hi] for lo, hi in zip(env.observation_space.low, env.observation_space.high)]
bins = np.tile(NB_BINS, (NB_TILINGS, N_OBS))
range_size = abs(np.subtract(*ranges[0]))
available_offsets = np.linspace(0, range_size/NB_BINS, NB_TILINGS + 1)[:-1] # symmetrical tiling offsets
offsets = np.repeat(available_offsets, N_OBS).reshape(-1, N_OBS)
tilings = create_tilings(ranges, NB_TILINGS, bins, offsets)

'''
Create tabular Q-function
'''
all_actions = [list(item) for item in product(*np.repeat([[0, 1, 2]], N_ACT, axis=0))]
qvf = QValueFunction(tilings, all_actions, lr=LR)

'''
Directory  handling
'''
par_dir = f'{repr(env)}_{NB_BINS}bins_{NB_TILINGS}tilings_{LR}lr_{GREEDY_EPS}eps-greedy'
save_path = os.path.join(par_dir, 'saves')
if os.path.exists(par_dir):
    print(f"You're gonna overwrite files in: {par_dir}")
    ans = input('You sure? [Y/n]')
    if ans.lower() == 'y' or ans == '':
        shutil.rmtree(par_dir)
os.makedirs(save_path)

'''
Save dynamics
'''
env.save_dynamics(save_path)

'''
Training
'''
T = 0
buffer = TrajBuffer()
for ep in range(NB_TRAINING_EPS):
    o = env.reset()
    buffer.reset()
    d = False
    while not d:
        # epsilon greedy
        if np.random.rand() < GREEDY_EPS or T < NB_INIT_STEPS:
            a = env.action_space.sample()
        else:
            # Q-learning - greedy selection of action with the largest value
            vals = [qvf.value(o, a_) for a_ in all_actions]     # evaluate all actions first
            a = all_actions[np.argmax(vals)]

        otp1, r, d, _ = env.step(a)
        buffer.add(o, a, r)

        o = otp1
        T += 1

    # consume buffer - on-policy
    while buffer:
        tup = buffer.pop_target_tuple()
        qvf.update(*tup)

    # logging
    if (ep+1) % EVAL_EVERY == 0:
        print(f"Episode {ep+1}")
        qvf.save(os.path.join(save_path, f'{ep+1}ep'))