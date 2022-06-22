import os
import shutil

import numpy as np
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions
from tile_coding_re.sarsa.tile_coding import get_tilings_from_env, QValueFunction
from tile_coding_re.sarsa.utils import TrajBuffer
from tile_coding_re.sarsa.constants import *

"""
Sarsa
Epsilon-greedy on-policy TD-control
"""

'''
Create RE to train on
'''
env = RandomEnvDiscreteActions(n_obs=N_OBS, n_act=N_ACT)

'''
Create tilings
'''
tilings = get_tilings_from_env(env, NB_TILINGS, NB_BINS)

'''
Create tabular Q-function
'''
all_actions = get_discrete_actions(N_ACT)
qvf = QValueFunction(tilings, all_actions, lr=LR)

'''
Directory  handling
'''
par_dir = f'{repr(env)}_{NB_BINS}bins_{NB_TILINGS}tilings_{LR}lr_{GREEDY_EPS}eps-greedy'
save_path = os.path.join(par_dir, 'saves')
if os.path.exists(par_dir):
    print(f"Run with these hparams already made: {par_dir}")
    ans = input('Continue? [Y/n]')
    if ans.lower() == 'y' or ans == '':
        title_found = False
        title_idx = 1
        temp = None
        while not title_found:
            temp = par_dir + f'_{title_idx}'
            title_found = not os.path.exists(temp)
            title_idx += 1
        par_dir = temp
        print(f'Continuing in directory: {par_dir}')
    else:
        exit(42)


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


def policy(state):
    if np.random.rand() < GREEDY_EPS:  # or T < NB_INIT_STEPS:
        return env.action_space.sample()
    else:
        # greedy selection of action with the largest value
        return qvf.greedy_action(state)


for ep in range(NB_TRAINING_EPS):
    buffer.reset()
    d = False

    o = env.reset()
    a1 = policy(o)
    while not d:
        otp1, r, d, info = env.step(a1)
        a2 = policy(otp1)

        if info['success']:
            target = 0.0
        else:
            target = r + GAMMA * qvf.value(otp1, a2)

        qvf.update(o, a1, target)

        o = otp1
        a1 = a2

    # logging
    if (ep+1) % EVAL_EVERY == 0:
        print(f"Episode {ep+1}")
        qvf.save(os.path.join(save_path, f'{ep+1}ep'))
