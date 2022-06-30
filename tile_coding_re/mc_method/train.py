import os

import numpy as np
from random_env.envs import RandomEnvDiscreteActions, get_discrete_actions
from tile_coding_re.tile_coding import get_tilings_from_env, QValueFunction2
from tile_coding_re.utils import TrajBuffer
from tile_coding_re.mc_method.constants import *

from tqdm import trange

"""
Epsilon-greedy on-policy MC method
Experience divided into episodes and all episodes terminate
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
qvf = QValueFunction2(tilings, all_actions, lr=LR)

'''
Directory  handling
'''
par_dir = f'{repr(env)}_{NB_BINS}bins_{NB_TILINGS}tilings_{LR}lr_{GREEDY_EPS}eps-greedy'
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
save_path = os.path.join(par_dir, 'saves')
os.makedirs(save_path)
# save training parameters
with open('constants.py', 'r') as readfile, open(os.path.join(par_dir, 'info.md'), 'a') as writefile:
    for line in readfile:
        writefile.write(line)

'''
Save dynamics
'''
env.save_dynamics(save_path)

'''
Training
'''
T = 0
buffer = TrajBuffer()
ep_lens = np.zeros(NB_TRAINING_EPS)
for ep in trange(NB_TRAINING_EPS):
    # o = np.clip(env.reset() * 0.5, -1., 1.)
    o = env.reset()
    # o = np.random.uniform(-0.2, 0.2, env.obs_dimension)
    buffer.reset()
    d = False
    while not d:
        # epsilon greedy
        if np.random.rand() < GREEDY_EPS*(EXPLORATION_DECAY**ep) or T < NB_INIT_STEPS:
            a = env.action_space.sample()
        else:
            # greedy selection of action with the largest value
            a = qvf.greedy_action(o)

        otp1, r, d, info = env.step(a)
        if info['success']:
            r = 0.
        buffer.add(o, a, r)

        o = otp1
        T += 1

    ep_lens[ep] = len(buffer)

    # consume episode - on-policy
    while buffer:
        tup = buffer.pop_target_tuple()
        qvf.update(*tup)

    # logging
    if (ep+1) % SAVE_EVERY == 0 or ep == 0:
        # print(f"\rEpisode {ep+1:5d}", end='')
        qvf.save(os.path.join(save_path, f'{ep+1}ep'))

np.save(os.path.join(par_dir, 'training_ep_lens.npy'), ep_lens)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(ep_lens)
ax.set_xlabel('Training episodes')
ax.set_ylabel('Emalta uom rso europisode length')
plt.savefig(os.path.join(par_dir, 'training_ep_lens.png'))
plt.show()
