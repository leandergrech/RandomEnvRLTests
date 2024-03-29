N_OBS, N_ACT = 2, 2
NB_BINS = 4
NB_TILINGS = 4

NB_TRAINING_EPS = 2000
NB_INIT_STEPS = 2000
SAVE_EVERY = 50

GREEDY_EPS = 0.1
EXPLORATION_DECAY = 0.99  # evaluated as: GREEDY_EPS * (EXPLORATION_DECAY**ep)
LR = 0.01

par_dir = f'REDA_{N_OBS}obsx{N_ACT}act_{NB_BINS}bins_{NB_TILINGS}tilings_{LR}lr_{GREEDY_EPS}eps-greedy'
