N_OBS, N_ACT = 2, 2
NB_BINS = 4
NB_TILINGS = 4

NB_TRAINING_EPS = 100
# NB_INIT_STEPS = 1000
SAVE_EVERY = 10

GAMMA = 0.99
GREEDY_EPS = lambda i: min(1 - i / 50, 0.1)
LR = 1e-2

# par_dir = f'REDA_{N_OBS}obsx{N_ACT}act_{NB_BINS}bins_{NB_TILINGS}tilings_{LR}lr_{GREEDY_EPS}eps-greedy_{GAMMA}gamma'
par_dir = f'REDA_{N_OBS}obsx{N_ACT}act_{NB_BINS}bins_{NB_TILINGS}tilings_{LR}lr_{GAMMA}gamma'
