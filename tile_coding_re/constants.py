N_OBS, N_ACT = 2, 2
NB_BINS = 4
NB_TILINGS = 10

NB_TRAINING_EPS = 100
NB_INIT_STEPS = 1000
EVAL_EVERY = 5

GREEDY_EPS = 0.1
LR = 0.01

par_dir = f'RandomEnvDiscreteActions_{N_OBS}obsx{N_ACT}act_{NB_BINS}bins_{NB_TILINGS}tilings_{LR}lr_{GREEDY_EPS}eps-greedy'