import numpy as np
import os
from tile_coding_re.mc_method.tile_coding import get_tilings_from_env
from random_env.envs import RandomEnvDiscreteActions

nb_tilings = 4
nb_bins = 4

env = RandomEnvDiscreteActions(2, 2)
