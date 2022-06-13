import numpy as np
from random_env.envs import RandomEnv


def create_tilings(ranges, nb_tilings, bins, offsets):
    """
    :param ranges: range of each feature; ex: x:[-1,1],y:[2,5] -> [[-1,1],[2,5]]
    :param nb_tilings: number of tilings
    :param bins: bin size for each tiling; ex: [[10,10],[10,10]]: 2 tilings * [x_bin, y_bin]
    :param offsets: offset for each tiling dimension
    :return:
    """
    tilings = []
    for tiling_idx in range(nb_tilings):
        tiling_bins = bins[tiling_idx]
        tiling = []
        # for each dimension
        for feature_idx in range(len(ranges)):
            feature_range = ranges[feature_idx]
            feature_offset = offsets[tiling_idx, feature_idx]
            feature_bins = bins[tiling_idx, feature_idx]
            feature_tiling = np.linspace(feature_range[0], feature_range[1],
                                         feature_bins + 1)[1:-1] + feature_offset
            tiling.append(feature_tiling)
        tilings.append(tiling)
    return np.array(tilings)

N_OBS, N_ACT = 2, 2
env = RandomEnv(n_obs=N_OBS, n_act=N_ACT)

NB_BINS = 10
NB_TILINGS = 4
ranges = [[lo, hi] for lo, hi in zip(env.observation_space.low, env.observation_space.high)]
bins = np.tile(NB_BINS, (NB_TILINGS, N_OBS))

# assuming symmetry among feature dimensions
range_size = abs(np.subtract(*ranges[0]))
available_offsets = np.linspace(0, range_size/NB_BINS, NB_TILINGS + 1)[:-1]
offsets = np.repeat(available_offsets, N_OBS).reshape(-1, N_OBS)

tilings = create_tilings(ranges, NB_TILINGS, bins, offsets)
print(tilings)
print(tilings.shape)
