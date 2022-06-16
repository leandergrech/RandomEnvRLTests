import numpy as np
from random_env.envs import RandomEnv

'''
Implementation from: 
https://towardsdatascience.com/reinforcement-learning-tile-coding-implementation-7974b600762b
'''


def create_tilings(ranges, nb_tilings, bins, offsets):
    """
    :param ranges: range of each feature; ex: x:[-1,1],y:[2,5] -> [[-1,1],[2,5]]
    :param nb_tilings: number of tilings
    :param bins: bin size for each tiling; ex: [[10,10],[10,10]]: 2 tilings * [x_bin, y_bin]
    :param offsets: offset for each tiling dimension
    :return:
    """
    assert nb_tilings == len(bins), "nb_tilings and bins info does not match" 
    assert nb_tilings == len(offsets), "nb_tilings and offsets info does not match" 
    tilings = []
    for i in range(nb_tilings):
        tiling = []
        # for each dimension
        for j in range(len(ranges)):
            feature_range = ranges[j]
            feature_bins = bins[i, j]
            feature_offset = offsets[i, j]
            feature_tiling = np.linspace(feature_range[0], feature_range[1],
                                         feature_bins + 1)[1:-1] + feature_offset
            tiling.append(feature_tiling)
        tilings.append(tiling)
    return np.array(tilings)


def get_tile_coding(feature, tilings):
    """
    feature: sample feature with multiple dims to encode
    tilings: the tiling edges
    return: encoding for feature within tilings
    """
    nb_dims = len(feature)
    nb_tilings = len(tilings)
    feature_codings = []
    for i in range(nb_tilings):
        feature_coding = []
        for j in range(nb_dims):
            x = feature[j]
            tiles = tilings[i, j]
            coding = np.digitize(x, tiles)
            feature_coding.append(coding)
        feature_codings.append(feature_coding)
    return np.array(feature_codings)


class QValueFunction:
    def __init__(self, tilings, actions, lr):
        self.tilings = tilings
        self.nb_tilings = len(self.tilings)
        self.actions = actions
        self.lr = lr
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling)
                            for tiling in self.tilings]
        self.q_tables = [np.zeros(shape=(state_size + (len(self.actions),)))
                         for state_size in self.state_sizes]

    def value(self, state, action):
        codings = get_tile_coding(state, self.tilings)
        action_idx = self.actions.index(action)
        value = 0
        for coding, q_table in zip(codings, self.q_tables):
            value += q_table[tuple(coding)+(action_idx,)]
        return value / self.nb_tilings

    def update(self, state, action, target):
        codings = get_tile_coding(state, self.tilings)
        action_idx = self.actions.index(action)
        for coding, q_table in zip(codings, self.q_tables):
            delta = target - q_table[tuple(coding)+(action_idx,)]
            q_table[tuple(coding)+(action_idx,)] += self.lr * delta



N_OBS, N_ACT = 10, 10
env = RandomEnv(n_obs=N_OBS, n_act=N_ACT)

NB_BINS = 10
NB_TILINGS = 6
ranges = [[lo, hi] for lo, hi in zip(env.observation_space.low, env.observation_space.high)]
bins = np.tile(NB_BINS, (NB_TILINGS, N_OBS))

# assuming symmetry among feature dimensions
range_size = abs(np.subtract(*ranges[0]))
available_offsets = np.linspace(0, range_size/NB_BINS, NB_TILINGS + 1)[:-1]
offsets = np.repeat(available_offsets, N_OBS).reshape(-1, N_OBS)

tilings = create_tilings(ranges, NB_TILINGS, bins, offsets)
o = env.reset()

print(tilings)
print(o)
print(get_tile_coding(o, tilings))
