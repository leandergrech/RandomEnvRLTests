import os
import numpy as np

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


def get_tilings_from_env(env, nb_tilings, nb_bins):
    n_obs = env.obs_dimension

    ranges = [[lo, hi] for lo, hi in zip(env.observation_space.low, env.observation_space.high)]
    range_size = abs(np.subtract(*ranges[0]))

    bins = np.tile(nb_bins, (nb_tilings, n_obs))

    available_offsets = np.linspace(0, range_size / nb_bins, nb_tilings + 1)[:-1]  # symmetrical tiling offsets
    offsets = np.repeat(available_offsets, n_obs).reshape(-1, n_obs)

    tilings = create_tilings(ranges, nb_tilings, bins, offsets)

    return tilings


class QValueFunction:
    def __init__(self, tilings, actions, lr):
        self.tilings = tilings
        self.nb_tilings = len(self.tilings)
        if type(actions) == np.ndarray:
            actions = actions.tolist()
        self.actions = actions
        self.lr = lr
        state_sizes = [tuple(len(splits) + 1 for splits in tiling)
                            for tiling in self.tilings]
        # self.q_tables = np.array([np.zeros(shape=(state_size + (len(self.actions),)))
        #                  for state_size in state_sizes])
        # Initialize to random Q-values
        self.q_tables = np.array([-np.random.rand(*(state_size + (len(self.actions),)))
                         for state_size in state_sizes])

    def value(self, state, action):
        codings = get_tile_coding(state, self.tilings)
        action_idx = self.actions.index(list(action))
        value = 0
        for coding, q_table in zip(codings, self.q_tables):
            value += q_table[tuple(coding)+(action_idx,)]
        return value / self.nb_tilings

    def update(self, state, action, target):
        codings = get_tile_coding(state, self.tilings)
        action_idx = self.actions.index(list(action))
        # for i, coding in enumerate(codings):
        for coding, q_table in zip(codings, self.q_tables):
            # q_index = (i,)+tuple(coding)+(action_idx,)
            q_index = tuple(coding)+(action_idx,)
            delta = target - q_table[q_index]
            q_table[q_index] += self.lr * delta

    def save(self, path):
        if '.npz' in path:
            path, filename = os.path.split(path)
        else:
            filename = 'qvf.npz'
        if not os.path.exists(path):
            os.makedirs(path)

        np.savez(os.path.join(path, filename),
                 tilings=self.tilings,
                 actions=self.actions,
                 q_tables=self.q_tables,
                 lr=self.lr)

    @staticmethod
    def load(path):
        if '.npz' not in path:
            path = os.path.join(path, 'qvf.npz')
        npz_archive = np.load(path)
        qvf = QValueFunction(npz_archive['tilings'],
                             npz_archive['actions'],
                             float(npz_archive['lr']))
        qvf.q_tables = npz_archive['q_tables']
        return qvf

    def greedy_action(self, state):
        q_vals = [self.value(state, a_) for a_ in self.actions]
        return self.actions[np.argmax(q_vals)]
