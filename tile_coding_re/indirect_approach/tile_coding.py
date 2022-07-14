import os
import numpy as np

'''
Implementation from: 
https://towardsdatascience.com/reinforcement-learning-tile-coding-implementation-7974b600762b
'''

# this version has the problem that one tiling has one less division that the rest
'''def create_tilings(ranges, nb_tilings, bins, offsets):
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
            feature_tiling = np.linspace(feature_range[0], feature_range[1], feature_bins + 1) + feature_offset
            feature_tiling = np.array([item for item in feature_tiling if feature_range[0] < item < feature_range[1]])
            tiling.append(feature_tiling)
        tilings.append(tiling)
    return np.array(tilings)'''


def create_tilings(ranges, nb_tilings, bins):
    """
    Tilings with automatic symmetrical offsets and equal number of partitions per tiling.
    The following formulas were derived from the Sutton&Barto diagram on p.217
    Let:
        R = feature range
        T = number of tilings
        B = number of bins
    Unknowns:
        W = width of tiling
        O = width of one offset
    We know that:
        W = R + (T - 1)O    ...(1)
        O = W/(B.T)         ...(2)
    Substitute (1) in (2) and simplify:
        O.(B.T) = R + (T - 1)O
        O.(B.T - (T-1)) = R
        O = R / (T(B - 1) + 1))

    :param ranges: range of each feature; ex: x:[-1,1],y:[2,5] -> [[-1,1],[2,5]]
    :param nb_tilings: number of tilings
    :param bins: bin size for each tiling; ex: [[10,10],[10,10]]: 2 tilings * [x_bin, y_bin]
    :param offsets: offset for each tiling dimension
    :return:
    """
    assert nb_tilings == len(bins), "nb_tilings and bins info does not match"
    nb_features = len(ranges)
    tilings = []
    for i in range(nb_tilings):
        tiling = []
        # for each dimension
        for j, feature_range in enumerate(ranges):
            R = np.ptp(feature_range)
            B = bins[i, j]
            O = R / (nb_tilings * (B - 1) + 1)
            W = R + (nb_tilings - 1) * O

            feature_offset = i * O

            feature_tiling = np.linspace(feature_range[0], feature_range[0] + W, B + 1) - feature_offset
            EPS = 1e-6  # needed because tiling with value of feature_range[1] was not removed
            feature_tiling = feature_tiling[feature_tiling > feature_range[0] + EPS]
            feature_tiling = feature_tiling[feature_tiling < feature_range[1] - EPS]

            tiling.append(feature_tiling)
        tilings.append(tiling)
    return np.array(tilings)


def create_tilings_asymmetrical(ranges, nb_tilings, bins):
    """
    Like create_tiling() - For now only works on 2D features - Attempt at asymmetrical offsetting
    """
    assert nb_tilings == len(bins), "nb_tilings and bins info does not match"
    displacement_vector = (1, 3)
    unit_offset = np.ptp(ranges) / (bins[0, 0] + sum(displacement_vector) + 1)
    tilings = []
    for i in range(nb_tilings):
        tiling = []
        # for each dimension
        for j, feature_range in enumerate(ranges):
            R = np.ptp(feature_range)
            B = bins[i, j]
            O = R / (nb_tilings * (B - 1) + 1)
            W = R + (nb_tilings - 1) * O

            feature_offset = displacement_vector[j] * i * unit_offset

            feature_tiling = np.linspace(feature_range[0], feature_range[0] + W, B + 1) - feature_offset
            EPS = 1e-6  # needed because tiling with value of feature_range[1] was not removed
            feature_tiling = feature_tiling[feature_tiling > feature_range[0] + EPS]
            feature_tiling = feature_tiling[feature_tiling < feature_range[1] - EPS]

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


def get_tilings_from_env(env, nb_tilings, nb_bins, range_scale=1., asymmetrical=False):
    n_obs = env.obs_dimension

    ranges = [[lo, hi] for lo, hi in zip(env.observation_space.low,
                                         env.observation_space.high)]
    ranges = np.multiply(ranges, range_scale)

    bins = np.tile(nb_bins, (nb_tilings, n_obs))

    # available_offsets = np.linspace(0, range_size / nb_bins, nb_tilings + 1)[:-1]  # symmetrical tiling offsets
    # offsets = np.repeat(available_offsets, n_obs).reshape(-1, n_obs)

    if asymmetrical:
        tilings = create_tilings_asymmetrical(ranges, nb_tilings, bins)  # , offsets)
    else:
        tilings = create_tilings(ranges, nb_tilings, bins)  # , offsets)

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
        self.q_tables = np.array([np.tile(-1., (state_size + (len(self.actions),)))
                                  for state_size in state_sizes])

    # Initialize to random Q-values
    # self.q_tables = np.array([-np.random.rand(*(state_size + (len(self.actions),)))
    #                  for state_size in state_sizes])

    def value(self, state, action):
        codings = get_tile_coding(state, self.tilings)
        action_idx = self.actions.index(list(action))
        value = 0
        for coding, q_table in zip(codings, self.q_tables):
            value += q_table[tuple(coding) + (action_idx,)]
        return value / self.nb_tilings

    def update(self, state, action, target):
        codings = get_tile_coding(state, self.tilings)
        action_idx = self.actions.index(list(action))
        cur_val = self.value(state, action)
        # for i, coding in enumerate(codings):
        for i, (coding, q_table) in enumerate(zip(codings, self.q_tables)):
            # q_index = (i,)+tuple(coding)+(action_idx,)
            q_index = (i,) + tuple(coding) + (action_idx,)
            old_tiling_q_val = self.q_tables[q_index]
            # error = target - cur_val
            error = target - old_tiling_q_val
            delta = self.lr * error
            self.q_tables[q_index] = old_tiling_q_val + delta

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


'''
Tried this out but in the end we cannot value the feature individually. We need to combine them. It's obvious but a
mistake I had to make.
'''


class QValueFunction2:
    def __init__(self, tilings, actions, lr=1e2):
        self.tilings = tilings
        if type(actions) == np.ndarray:
            actions = actions.tolist()
        self.actions = actions
        self.lr = lr

        self.nb_tilings, nb_features, nb_bins = tilings.shape
        nb_bins += 1
        nb_actions = len(actions)

        self.q_tables = -np.random.rand(self.nb_tilings, nb_features, nb_bins, nb_actions)

    def get_coding_indices(self, state, action):
        action_idx = self.actions.index(list(action))
        codings = get_tile_coding(state, self.tilings)
        indices = []
        for i, coding in enumerate(codings):
            for j, feature_code in enumerate(coding):
                indices.append((i, j, feature_code, action_idx))
        return indices

    def value(self, state, action):
        codings = self.get_coding_indices(state, action)
        value = 0
        for coding in codings:
            value += self.q_tables[coding]
        return value / self.nb_tilings

    def update(self, state, action, target):
        codings = self.get_coding_indices(state, action)
        for coding in codings:
            delta = target - self.q_tables[coding]
            self.q_tables[coding] += self.lr * delta

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

    @classmethod
    def load(cls, path):
        if '.npz' not in path:
            path = os.path.join(path, 'qvf.npz')
        npz_archive = np.load(path)
        qvf = cls(npz_archive['tilings'],
                  npz_archive['actions'],
                  float(npz_archive['lr']))
        qvf.q_tables = npz_archive['q_tables']
        return qvf

    def greedy_action(self, state):
        q_vals = [self.value(state, a_) for a_ in self.actions]
        return self.actions[np.argmax(q_vals)]
