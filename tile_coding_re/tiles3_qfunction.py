import os
import pickle as pkl
from tile_coding_re.tiles3 import IHT, tiles, tilesclip


class Tilings:
    def __init__(self, nb_tilings, nb_bins, feature_ranges, max_tiles):
        # max_tiles = nb_tilings * nb_bins ** len(feature_ranges) * nb_ints
        self.max_tiles = max_tiles
        self.iht = IHT(max_tiles)

        self.nb_tilings = nb_tilings
        self.nb_bins = nb_bins
        self.feature_ranges = feature_ranges
        self.scale_factors = self.__get_scalings()

    def tiles(self, features, ints=[]):
        return tiles(self.iht, self.nb_tilings, features, ints)

    def count(self):
        return self.iht.count()

    def __len__(self):
        return self.nb_tilings

    def __get_scalings(self):
        scale_factors = []
        for fr in self.feature_ranges:
            scale_factors.append(float(self.nb_bins) / (fr[1] - fr[0]))
        return scale_factors

    def get_tiling_indices(self, features, ints=[]):
        scaled_features = []
        for f, scale in zip(features, self.scale_factors):
            scaled_features.append(f * scale)
        return self.tiles(scaled_features, ints)


def argmax(arr):
    return max((x, i) for i, x in enumerate(arr))[1]


class QValueFunctionTiles3:
    def __init__(self, tilings: Tilings, n_discrete_actions):#, lr: Generator[float, None, None]):
        """
        param tilings: Tiling instance
        param actions: List of all possible discrete actions
        param lr: Generator that yields learning rate
        """
        self.tilings = tilings
        self.n_discrete_actions = n_discrete_actions
        # self.lr = lambda: next(lr) / len(tilings)
        init_q_val = 2.0
        self.q_table = [init_q_val for _ in range(tilings.max_tiles)]
        # self.q_table = [-rand() for _ in range(tilings.max_tiles)]

        self.nb_updates = 0

    def value(self, state, action_idx=0):
        codings = self.tilings.get_tiling_indices(features=state,
                                                  ints=[action_idx])
        estimate = 0.
        for coding in codings:
            estimate += self.q_table[coding]
        return estimate / len(self.tilings)

    def update(self, state, action_idx, target, lr):
        self.nb_updates += 1

        codings = self.tilings.get_tiling_indices(features=state,
                                                  ints=[action_idx])
        error = target - self.value(state, action_idx)

        # alpha = self.lr()/len(self.tilings)
        alpha = lr / self.tilings.nb_tilings

        for coding in codings:
            self.q_table[coding] += alpha * error
        return error

    def set(self, state, action_idx, val):
        codings = self.tilings.get_tiling_indices(features=state, ints=[action_idx])
        for coding in codings:
            self.q_table[coding] = val

    def greedy_action(self, state, verbose=False):
        if verbose: print(self.nb_updates)
        vals = [self.value(state, a_) for a_ in range(self.n_discrete_actions)]
        action_idx = argmax(vals)
        return action_idx

    def count(self):
        return self.tilings.count()


    def save(self, save_path):
        with open(save_path, 'wb') as  f:
            pkl.dump(dict(
                q_table=self.q_table[:self.count()],
                tilings=self.tilings,
                n_discrete_actions=self.n_discrete_actions))

    @staticmethod
    def load(load_path):
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                d = pkl.load(f)
                tilings = d['tilings']
                n_discrete_actions = d['n_discrete_actions']
                self = QValueFunctionTiles3(tilings, n_discrete_actions)
                self.q_table[:tilings.count()] = d['q_table']
                return self

        else:
            raise FileNotFoundError(f'Path passed: {load_path}, does not exist.')
