from collections.abc import Generator
from tile_coding_re.tiles3 import tiles, IHT


class Tilings:
    def __init__(self, nb_tilings, nb_bins, feature_ranges, size):
        # size = nb_tilings * nb_bins ** len(feature_ranges) * nb_ints
        self.size = size
        self.iht = IHT(size)

        self.nb_tilings = nb_tilings
        self.nb_bins = nb_bins
        self.feature_ranges = feature_ranges
        self.scale_factors = self.__get_scalings()

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
        return tiles(self.iht, self.nb_tilings, scaled_features, ints)


def argmax(arr):
    return max((x, i) for i, x in enumerate(arr))[1]


class QValueFunctionTiles3:
    def __init__(self, tilings: Tilings, actions: list, lr: Generator[float, None, None]):
        """
        param tilings: Tiling instance
        param actions: List of all possible discrete actions
        param lr: Generator that yields learning rate
        """
        self.tilings = tilings
        self.actions = actions
        self.lr = lambda: next(lr) / len(tilings)
        self.q_table = [-1.] * tilings.size

        self.nb_updates = 0

    def get_action_value(self, action):
        return self.actions.index(action)

    def value(self, state, action):
        action_idx = self.get_action_value(action)
        codings = self.tilings.get_tiling_indices(features=state,
                                                  ints=[action_idx])
        estimate = 0.
        for coding in codings:
            estimate += self.q_table[coding]
        return estimate

    def update(self, state, action, target):
        self.nb_updates += 1

        action_idx = self.get_action_value(action)
        codings = self.tilings.get_tiling_indices(features=state,
                                                  ints=[action_idx])
        error = target - self.value(state, action)
        for coding in codings:
            self.q_table[coding] += self.lr() * error

    def greedy_action(self, state, verbose=False):
        if verbose: print(self.nb_updates)
        action_idx = argmax([self.value(state, a_) for a_ in self.actions])
        return self.actions[action_idx]
