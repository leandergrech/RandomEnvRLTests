import os
import numpy as np
import pickle as pkl


class FeatureExtractor:
    def __init__(self, n_obs):
        self.n_obs = n_obs
        self.n_features = None

    def get_features(self, state):
        return state

    def get_n_features(self):
        if self.n_features is None:
            self.n_features = len(self.get_features(np.zeros(self.n_obs)))
        return self.n_features

    def __call__(self, state, **kwargs):
        return self.get_features(state)


class QValueFunctionLinear:
    def __init__(self, feature_fn: FeatureExtractor, n_actions: int):
        """
        param tilings: Tiling instance
        param actions: List of all possible discrete actions
        param lr: Generator that yields learning rate
        """
        self.feature_fn = feature_fn
        self.n_features = feature_fn.get_n_features()
        self.n_actions = n_actions

        self.w = np.zeros(self.n_features + n_actions)

        self.nb_updates = 0

    def get_full_input(self, state, action):
        state_feature = self.feature_fn.get_features(state)
        state_action = np.concatenate([state_feature, action])
        return state_action

    def value(self, state, action):
        return self.w.dot(self.get_full_input(state, action))

    def update(self, state, action, target, lr):
        self.nb_updates += 1

        state_action = self.get_full_input(state, action)
        error = (target - self.value(state, action)) * state_action

        self.w += lr * error

        return np.mean(error)

    def greedy_action(self, state, verbose=False):
        if verbose: print(self.nb_updates)
        vals = [self.value(state, a_) for a_ in range(self.n_actions)]
        action_idx = argmax(vals)
        return action_idx

    def count(self):
        return self.tilings.count()


    def save(self, save_path):
        with open(save_path, 'wb') as  f:
            pkl.dump(dict(
                q_table=self.q_table[:self.count()],
                tilings=self.tilings,
                n_discrete_actions=self.n_actions), f)

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
