import os
import numpy as np
import pickle as pkl

from utils.training_utils import argmax, QFuncBaseClass


class FeatureExtractor:
    def __init__(self, n_obs):
        self.n_obs = n_obs
        self._n_features = None

    def _get_feature(self, state):
        return state

    @property
    def n_features(self):
        if self._n_features is None:
            self._n_features = len(self._get_feature(np.zeros(self.n_obs)))
        return self._n_features

    def __call__(self, state, **kwargs):
        return self._get_feature(state)


class QValueFunctionLinear(QFuncBaseClass):
    def __init__(self, feature_fn: FeatureExtractor, actions: list):
        """
        param feature_fn: FeatureExtractor instance which converts continuous state to features
        param actions: List of all possible actions
        param lr: Generator that yields learning rate
        """
        super(QValueFunctionLinear, self).__init__()
        self.feature_fn = feature_fn
        self.n_features = feature_fn.n_features

        self.actions = actions
        self.n_discrete_actions = len(actions)
        self.n_act = len(actions[0])

        self.w = np.zeros(self.n_features + self.n_act)

        self.nb_updates = 0

    def get_full_input(self, state, action):
        state_feature = self.feature_fn(state)
        state_action = np.concatenate([state_feature, action])
        return state_action

    def value(self, state, action_idx):
        return self.w.dot(self.get_full_input(state, self.actions[action_idx]))

    def update(self, state, action_idx, target, lr):
        self.nb_updates += 1

        state_action = self.get_full_input(state, self.actions[action_idx])
        error = (target - self.value(state, action_idx)) * state_action

        self.w += lr * error

        return np.mean(error)

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pkl.dump(dict(
                feature_fn=self.feature_fn,
                actions=self.actions,
                w=self.w,
                nb_updates=self.nb_updates
            ), f)

    @staticmethod
    def load(load_path):
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                d = pkl.load(f)
                feature_fn = d['feature_fn']
                actions = d['actions']
                self = QValueFunctionLinear(feature_fn, actions)
                self.w = d['w']
                self.nb_updates = d['nb_updates']
                return self
        else:
            raise FileNotFoundError(f'Path passed: {load_path}, does not exist.')
