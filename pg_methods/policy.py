import numpy as np
import torch as pt


class LinearPolicyDiscreteAction():
    def __init__(self, actions, feature_size):
        self.actions = actions
        self.n_actions = n_actions = len(actions)

        self.weights = np.zeros(n_actions * feature_size)

    def action_dist(self, state_features):
        """
        Returns a list of probabilities, whose sum is one, having size of self.actions
        :param state: Get action probabilities for this state
        """
        h = np.exp(self.weights.reshape(self.n_actions, -1).dot(state_features))
        sum_h = np.sum(h)

        return np.array([item/sum_h for item in h])


    def sample_action(self, state_features):
        probas = self.action_dist(state_features)

        action_idx = np.searchsorted(np.cumsum(probas), np.random.rand())

        return self.actions[action_idx]
