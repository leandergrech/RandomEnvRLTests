def _get_feature(self, state, action):
    r = np.sqrt(np.sum(np.square(state)))
    return np.concatenate([state, [r]])