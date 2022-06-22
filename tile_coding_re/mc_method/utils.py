import numpy as np


class TrajSimple:
    def __init__(self):
        self.data = None
        self.reset()

    def reset(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        if self.data:
            return True
        else:
            return False

    def __iter__(self):
        return self.data

    def add(self, datum):
        if isinstance(datum, np.ndarray):
            datum = datum.tolist()
        if isinstance(datum, list):
            self.data.append(datum.copy())
        else:
            self.data.append(datum)

    def pop(self, idx):
        return self.data.pop(idx)


class TrajBuffer:
    def __init__(self):
        self.o_ = None
        self.a_ = None
        self.r_ = None
        self.g_ = None
        self.reset()

    def reset(self):
        self.o_ = []
        self.a_ = []
        self.r_ = []
        self.g_ = []

    def __len__(self):
        return len(self.o_)

    def __bool__(self):
        if self.o_:
            return True
        else:
            return False

    def add(self, state, action, reward):
        self.o_.append(state.copy())
        self.a_.append(action.copy())
        self.r_.append(reward)

    def get_returns(self):
        return np.flip(np.cumsum(np.flip(self.r_))).tolist()

    def pop_target_tuple(self):
        if not self.g_:
            self.g_ = self.get_returns()

        if not self.o_:
            raise ValueError('No items in buffer')

        o = self.o_.pop(0)
        a = self.a_.pop(0)
        r = self.r_.pop(0)
        g = self.g_.pop(0)

        return o, a, g

    @property
    def o(self):
        return self.o_

    @property
    def a(self):
        return self.a_

    @property
    def r(self):
        return self.r_

    @property
    def g(self):
        return self.g_