import numpy as np
from tile_coding_re.tiles3_qfunction import QValueFunctionTiles3


def eps_greedy(state: np.ndarray, qfunc: QValueFunctionTiles3, epsilon: float) -> int:
    nb_actions = qfunc.n_discrete_actions
    if np.random.rand() < epsilon:
        return np.random.choice(nb_actions)
    else:
        return qfunc.greedy_action(state)


def boltzmann(state: np.ndarray, qfunc: QValueFunctionTiles3, tau: float) -> int:
    nb_actions = qfunc.n_discrete_actions
    qvals_exp = np.exp([qfunc.value(state, a_) / tau for a_ in range(nb_actions)])
    qvals_exp_sum = np.sum(qvals_exp)

    cum_probas = np.cumsum(qvals_exp / qvals_exp_sum)
    return np.searchsorted(cum_probas, np.random.rand())