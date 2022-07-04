import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

N_FEATURES = 6

# create dataset
X, y = make_regression(n_features=N_FEATURES)
X, X_test, y, y_test = train_test_split(X, y)

# model weights
params = {
    'w': jnp.zeros(N_FEATURES),
    'b': 0.
}


def forward(params, X):
    return jnp.dot(X, params['w']) + params['b']


def loss_fn(params, X, y):
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))  # mse


grad_fn = jax.grad(loss_fn)

ALPHA = 0.05


def update(params, grads):
    return jax.tree_util.tree_map(lambda p, g: p - ALPHA * g, params, grads)


# main training loop
for _ in range(50):
    loss = loss_fn(params, X_test, y_test)
    print(loss)

    grads = grad_fn(params, X, y)
    params = update(params, grads)

import matplotlib.pyplot as plt

fig, (axs) = plt.subplots(3, 2)
for i, (x, ax) in enumerate(zip(X.T, fig.axes)):
    y_pred = forward({'w': params['w'][i], 'b': params['b']}, x)
    ax.scatter(x, y)
    ax.plot(x, y_pred, c='r')

plt.show()
