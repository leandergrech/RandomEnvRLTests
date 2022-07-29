import matplotlib.pyplot as plt
def plot_learning_rates():
    init = 1.0
    decay_rate = 0.95
    decay_every = 1000  # int(nb_eps/10)
    func = lambda i: init * decay_rate ** (i // decay_every)

    xrange = [item for item in range(100000)]
    yrange = [func(item) for item in xrange]

    plt.plot(xrange, yrange)
    plt.show()


if __name__ == '__main__':
    plot_learning_rates()
