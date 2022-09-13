import numpy as np
import matplotlib.pyplot as plt

qvals = [1,1,4.9,5,4.2]
tau = 1.0

def boltz():
    qexp = np.exp(np.array(qvals)/tau)
    qexpsum = np.sum(qexp)
    probas = np.cumsum(qexp/qexpsum)
    return np.searchsorted(probas, np.random.rand())


bin_edges = np.arange(len(qvals)+1)-0.5
fig, ax = plt.subplots()
samples = []
# taus = np.linspace(1, 1e-2, 10)
taus = [1.0, 0.5, 0.1, 0.05, 0.02, 0.01]
for i, tau in enumerate(taus):
    samples.append([boltz() for _ in range(1000)])
plt.hist(samples, bin_edges, histtype='barstacked', label=taus, density=True)
plt.legend()
plt.show()