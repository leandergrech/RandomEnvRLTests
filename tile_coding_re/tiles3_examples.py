from numpy import min as npmin, max as npmax
import matplotlib.pyplot as plt
from tiles3 import tiles, IHT
from heatmap_utils import *

maxSize = 2**13
iht = IHT(maxSize)
weights = [0.0] * maxSize
numTilings = 8
stepSize = 0.1/numTilings

dim_size = 20
xr = [(item + 1)/dim_size for item in range(dim_size)]  # [0.1,...,1.0]
yr = [(item + 1)/dim_size for item in range(dim_size)]  # [0.1,...,1.0]

nb_binsX = 2
nb_binsY = 4

def mytiles(x, y):
    # scaleFactor = 10.0 / (3-1)  # x and y run each from 1 to 3, and you would like to use 10 x 10 grid tilings
    scaleFactorX = float(nb_binsX) / (max(xr) - min(xr))
    scaleFactorY = float(nb_binsY) / (max(yr) - min(yr))

    return tiles(iht, numTilings,
                 list((x*scaleFactorX, y*scaleFactorY)))

def test(x, y):
    tiles = mytiles(x, y)
    estimate = 0
    for tile in tiles:
        estimate += weights[tile]
    return estimate

def learn(x, y, z):
    tiles = mytiles(x, y)
    estimate = test(x, y)
    error = z - estimate
    for tile in tiles:
        weights[tile] += stepSize * error   # learn weights
        
# Function to learn
from math import sin
zr = [[0. for _ in range(len(xr))] for _ in range(len(yr))]
for i, y in enumerate(yr):
    for j, x in enumerate(xr):
        zr[i][j] = x * sin(y*4)

# Placeholder for estimate
zest = [[0. for _ in range(len(xr))] for _ in range(len(yr))]

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2)
im1 = make_heatmap(ax1, zr, xr, yr)
ax1.set_title('Actual')
im2 = make_heatmap(ax2, zest, xr, yr)

# fig.tight_layout()
# plt.show(block=False)

error = []

from numpy.random import choice
nb_timesteps = 400
eval_every = 1
for t in range(nb_timesteps):
    [i, j] = choice(dim_size, 2)
    x = xr[i]
    y = yr[j]
    z = zr[i][j]
    
    learn(x, y, z)
    error.append(z - test(x, y))
    
    if (t + 1) % eval_every == 0:
        for i, x in enumerate(xr):
            for j, y in enumerate(yr):
                zest[i][j] = test(x, y)
        update_heatmap(im2, zest, f'Training step {t}')
        # ax2.set_title()
        plt.pause(0.1)

print(iht.count())
plt.plot(error)
plt.axhline(0.0, c='k')





plt.show()



