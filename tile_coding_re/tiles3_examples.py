from numpy import min as npmin, max as npmax, linspace, pi, sign
from numpy.random import randn, choice, rand
import matplotlib.pyplot as plt
from tiles3 import tiles, IHT, tilesclip, tilesasymmetric
from heatmap_utils import *
from training_utils import lr
from math import cos, sin, sqrt

def actual_func(x, y):
    # return sign(x*y)
    # return sin(4*x + pi/3.) * cos(4*y)
    x *= 2.
    y *= 2.
    res = (1 + (x**2 + y**2))/((x**2 + y**2) + 2)
    return (1-res)*10

maxSize = 2**15
iht = IHT(maxSize)
iht2 = IHT(maxSize)
weights = [0.0] * maxSize
weights2 = [0.0] * maxSize

numTilingsCoarse = 8
# numTilingsFine = 64

nb_binsX_coarse = 4
nb_binsY_coarse = 4
# nb_binsX_fine = 2
# nb_binsY_fine = 2

dim_size = 50
test_scale = 1.5
bound = 1.
xr = linspace(-bound, bound, dim_size).tolist()
yr = linspace(-bound, bound, dim_size).tolist()
xrtest = linspace(-test_scale * bound, test_scale * bound, int(dim_size * test_scale))
yrtest = linspace(-test_scale * bound, test_scale * bound, int(dim_size * test_scale))

scaleFactorXCoarse = float(nb_binsX_coarse) / (max(xr) - min(xr))
scaleFactorYCoarse = float(nb_binsY_coarse) / (max(yr) - min(yr))
# scaleFactorXFine = float(nb_binsX_fine) / (max(xr) - min(xr))
# scaleFactorYFine = float(nb_binsY_fine) / (max(yr) - min(yr))



# stepSize = 0.2/numTilings

lr_gen = lr(0.1, 10000)
lr = lambda: next(lr_gen)
    
inf = float('inf')
clip = 0.3
clipXCoarse = clip * scaleFactorXCoarse
clipYCoarse = clip * scaleFactorYCoarse
# clipXFine = clip * scaleFactorXFine
# clipYFine = clip * scaleFactorYFine
def mytiles(x, y):
    XCoarse = x*scaleFactorXCoarse
    YCoarse = y*scaleFactorYCoarse
    
    # XFine = x*scaleFactorXFine
    # YFine = y*scaleFactorYFine
    return tiles(iht2, numTilingsCoarse, [XCoarse, YCoarse])
    
def mytiles2(x, y):
    XCoarse = x*scaleFactorXCoarse
    YCoarse = y*scaleFactorYCoarse
    return tilesasymmetric(iht, numTilingsCoarse, [XCoarse, YCoarse], (1, 3))
    
    # return tilesclip(iht, numTilingsCoarse, [XCoarse, YCoarse], 
    #                  ((-inf, -clipXCoarse), (-inf, -clipYCoarse)), [0]) + \
    #     tilesclip(iht, numTilingsCoarse, [XCoarse, YCoarse], 
    #               ((clipXCoarse, inf), (clipYCoarse, inf)), [1]) + \
    #     tilesclip(iht, numTilingsFine, [XFine, YFine], 
    #               ((-clipXFine, clipXFine), (-clipYFine, clipYFine)), [2])# + \
    # return tilesclip(iht, numTilingsCoarse, [XFine], 
    #               [[-inf, -clipXCoarse]], [3]) + \
    #     tilesclip(iht, numTilingsCoarse, [XFine], 
    #               [[clipXCoarse, inf]], [4]) + \
    #     tilesclip(iht, numTilingsCoarse, [YFine], 
    #               [[-inf, -clipYCoarse]], [5]) + \
    #     tilesclip(iht, numTilingsCoarse, [YFine], 
    #               [[clipYCoarse, inf]], [6])
    

def test(x, y, w, t):
    tiles = t(x, y)
    estimate = 0
    for tile in tiles:
        estimate += w[tile]
    return estimate

def learn(x, y, z, w, t):
    tiles = t(x, y)
    estimate = test(x, y, w, t)
    error = z - estimate
    for tile in tiles:
        w[tile] += lr() * error   # learn weights
        
# Function to learn
zr = [[0. for _ in range(len(xr))] for _ in range(len(yr))]
for i, y in enumerate(yr):
    for j, x in enumerate(xr):
        zr[i][j] = actual_func(x, y)

# Placeholder for estimate
zest = [[0. for _ in range(len(xrtest))] for _ in range(len(yrtest))]
zest2 = [[0. for _ in range(len(xrtest))] for _ in range(len(yrtest))]

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,7.5))
fig.suptitle(f'{numTilingsCoarse} Tilings - {nb_binsX_coarse} Bins', size=20)
im1 = make_heatmap(ax1, zr, xr, yr)
ax1.set_title('Actual')
im2 = make_heatmap(ax2, zest, xrtest, yrtest)
im3 = make_heatmap(ax3, zest2, xrtest, yrtest)
fig.tight_layout()

# fig.tight_layout()
# plt.show(block=False)

error = []

nb_timesteps = 5000
eval_every = 50
for t in range(nb_timesteps):
    # [i, j] = choice(dim_size, 2)
    # x = xr[i]
    # y = yr[j]
    # z = zr[i][j]
    x, y = randn(2)
    z = actual_func(x, y)
    
    # error.append(z - test(x, y))
    learn(x, y, z, weights, mytiles)
    learn(x, y, z, weights2, mytiles2)
    
    if (t + 1) % eval_every == 0:
        for i, x in enumerate(xrtest):
            for j, y in enumerate(yrtest):
                zest[i][j] = test(x, y, weights, mytiles)
                zest2[i][j] = test(x, y, weights2, mytiles2)
        update_heatmap(im2, zest, f'mytiles\nTraining step {t+1:4d}\nIHT count {iht.count():6d}')
        update_heatmap(im3, zest2, f'mytiles2\nTraining step {t+1:4d}\nIHT count {iht2.count():6d}')
        # ax2.set_title()
        plt.pause(0.05)
plt.ioff()
print(iht.count())
# plt.figure()
# from pandas import Series
# plt.plot(Series(error).rolling(nb_timesteps//10).mean().tolist())
# plt.axhline(0.0, c='k')


plt.show()



