from numpy import min as npmin, max as npmax, linspace, pi, sign
from numpy.random import randn, choice, rand
import matplotlib.pyplot as plt
from tiles3 import tiles, IHT, tilesclip
from heatmap_utils import *
from math import cos, sin, sqrt

def actual_func(x, y):
    # return sign(x*y)
    # return sin(4*x + pi/3.) * cos(4*y)
    x *= 4.
    y *= 4.
    res = (1 + (x**2 + y**2))/((x**2 + y**2) + 2)
    return (1-res)*10

maxSize = 2**15
iht = IHT(maxSize)
weights = [0.0] * maxSize

numTilingsCoarse = 8
numTilingsFine = 64

nb_binsX_coarse = 4
nb_binsY_coarse = 4
nb_binsX_fine = 4
nb_binsY_fine = 4

dim_size = 70
# xr = [(item + 1)/dim_size for item in range(dim_size)]  # [0.1,...,1.0]
# yr = [(item + 1)/dim_size for item in range(dim_size)]  # [0.1,...,1.0]
xr = linspace(-1, 1, dim_size).tolist()
yr = linspace(-1, 1, dim_size).tolist()



# stepSize = 0.2/numTilings
def lr():
    calls = 0.
    halflife = 1000 # in number of calls
    init_lr = 5e-4
    while True:
        # yield (init_lr/(1. + (1./halflife)*calls))/numTilings
        # calls += 1.
        yield init_lr
lr_gen = lr()
lr = lambda: next(lr_gen)



# def mytiles(x, y):
#     index = 0
#     numTilings = 0
#     if sqrt(x**2+y**2) <= 0.2 and rand() < 0.8:
#         scaleFactorX = float(nb_binsX_coarse) / (max(xr) - min(xr))
#         scaleFactorY = float(nb_binsY_coarse) / (max(yr) - min(yr))
#         numTilings = numTilings_coarse
#     else:
#         index = 1
#         scaleFactorX = float(nb_binsX_fine) / (max(xr) - min(xr))
#         scaleFactorY = float(nb_binsY_fine) / (max(yr) - min(yr))
#         numTilings = numTilings_fine
    
#     scaledX = x*scaleFactorX
#     scaledY = y*scaleFactorY*3

#     return tiles(iht, numTilings, [scaledX, scaledY], [index])
#     # return tiles(iht, numTilings, [scaledX, scaledY], [0]) + \
#     #             tiles(iht, numTilings//2, [scaledX], [1]) + \
#     #             tiles(iht, numTilings//2, [scaledY], [2])
#     # return tiles(iht, numTilings, [scaledX], [1]) + \
#     #             tiles(iht, numTilings//2, [scaledY], [1])
    
inf = float('inf')
clip = 0.2
def mytiles(x, y):
    scaleFactorXCoarse = float(nb_binsX_coarse) / (max(xr) - min(xr))
    scaleFactorYCoarse = (float(nb_binsY_coarse) / (max(yr) - min(yr)))*2
    XCoarse = x*scaleFactorXCoarse
    YCoarse = y*scaleFactorYCoarse
    
    scaleFactorXFine = float(nb_binsX_fine) / (max(xr) - min(xr))
    scaleFactorYFine = (float(nb_binsY_fine) / (max(yr) - min(yr)))*2
    XFine = x*scaleFactorXFine
    YFine = y*scaleFactorYFine
    
    clipXCoarse = clip * scaleFactorXCoarse
    clipYCoarse = clip * scaleFactorYCoarse
    clipXFine = clip * scaleFactorXFine
    clipYFine = clip * scaleFactorYFine
    
    return tilesclip(iht, numTilingsCoarse, [XCoarse, YCoarse], 
                     ((-inf, -clipXCoarse), (-inf, -clipYCoarse)), [0]) + \
        tilesclip(iht, numTilingsCoarse, [XCoarse, YCoarse], 
                  ((clipXCoarse, inf), (clipYCoarse, inf)), [1]) + \
        tilesclip(iht, numTilingsFine, [XFine, YFine], 
                  ((-clipXFine, clipXFine), (-clipYFine, clipYFine)), [2])# + \
    # return tilesclip(iht, numTilingsCoarse, [XFine], 
    #               [[-inf, -clipXCoarse]], [3]) + \
    #     tilesclip(iht, numTilingsCoarse, [XFine], 
    #               [[clipXCoarse, inf]], [4]) + \
    #     tilesclip(iht, numTilingsCoarse, [YFine], 
    #               [[-inf, -clipYCoarse]], [5]) + \
    #     tilesclip(iht, numTilingsCoarse, [YFine], 
    #               [[clipYCoarse, inf]], [6])
    

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
        weights[tile] += lr() * error   # learn weights
        
# Function to learn
zr = [[0. for _ in range(len(xr))] for _ in range(len(yr))]
for i, y in enumerate(yr):
    for j, x in enumerate(xr):
        zr[i][j] = actual_func(x, y)

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

nb_timesteps = 1000
eval_every = 10
for t in range(nb_timesteps):
    # [i, j] = choice(dim_size, 2)
    # x = xr[i]
    # y = yr[j]
    # z = zr[i][j]
    x = randn()
    y = randn()
    z = actual_func(x, y)
    
    error.append(z - test(x, y))
    learn(x, y, z)
    
    if (t + 1) % eval_every == 0:
        for i, x in enumerate(xr):
            for j, y in enumerate(yr):
                zest[i][j] = test(x, y)
        update_heatmap(im2, zest, f'Training step {t+1:4d}\nIHT count {iht.count():6d}')
        # ax2.set_title()
        plt.pause(0.1)
plt.ioff()
print(iht.count())
plt.figure()
from pandas import Series
plt.plot(Series(error).rolling(nb_timesteps//10).mean().tolist())
# plt.gca().set_yscale('symlog')
plt.axhline(0.0, c='k')


plt.show()



