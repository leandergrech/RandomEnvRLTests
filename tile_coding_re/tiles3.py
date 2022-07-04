"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed.
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)),
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile
coordinates are to be returned without being converted to indices).
"""

basehash = hash


class IHT:
	"Structure to handle collisions"

	def __init__(self, sizeval):
		self.size = sizeval
		self.overfullCount = 0
		self.dictionary = {}

	def __str__(self):
		"Prepares a string for printing whenever this object is printed"
		return "Collision table:" + \
			   " size:" + str(self.size) + \
			   " overfullCount:" + str(self.overfullCount) + \
			   " dictionary:" + str(len(self.dictionary)) + " items"

	def count(self):
		return len(self.dictionary)

	def fullp(self):
		return len(self.dictionary) >= self.size

	def getindex(self, obj, readonly=False):
		d = self.dictionary
		if obj in d:
			return d[obj]
		elif readonly:
			return None
		size = self.size
		count = self.count()
		if count >= size:
			# raise Exception('IHT full - no space for more')
			if self.overfullCount == 0: print(f'IHT full, starting to allow collisions: size={size}')
			self.overfullCount += 1
			return basehash(obj) % self.size
		else:
			d[obj] = count
			return count


def hashcoords(coordinates, m, readonly=False):
	if type(m) == IHT: return m.getindex(tuple(coordinates), readonly)
	if type(m) == int: return basehash(tuple(coordinates)) % m
	if m == None: return coordinates


from math import floor, log
from itertools import zip_longest

import numpy as np


def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
	"""returns num-tilings tile indices corresponding to the floats and ints"""
	qfloats = [floor(f * numtilings) for f in floats]
	Tiles = []
	for tiling in range(numtilings):
		tilingX2 = tiling * 2
		coords = [tiling]
		b = tiling
		for q in qfloats:
			coord = (q + b) // numtilings
			coords.append(coord)
			b += tilingX2
		coords.extend(ints)
		Tiles.append(hashcoords(coords, ihtORsize, readonly))
	return Tiles


def tilesasymmetric(ihtORsize, numtilings, floats, displacement, ints=[], readonly=False):
	"""returns num-tilings tile indices corresponding to the floats and ints, displacement vector for asymmetric offsetting"""
	qfloats = [floor((f) * numtilings) for f, d in zip(floats, displacement)]
	Tiles = []

	coords = [[0. for _ in range(1 + len(floats) + len(ints))] for _ in range(numtilings)]
	for i, q in enumerate(qfloats):
		for tilingactual in range(numtilings):
			d = displacement[i]
			tiling = d * tilingactual

			coords[tilingactual][0] = tiling

			coords[tilingactual][i + 1] = (q + tiling * (1 + i * 2)) // numtilings
			if len(ints) > 0:
				coords[tilingactual][-len(ints):] = ints

	for coord in coords:
		Tiles.append(hashcoords(coord, ihtORsize, readonly))
	return Tiles


def tilesclip(ihtORsize, numtilings, floats, clipranges, ints=[], readonly=False):
	"""returns num-tilings tile indices corresponding to the floats and ints, clipping floats"""
	cfloats = [f if f > cr[0] else cr[0] for f, cr in zip(floats, clipranges)]
	cfloats = [f if f < cr[1] else cr[1] for f, cr in zip(cfloats, clipranges)]
	qfloats = [floor(f * numtilings) for f in cfloats]
	Tiles = []
	for tiling in range(numtilings):
		tilingX2 = tiling * 2
		coords = [tiling]
		b = tiling
		for q in qfloats:
			coords.append((q + b) // numtilings)
			b += tilingX2
		coords.extend(ints)
		Tiles.append(hashcoords(coords, ihtORsize, readonly))
	return Tiles


def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
	"""returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
	qfloats = [floor(f * numtilings) for f in floats]
	Tiles = []
	for tiling in range(numtilings):
		tilingX2 = tiling * 2
		coords = [tiling]
		b = tiling
		for q, width in zip_longest(qfloats, wrapwidths):
			c = (q + b % numtilings) // numtilings
			coords.append(c % width if width else c)
			b += tilingX2
		coords.extend(ints)
		Tiles.append(hashcoords(coords, ihtORsize, readonly))
	return Tiles


if __name__ == '__main__':
	iht = IHT(2048)
	ind1 = tiles(iht, 8, [3.6, 7.21])
	print(ind1)
	ind2 = tiles(iht, 8, [3.7, 7.21])
	print(ind2)
	ind3 = tiles(iht, 8, [4, 7])
	print(ind3)
	ind4 = tiles(iht, 8, [-37.2, 7])
	print(ind4)
