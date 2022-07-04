import matplotlib.pyplot as plt
from numpy import nanmin as npmin, nanmax as npmax


def get_min_and_max(xrange):
	return npmin(xrange), npmax(xrange)


def make_heatmap(ax, z, x, y, title=None):
	xmin, xmax = get_min_and_max(x)
	ymin, ymax = get_min_and_max(y)

	im = ax.imshow(z, extent=(xmin, xmax, ymin, ymax), aspect='auto')
	ax.axvline(0.0, c='w')
	ax.axhline(0.0, c='w')
	plt.colorbar(im, ax=ax, orientation='horizontal')
	if title is not None:
		im.axes.set_title(title)
	return im


def update_heatmap(im, z, title=None):
	im.set_data(z)
	im.set_clim(npmin(z), npmax(z))
	if title is not None:
		im.axes.set_title(title)
