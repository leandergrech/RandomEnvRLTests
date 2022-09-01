import matplotlib.pyplot as plt
from numpy import nanmin as npmin, nanmax as npmax, array


def get_min_and_max(xrange):
    return npmin(xrange), npmax(xrange)


def make_heatmap(ax, z, x, y, title=None):
    xmin, xmax = get_min_and_max(x)
    ymin, ymax = get_min_and_max(y)

    im = ax.matshow(z, extent=(xmin, xmax, ymin, ymax), aspect='auto', origin='lower', cmap='jet', zorder=5)
    ax.axvline(0.0, c='w', zorder=15)
    ax.axhline(0.0, c='w', zorder=15)
    plt.colorbar(im, ax=ax, orientation='horizontal')
    if title is not None:
        im.axes.set_title(title)
    return im


def update_heatmap(im, z, title=None):
    im.set_data(z)
    im.set_clim(npmin(z), npmax(z))
    if title is not None:
        im.axes.set_title(title)
