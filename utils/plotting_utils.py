import matplotlib.pyplot as plt
from numpy import nanmin as npmin, nanmax as npmax, array

from matplotlib.ticker import MultipleLocator


def grid_on(ax, axis='y', major_loc=None, minor_loc=None, major_grid=True, minor_grid=True):
    if axis == 'y':
        axis_ = ax.yaxis
    else:
        axis_ = ax.xaxis

    if major_loc is not None:
        axis_.set_major_locator(MultipleLocator(major_loc))
    if minor_loc is not None:
        axis_.set_minor_locator(MultipleLocator(minor_loc))
    ax.minorticks_on()
    if major_grid:
        ax.grid(which='major', c='gray', axis=axis)
    if minor_grid:
        ax.grid(which='minor', c='gray', ls='--', alpha=0.5, axis=axis)


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
