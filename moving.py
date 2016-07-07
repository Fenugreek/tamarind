"""
Compute and plot statistics such as the mean in a rolling window of data.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import division

import sys
import numpy
from matplotlib import pyplot
import stats, arrays


def _get_size_offset(data, size, overlap, fit_edges=True):
    """
    Utility function for determining rolling window parameters, for given data size.

    size: nominal size of the rolling window. If a fraction, is multiplied by
          data size.
    overlap: How much consecutive windows overlap, expressed as an integer number
             of windows a data point appears in.
    fit_edges: If True (default), tweak size parameter so that last datapoint
               forms the last datapoint of the last window.

    Returns size and offset parameters of rolling window.
    """
    if size <= 1:
        if isinstance(data, numpy.ma.masked_array):
            size = int(round(size * data.count()))
        else: size = int(round(size * len(data)))
        
    offset = max(int(round(size / overlap)), 1)
    if fit_edges:
        while ((len(data) - size) % offset): size += 1

    return size, offset

                     
def rolling_stats(data, size=0.05, overlap=5, weights=None, cls='Full',
                  fit_edges=True, offset=None):
    """
    Return an array of stats, as a stats.Datab object, for given data. Each element
    corresponds to statistics on a subsample of the data, as selected by the
    moving window with parameters size and offset. Offset parameter is determined
    via overlap parameter if not explicitly given -- see _get_size_offset().

    Data can be 2D, in which case stats are multivariate.
    """
    
    if offset is None:
        size, offset = _get_size_offset(data, size, overlap, fit_edges=fit_edges)
    if weights is not None:
        weights = arrays.rolling_window(weights, size, offset)
    
    if numpy.ndim(data) == 1: 
        windows = arrays.rolling_window(data, size, offset)
        stats_obj = getattr(stats, cls)
        return stats_obj.stats(windows, weights=weights, axis=1, label_all=None)
    elif numpy.ndim(data) != 2:
        sys.exit('Data must be one or two dimensional.')

    # rolling_window works only on the last axis, so perform the necessary
    # axis manipulations.
    windows = arrays.rolling_window(data.T, size, offset)
    windows = windows.swapaxes(0,1).swapaxes(1,2)
    
    return stats.Multivariate.stats(windows, weights=weights, axis=0, label_all=None)


def nearest_stats(x, y, x_weights=None, y_weights=None, size=0.05, cls='Full',
                  sliced=None, select=None, overlay=None, overlap=5, fit_edges=True):
    """
    Sort x and y input arrays by values of x, and return rolling stats on the results.

    The rolling stats can be weighted, with separate weights for x and for y. If only
    x_weights are given, y_weights are set == x_weights.

    y can have one extra dimension than x, in which case the y rolling stats are
    Multivariate stats.

    size, overlap, fit_edges: rolling window parameters; see rolling_stats.
    sliced, select: optional array element selection parameters.
    """

    x, y, x_weights, y_weights = arrays.select((x, y, x_weights, y_weights),
                                                 sliced=sliced, select=select, overlay=overlay)

    indices = arrays.argsort(x)
    x = x[indices]
    y = y[indices]
          
    if y_weights is not None:
        if x_weights is None: sys.stderr.write('Warn: unusual options to nearest stats() -- y_weights present but x_weights absent')
        y_weights = y_weights[indices]
    if x_weights is not None:
        x_weights = x_weights[indices]
        if y_weights is None: y_weights = x_weights

    size, offset = _get_size_offset(x, size, overlap,
                                    fit_edges=fit_edges)

    return rolling_stats(x, size=size, offset=offset, weights=x_weights, cls=cls), \
           rolling_stats(y, size=size, offset=offset, weights=y_weights, cls=cls)


def plot(x, y, x_weights=None, y_weights=None, size=0.05,
         sliced=None, select=None, overlay=None,
         x_statistic='mean', y_statistic=None,
         error_band=None, error_statistic='std_err',
         overlap=5, fit_edges=True, line_args=[], error_args=['g']):
    """
    Produce a moving average plot of y against x.

    By default, +/1.0 standard error bands are additionally plotted. Specify
    some other value to error_band option as desired.
    
    The moving average can be weighted, with separate weights for x and for y. If only
    x_weights are given, y_weights are set == x_weights.

    Instead of a moving average, some other moving statistic can be plotted by
    setting [x|y]_statistic option (e.g. to 'median'). If x_statistic option is
    set to None, a rank transform of x is used for the x-axis.

    y can have one extra dimension than x, in which case the y rolling stats are
    Multivariate stats, and the plotted statistic defaults to sum(y0*y1)/sum(y1*y1).

    size, overlap, fit_edges: rolling window parameters; see rolling_stats.
    sliced, select: optional array element selection parameters.

    line_args, error_args:
    args to pass to plot() when plotting the main line and error bands respectively.
    """
    
    if numpy.ndim(y) > numpy.ndim(x):
        # multivariate response; plot sum(y0*y1)/sum(y1*y1) by default
        if y_statistic is None: y_statistic = 'coeff_0_1'
    else:
        if error_band is None: error_band = 1.0
        if y_statistic is None: y_statistic = 'mean'

    cls = 'Sparse'
    for statistic in (x_statistic, y_statistic):
        if not statistic: continue
        if statistic in ('median', 'mad') or statistic[-2:] == 'le': cls = 'Full'
        
    x_stats, y_stats = nearest_stats(x, y, x_weights=x_weights, y_weights=y_weights,
                                     size=size, sliced=sliced, select=select, overlay=overlay,
                                     overlap=overlap, fit_edges=fit_edges, cls=cls)
    if x_statistic: x_values = x_stats[x_statistic]
    else: x_values = numpy.arange(len(x_stats)) / len(x_stats)

    if line_args and numpy.isscalar(line_args): line_args = [line_args]
    pyplot.plot(x_values, y_stats[y_statistic], *line_args)

    if not error_band: return
    if error_args and numpy.isscalar(error_args): error_args = [error_args]
    pyplot.plot(x_values, y_stats[y_statistic] + y_stats[error_statistic] * error_band, *error_args)
    pyplot.plot(x_values, y_stats[y_statistic] - y_stats[error_statistic] * error_band, *error_args)

