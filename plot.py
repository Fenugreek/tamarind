"""
Macro-like utility functions for plotting with matplotlib.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import numpy
from matplotlib import pyplot
import stats
        
def overlay(x_values, y_values, *args, **kwargs):
    """
    Plot unmasked values as a solid line, and masked values as a dashed line.
    *args and **args are passed onto pyplot.plot().
    """

    if 'overlay' not in kwargs:
        pyplot.plot(x_values, y_values, *args, **kwargs)
        return

    masked = numpy.ma.empty(numpy.shape(x_values), dtype=bool)
    masked.mask = ~kwargs['overlay']
    del kwargs['overlay']
    slices = numpy.ma.notmasked_contiguous(masked)
    if slices is None: return
    
    s0 = slices[0]
    pyplot.plot(x_values[max(s0.start - 1, 0) : s0.stop],
                y_values[max(s0.start - 1, 0) : s0.stop],
                *args, **kwargs)
    if 'label' in kwargs: kwargs['label'] = '_nolegend_'
    for s in slices[1:]:
        pyplot.plot(x_values[s.start - 1 : s.stop],
                    y_values[s.start - 1 : s.stop],
                    *args, **kwargs)
    
    
def multi(data, ylim=None, columns=1, xlim=None, sliced=None,
          verticals=None, horizontals=None, line_args=[]):
    """
    Plot multiple graphs in columns no. of columns.
    vertical_lines is a list of x-values that gets passed onto lines().

    If data is a tuple, the function recursively calls itself for
    each element in data.
    """
    if type(data) == tuple:
        for d in data[:-1]:
            multi(d, columns=columns, sliced=sliced, line_args=line_args)
        multi(data[-1], ylim=ylim, columns=columns, sliced=sliced,
              xlim=xlim, line_args=line_args,
              verticals=verticals, horizontals=horizontals)
        return

    if sliced:
        if np.isscalar(sliced): sliced = [sliced]
        data = data[slice(*sliced)]
    rows = numpy.ceil(len(data) / columns)
    for count, records in enumerate(data):
        pyplot.subplot(rows, columns, count+1)
        records = records.squeeze()
        pyplot.plot(records, *line_args)
        if ylim is not None: pyplot.ylim(ylim)
        if xlim is not False:
            if xlim is None: pyplot.xlim([0, len(records)])
            else: pyplot.xlim(xlim)
            
        if verticals is not None: lines(verticals)
        if horizontals is not None: lines(horizontals, axis=0)


def twin(data, colors=['Blue', 'Red'], ylabels=['y1', 'y2'], xlabel='x',
         func=['plot', 'plot'], ylims=[None, None], kwargs=[{}, {}]):
    """
    Make two plots on the same figure, with different y axes.
    """
    fig, ax = pyplot.subplots()

    # Twin the x-axis to make independent y-axes.
    axes = [ax, ax.twinx()]
    for i in range(2):
        ax, color, args = axes[i], colors[i], data[i]
        
        plot_func = getattr(ax, func[i])
        if type(args) == tuple: plot_func(*args, color=color, **kwargs[i])
        else: plot_func(args, color=color, **kwargs[i])

        if ylims[i] is not None: ax.set_ylim(*ylims[i])

        ax.set_ylabel(ylabels[i], color=color)
        ax.tick_params(axis='y', colors=color)
        
    axes[0].set_xlabel(xlabel)
    pyplot.show()
    
    return axes


def lines(values, axis=1, line_args='k'):
    """
    Plot straight lines at each axis value, in black.
    """

    if numpy.isscalar(values): values = [values]
    if numpy.isscalar(line_args): line_args = [line_args]
    
    ranges = pyplot.axis()
    for val in values:
        if axis: pyplot.plot([val, val], ranges[2:], *line_args)
        else: pyplot.plot(ranges[:2], [val, val], *line_args)
    pyplot.axis(ranges)


def summary(values, axis=0, statistic='mean', x_values=None, weights=None,
            error_band=1, error_statistic='std_err',
            line_args=[], error_args=['g']):
    """
    Given 2D values, plot summary statistic along <axis> dimension.

    statistic:
    Plot this statistic of the values in given <axis>.

    weights:
    Use this array as weights for <values> when computing
    statistic and error_statistic.
    
    x_values:
    Plot summary statistic against this. Default is just the index of <values>.

    error_statistic, error_band:
    Plot also this statistic, as +/- <error_band> times <error_statistic>.

    line_args, error_args:
    Pass these to plot() call.
    """
    
    cls = 'Sparse'
    for stat in (statistic, error_statistic):
        if not stat: continue
        if stat in ('median', 'mad') or stat[-2:] == 'le': cls = 'Full'

    s = getattr(stats, cls).stats(values, axis=axis, label_all=None,
                                  weights=weights)
    y_values = s[statistic]
    if x_values is None: x_values = range(len(y_values))
    
    if line_args and numpy.isscalar(line_args): line_args = [line_args]
    pyplot.plot(x_values, y_values, *line_args)

    if not error_band: return
    if error_args and numpy.isscalar(error_args): error_args = [error_args]
    pyplot.plot(x_values,
                y_values + s[error_statistic] * error_band,
                *error_args)
    pyplot.plot(x_values,
                y_values - s[error_statistic] * error_band,
                *error_args)
