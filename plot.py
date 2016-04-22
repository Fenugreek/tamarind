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
    
    
def multiple(columns, data, vertical_lines=None, axis=None, xlim=None, ylim=None):
    """
    Plot multiple graphs in columns no. of columns.
    vertical_lines is a list of x-values that gets passed onto lines().
    """
    rows = numpy.ceil(len(data) / columns)
    for count, records in enumerate(data):
        pyplot.subplot(rows, columns, count+1)
        pyplot.plot(records)
        if axis is not None: pyplot.axis(axis)
        if xlim is not None: pyplot.xlim(xlim)
        if ylim is not None: pyplot.ylim(ylim)
        if plotline_values is not None: lines(vertical_lines)


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
