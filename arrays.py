"""
Some utilities for numpy array manipulation.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import numpy


def argsort(data, reverse=False, last_dim=False, mask=None, order=None):
    """
    Returns indices, a la numpy.where(), which would sort the data array.
    Unlike numpy, nans are handled, and two-dimensional arrays get
    two-dimensional results (not flattened).

    NaNs are excluded from the results -- number of indices returned
    would be the number of records which are not NaN.

    last_dim:
    sort each row of the last dimension separately.

    mask:
    mask with same dimensions as data with True where elements are to be ignored.
    """

    if not isinstance(data, numpy.ndarray): data = numpy.array(data)
    
    if last_dim:
        result = numpy.empty(data.shape[:-1], dtype=list)
        for ndindex in numpy.ndindex(data.shape[:-1]):
            result[ndindex] = list(argsort(data[ndindex],
                                           mask=mask[ndindex] if mask is not None else None,
                                           reverse=reverse, order=order))
        return result
    
    # We remove the NaNs first. Setting NaNs to 0 is not efficient
    # since that would increase sort time and memory usage, which is
    # a real issue for sparse arrays (e.g. TxN arrays of stock splits).
    #
    # Code below keeps track of where the NaNs were removed, so that
    # after the sort, returned indices are transformed back to refer to
    # the array with NaNs present.

    if order: data = data[order]
    if isinstance(data, numpy.ma.masked_array):
        if 'float' in str(data.dtype): nan_mask = numpy.isnan(data.data) | data.mask
        else: nan_mask = data.mask
    elif 'float' in str(data.dtype): nan_mask = numpy.isnan(data)
    else: nan_mask = numpy.zeros(numpy.shape(data), dtype=bool)
    if mask is not None:
        nan_mask |= mask
    sort_indices = data[~nan_mask].argsort()
    if reverse: sort_indices = sort_indices[::-1]

    if nan_mask.any():
        nonan_count = numpy.cumsum(~nan_mask)
        nan_stuff = numpy.zeros(nonan_count[-1], dtype=int) - 1
        for count, index in enumerate(nonan_count):
            if not index or nan_stuff[index - 1] >= 0 : continue
            nan_stuff[index - 1] = count
        for count, index in enumerate(sort_indices):                
            sort_indices[count] = nan_stuff[index]

    if numpy.ndim(data) != 2: return sort_indices

    # If array is 2D, convert 1D result to 2D.
    rows, columns = numpy.shape(data)
    indices = numpy.array([divmod(index, columns) for index in sort_indices])
    return tuple(indices.transpose())

    
def index_array(records, keys, field=None, arg=False):
    """
    Given a recarray and a list of key fields, return a multi-level dict with values
    of key fields as keys and corresponding recarray records as values.

    If field is not None, end values are record[field].
    If arg is True, end values are array indices.

    Note: end value of the dict is a scalar.
    """

    if arg and field is not None: raise SyntaxError('Can not satisfy both arg and field options')
    
    output = {}
    for count, record in enumerate(records):
        key_values = [record[i] for i in keys]
        layer = output
        for key in key_values[:-1]:
            if key not in layer: layer[key] = {}
            layer = layer[key]
        if arg: layer[key_values[-1]] = count
        elif field is not None: layer[key_values[-1]] = record[field]
        else: layer[key_values[-1]] = record

    return output


def rank(data, axis=None, reverse=False, mask=None):
    """
    Returns array of integers representing rank of elements in given array.

    mask:
    mask with same dimensions as data with True where elements are to be ignored.
    """

    if axis is None:
        # initialize ranks to -1, value to return for NaNs
        indices_rank = numpy.zeros(numpy.shape(data), dtype=int) - 1
        sort_indices = argsort(data, reverse=reverse, mask=mask)
        indices_rank[sort_indices] = numpy.array(range(numpy.shape(sort_indices)[-1]))
        return indices_rank

    if not isinstance(data, numpy.ndarray): data = numpy.array(data)    
    sort_indices = argsort(data.swapaxes(axis, -1),
                           reverse=reverse, last_dim=True, mask=mask)

    indices_rank = numpy.zeros(data.swapaxes(axis, -1).shape, dtype=int) - 1
    for ndindex, sort_list in numpy.ndenumerate(sort_indices):
        for rank, index in enumerate(sort_list):
            indices_rank[ndindex][index] = rank

    return indices_rank.swapaxes(-1, axis)


def nice_array(values, shape=None, logger=None, copy=False):
    """
    Utility function to convert input data to a nice ma.array with appropriate shape
    if necessary. Converts integer arrays to float arrays.
    """
    
    if isinstance(values, numpy.ma.masked_array):
        if values.dtype.kind == 'i': result = numpy.ma.asarray(values, dtype=float)
        elif copy: result = values.copy() 
        else: result = values
    elif values is None: return None
    else: result = numpy.ma.array(values, mask=numpy.isnan(values),
                                  fill_value=numpy.nan, copy=copy, dtype=float)

    if shape is not None and result.shape != shape:
        if result.size == numpy.prod(shape):
            result = result.reshape(shape)
        elif len(result) == shape[0]:
            if logger is not None: logger.debug('nice_array: Broadcasting 1D values for 2D values across columns.')
            result = result[:, numpy.newaxis] + numpy.ma.zeros(shape[1])
        elif len(result) == shape[1]:
            if logger is not None: logger.debug('nice_array: Broadcasting 1D values for 2D values across rows.')
            result = (result[:, numpy.newaxis] + numpy.ma.zeros(shape[1])).T
        else: raise ValueError('shape mismatch: 1D values cannot be broadcast to shape of values')

    if numpy.shape(result) != numpy.shape(result.mask):
        if logger is not None: logger.debug('Badly shaped mask in input values array. Setting mask to isnan(values).')
        result.mask = numpy.isnan(result)

    return result


def select(data, sliced=None, overlay=None, select=None):
    """
    Given a list of arrays as inputs, return selected elements from each as follows.

    slice:
    apply slice to each array.

    overlay:
    set mask for each to ~overlay. This mask can have one less dimension than the data.

    select:
    return elements corresponding to this mask (i.e. a[select]).
    """

    data = list(data)
    results = data

    if overlay is not None:
        for count, records in enumerate(data):
            if records is None: continue
            if overlay.ndim < results[count].ndim:
                new_mask = extend(~overlay, numpy.shape(records)[-1])
            else: new_mask = ~overlay
            results[count] = numpy.ma.array(records, mask=new_mask, keep_mask=True)
    
    if sliced is None and select is None:
        return results

    if sliced is not None:
        if numpy.isscalar(sliced): sliced = [sliced]
        for count, records in enumerate(data):
            if records is None: continue
            results[count] = records[slice(*sliced)]
        if select is not None: select = select[slice(*sliced)]

    if select is not None:
        for count, records in enumerate(data):
            if records is None: continue
            results[count] = records[select]

    return results


def extend(data, count, copy=False):
    """
    Given array, return array with new dimension containing duplicated values.

    If copy is False, a view of the original array is returned.
    """

    if copy: return numpy.tile(data[..., numpy.newaxis], count)
    else:
        length = data.shape[-1]
        shape = data.shape + (count,)
        strides = data.strides + (0,)
        return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    

