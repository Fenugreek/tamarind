"""
Some utilities for numpy array manipulation.

Copyright 2013-2014 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import division
import cPickle
import numpy


def load(fromfile, list=False):
    """
    Load from file where array(s) was previously cPickle'd.

    fromfile:
    filename or file handle.
    If file handle, file handle is closed before results are returned.
    
    list:
    If True, multiple arrays are returned, as a list;
    as many as were saved to the file.
    """

    handle = fromfile if type(fromfile) == file else open(fromfile, 'rb')
    arr = cPickle.load(handle)
    if not list:
        handle.close()
        return arr

    results = [arr]
    while True:
        try: results.append(cPickle.load(handle))
        except EOFError: break

    handle.close()
    return results


def save(arrs, tofile):
    """
    Save array(s) to file using cPickle.

    arrs:
    A numpy array, or a list of numpy arrays.

    tofile:
    filename or file handle.
    If file handle, file handle is not closed after array(s) are saved.    
    """

    if type(arrs) != list: arrs = [arrs]

    handle = open(tofile, 'wb') if type(tofile) == str else tofile
    for arr in arrs: cPickle.dump(arr, handle)
    if type(tofile) == str: handle.close()


def nans(shape):
    """Return an array of shape shape filled with nans."""
    result = numpy.empty(shape)
    result.fill(numpy.nan)
    return result


def isnanzero(data):
    """Return mask with True values corresponding to nan or 0 in input array"""
    if data is None: return True
    if ~numpy.isscalar(data) and ~isinstance(data, numpy.ndarray):
        data = numpy.array(data)
    return numpy.isnan(data) | (data==0)


def plane(data):
    """
    Flatten data to 2D, retaining original length of the 1st dimension.
    If data is 1D, 2nd dimension has length of 1.
    """
    return numpy.reshape(data, (data.shape[0], -1))


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


def percentile(arr, quants, weights):
    """
    Weighted version of numpy's percentile function.
    No interpolation is done.
    """
    
    sort_indices = numpy.argsort(arr)
    sort_weights = weights[sort_indices]
    cum_weights = numpy.cumsum(sort_weights)
    cum_weights = cum_weights / cum_weights[-1]
    indices = numpy.searchsorted(cum_weights, quants)

    return arr[sort_indices][indices]


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


def percentile_rank(data, weights=None, axis=None, reverse=False, mask=None):
    """
    Returns percentile rank of elements in given array.

    mask:
    mask with same dimensions as data with True where elements are to be ignored.
    """
    
    if weights is None: weights = numpy.ma.ones(numpy.shape(data))
    else: weights = nice_array(weights)
    
    if axis is None:
        sort_indices = argsort(data, reverse=reverse, mask=mask)
        cum_weights = numpy.ma.cumsum(weights[sort_indices])
        percentiles = nans(weights.shape)
        percentiles[sort_indices] = cum_weights / numpy.max(cum_weights)
        return percentiles
        
    if not isinstance(data, numpy.ndarray): data = numpy.array(data)    
    sort_indices = argsort(data.swapaxes(axis, -1), mask=mask,
                           reverse=reverse, last_dim=True)

    percentiles = nans(data.swapaxes(axis, -1).shape)
    weights = weights.swapaxes(axis, -1)
    for ndindex, sort_list in numpy.ndenumerate(sort_indices):
        cum_weights = numpy.ma.cumsum(weights[ndindex][sort_list])
        percentiles[ndindex][sort_list] = cum_weights / numpy.max(cum_weights)

    return percentiles.swapaxes(-1, axis)


def nice_array(values, shape=None, logger=None, copy=False):
    """
    Utility function to convert input data to a nice ma.array with appropriate shape
    if necessary. Converts integer arrays to float arrays.
    """
    
    if isinstance(values, numpy.ma.masked_array):
        if values.dtype.kind == 'i': result = numpy.ma.asarray(values, dtype=float)
        elif copy: result = values.copy() 
        else: result = values
        result.fill_value = numpy.nan
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
    

def nanmask(data, fields=[], overlay=None, copy=True):
    """
    Given a 1D/2D recarray, return a MaskedArray with same data, with a mask
    where nan values are present.

    overlay:
    mask with same dimensions as data to layer on top of nan mask.
    """

    bool_dtype = numpy.dtype([(i, bool) for i in data.dtype.names])
    mask = numpy.zeros(numpy.shape(data), dtype=bool_dtype)

    if not len(fields):
        for field_descr in data.dtype.descr:
            if 'f' in field_descr[1]: fields.append(field_descr[0]) 
    for field in fields:
        if overlay is not None: mask[field] = numpy.isnan(data[field]) | overlay
        else: mask[field] = numpy.isnan(data[field])
        
    return numpy.ma.array(data, mask=mask, copy=copy)


def construct_map(source_IDs, other_ID, overlay=None):
    """
    Given a list of symbols, and a dict symbol->index, return a tuple of
    lists mapping indices from the first list to the second (for elements
    in intersection of the symbols only).

    overlay:
    boolean list with elements in source_IDs to selectively include in the results.
    """

    source, other = [], []
    for count, identifier in enumerate(source_IDs):
        if overlay is not None and not overlay[count]: continue
        if identifier in other_ID:
            source.append(count)
            other.append(other_ID[identifier])

    return (source, other)


def align_sorted(source_IDs, other_IDs, overlay=None):
    """
    Given two lists of elements (second one sorted), return a tuple of lists
    mapping indices from the first list to the second s.t. the mapped-to element
    is closest to the mapped-from element in the mapped-to <= mapped-from sense.

    overlay:
    boolean list with elements in source_IDs to not include in the results.
    """
    
    source, other = [], []
    indices = numpy.searchsorted(other_IDs, source_IDs, side='right')
    for count, index in enumerate(indices):
        if overlay is not None and not overlay[count]: continue
        if index == 0: continue
        source.append(count)
        other.append(index - 1)

    return (source, other)


def dict_array(records, key, field=None):
    """
    Given a recarray and a key field, return a dict with values of key field
    as keys and corresponding recarray records as values.

    If field is not None, returned values are recarray[field].
    """

    output = {}
    for count, entry in enumerate(records):
        identifier = entry[key]
        if identifier not in output: output[identifier] = [count]
        else: output[identifier].append(count)
        
    if field is None:
        for identifier in output.keys():
            output[identifier] = records[output[identifier]]
    else:
        for identifier in output.keys():
            output[identifier] = records[output[identifier]][field]

    return output


def group_mask(values):
    """
    Returns list of (group, mask) where group is every distinct value in values,
    and mask is True for every element in values that equals this distinct value.
    """
    
    groups = numpy.unique(values)
    return [(group, values == group) for group in groups]


def member_mask(values, other):
    """
    Returns mask which is True for every element in values that == some element in other.
    """

    dict_other = dict((o, True) for o in other)
    return numpy.array([v in dict_other for v in values])


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


def autocorr(arr):
    """
    Returned lagged autocorrelation coefficients of given 1D array.
    Lags are up to half the length of input array, starting with lag of 1.
    """
    
    length = len(arr)//2
    results = numpy.empty(length)
    for i in range(1, length + 1):
        results[i - 1] = numpy.corrcoef(arr[:-i], arr[i:])[1, 0]

    return results

    
def rolling_window(a, size, offset=1):
    """
    [adapted from Erik Rigtorp <erik@rigtorp.com>]
    
    Make an ndarray with a rolling window of the last dimension; each window
    offset by offset elements.

     Parameters
     ----------
     a : array_like
         Array to add rolling window to
     size : int
         Size of rolling window

     Returns
     -------
     Array that is a view of the original array with a added dimension
     of size w.

     Examples
     --------
     >>> x=np.arange(10).reshape((2,5))
     >>> rolling_window(x, 3)
     array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

     Calculate rolling mean of last dimension:
     >>> np.mean(rolling_window(x, 3), -1)
     array([[ 1.,  2.,  3.],
            [ 6.,  7.,  8.]])
    """

    length = a.shape[-1]
    if size < 1:
        raise ValueError, "`size` must be at least 1."
    if size > length:
        raise ValueError, "`size` is too long."

    # remainder is the last few elements of array a that are left out
    # because (offset, size) combination doesn't fit length of a.
    remainder = (length - size) % offset
    shape = a.shape[:-1] + (int((length - size) / offset) + 1, size)
    strides = a.strides[:-1] + (a.strides[-1] * offset, a.strides[-1])
    return numpy.lib.stride_tricks.as_strided(a[..., :length-remainder],
                                                shape=shape, strides=strides)
  

def zip_func(func, *args):
    """
    For a variable number of arrays and scalars, zip elements and apply func across them.

    a = array([-1.5,  2.4,  5.1,  9.8,  nan])
    b = a + 0.01
    zip_func(numpy.min, a, b, 5)
      => array([-1.5,  2.4,  5.0,  5.0,  nan])
    """
    
    if len(args) < 2: return args[0]
    
    length = 0
    dtype = None
    for arg in args:
        if not numpy.isscalar(arg):
            length = len(arg)
            dtype = arg.dtype
            break
    if not length: return func(args)

    stack = []
    for arg in args:
        if numpy.isscalar(arg):
            arg_array = numpy.empty(length, dtype=dtype)
            arg_array.fill(arg)
            stack.append(arg_array)
        else: stack.append(arg)
    return func(numpy.vstack(stack), axis=0)

