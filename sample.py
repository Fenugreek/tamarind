"""
Some utilities for sampling elements from numpy arrays.
"""


import numpy as np
from numpy.random import randint, random_sample


def sample_draw(count, size, weights=None):
    """
    Return random sample (without replacement) of <count> integers between 0 and <size>.

    weights:
    An optional array of length <size> weighting each integer between 0 and <size>
    according to likelihood of the integer being chosen. i.e. weighted sampling is done
    (without replacement).
    """
    
    if count == 0: return []
    elif count > size: raise AssertionError('count can not be > size')
    elif count < 0: raise AssertionError('count cannot be negative')
    if size < 1: raise AssertionError('size must be > 1')        

    deck = list(range(size))
    for index in range(count):
        if weights is None:
            swap_index = randint(index, size)
        else:
            cum_weights = np.cumsum(weights[index:])
            swap_index = index + np.searchsorted(cum_weights,
                                                 random_sample() * cum_weights[-1])
            current = weights[index]
            weights[index] = weights[swap_index]
            weights[swap_index] = current
            
        current = deck[index]            
        deck[index] = deck[swap_index]
        deck[swap_index] = current
        
    return deck[:count]


def sample(data, count=1, axis=None):
    """
    Get random sample data. If count is None, return random element.
    Sampling is without replacement.
    """
    
    if count == 0: return None

    if axis is None: view = data.flat
    else: view = data.swapaxes(0, axis)
    
    result = view[sample_draw(count, len(view))]
    return result.swapaxes(axis or 0, 0)

    
def sample_each_col(data, count=1):
    """
    Get random <count> rows from data, where each column is sampled independently.
    Sampling is without replacement.
    """
    shape = data.shape
    sample = np.empty([count] + list(shape[1:]), dtype=data.dtype)
    for col in range(shape[1]):
        draw = sample_draw(count, shape[0])
        sample[:, col, ...] = data[draw, col, ...]
    return sample


def sample_where(data, count=None):
    """
    Get random sample of True elements from a multi-dimensional bool array, returning
    indices ala np.where(). Sampling is without replacement.

    If count is None, return random element's indices.
    If 0 < count < 1.0, return count fraction of True elements' indices.
    """

    true_indices = np.where(data)
    if count is None:
        random_slice = randint(len(true_indices[0]))
        return tuple([d[random_slice] for d in true_indices])
    elif count == 0: return None
    elif count < 1:
        count = int(count * np.sum(data))
        
    indicesT = []
    for random_slice in sample_draw(count, len(true_indices[0])):
        indicesT.append([d[random_slice] for d in true_indices])
    return tuple([d for d in np.array(indicesT).T])

    
def sample_mask(data, count=None):
    """
    Like sample_where, except a mask with same shape of data is returned, with
    True for the chosen elements.
    """

    indices = sample_where(data, count)
    mask = np.zeros(np.shape(data), dtype=bool)
    mask[indices] = True
    return mask


