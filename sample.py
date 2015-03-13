"""
Some utilities for sampling elements from numpy arrays.
"""

from __future__ import division
import numpy
from numpy.random.mtrand import randint


def sample_draw(count, size):
    """
    Return random sample (without replacement) of count integers between 0 and size.
    """
    
    if count == 0: return []
    elif count > size: raise AssertionError('count can not be > size')
    elif count < 0: raise AssertionError('count cannot be negative')
    if size < 1: raise AssertionError('size must be > 1')
    
    deck = range(size)
    for index in range(count):
        current = deck[index]
        swap_index = randint(index, size)
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
    

def sample_where(data, count=None):
    """
    Get random sample of True elements from a multi-dimensional bool array, returning
    indices ala numpy.where(). Sampling is without replacement.

    If count is None, return random element's indices.
    If 0 < count < 1.0, return count fraction of True elements' indices.
    """

    true_indices = numpy.where(data)
    if count is None:
        random_slice = randint(len(true_indices[0]))
        return tuple([d[random_slice] for d in true_indices])
    elif count == 0: return None
    elif count < 1:
        count = int(count * numpy.sum(data))
        
    indicesT = []
    for random_slice in sample_draw(count, len(true_indices[0])):
        indicesT.append([d[random_slice] for d in true_indices])
    return tuple([d for d in numpy.array(indicesT).T])

    
def sample_mask(data, count=None):
    """
    Like sample_where, except a mask with same shape of data is returned, with
    True for the chosen elements.
    """

    indices = sample_where(data, count)
    mask = numpy.zeros(numpy.shape(data), dtype=bool)
    mask[indices] = True
    return mask


