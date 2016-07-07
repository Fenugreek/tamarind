"""
Some useful utility functions missing from numpy/scipy.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import division
import numpy as np


def dir_clip(data, clips):
    """
    'Directional' clip. Dimension of data and clips must be the same. Values in data
    are clipped according to corresponding values in clips and returned as a new array.
    new_value = portion of value between 0 and clip.
    If clip is nan, new_value = value.
    """
    
    if isinstance(data, np.ndarray): results = data.copy()
    else: results = np.array(data)

    mask = (np.sign(data) != np.sign(clips)) \
            & ~np.isnan(data) & ~np.isnan(clips)
    results[mask] = 0.0
    mask = ~mask & (abs(data) > abs(clips))
    results[mask] = clips[mask]
    return results


def toward_zero(data, value):
    """
    Subtract value from postive values of data, and add value to negative values
    of data. Do not cross zero.
    """

    results = data.copy()
    results[data > 0] -= value
    results[data < 0] += value
    results[(data > 0) & (results < 0)] = 0.0 
    results[(data < 0) & (results > 0)] = 0.0 

    return results


def per_cap(data, caps):
    """
    Return values in data clipped between %le values of (caps[0], caps[1]).
    If caps is a scalar, only large values are capped.
    """

    if np.isscalar(caps):
        return np.fmin(data, np.percentile(data, caps))

    low, high = np.percentile(data, caps)
    return np.clip(data, low, high)


def unit_scale(data, axis=None, eps=1e-8):
    """
    Scales all values in the ndarray data to be between 0 and 1.

    Adapted from deeplearning.net's utils.scale_to_unit().
    """
    
    result = data.copy()
    if axis: result = result.swapaxes(0, axis)
    
    result -= data.min(axis=axis)
    result /= result.max(axis=0 if axis else axis) + eps
    
    return result.swapaxes(0, axis) if axis else result


def softmax(data, axis=None, eps=1e-8):
    """Scale data to unit interval using softmax function."""
    
    return unit_scale(np.exp(data), axis=axis, eps=eps)


def sigmoid(data):
    """Sigmoid activation function."""

    return 1 / (1 + np.exp(-data))


def logit(data, eps=1e-8):
    """Inverse of the sigmoid function."""

    return -np.log(1 / (data + eps) - 1 + eps)
