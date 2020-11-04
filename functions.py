"""
Some useful utility functions missing from numpy/scipy.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

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


def per_clip(data, caps):
    """
    Return values in data clipped between %le values of (caps[0], caps[1]).
    If caps is a scalar, only large values are capped.
    """

    if np.isscalar(caps):
        return np.fmin(data, np.percentile(data, caps))

    low, high = np.percentile(data, caps)
    return np.clip(data, low, high)


def scale2unit(data, eps=.1, dtype=None, soft_clip=99.99):
    """
    Scale values to between -1.0 and +1.0 strictly, and less strictly between
    -1.0 + <eps> or 1.0 - <eps>.
    
    More precisely, amplitude is scaled such that <large_value> is set to
    -1.0 + <eps> or 1.0 - <eps>, where <large_value> is
        if soft_clip is None:
            the max value of abs(data)
        else:
            soft_clip %le value of abs(data)

    Result is returned as type <dtype>, which defaults to
        if data.dtype is an integer type: float32
        else: data.dtype
    """

    if dtype is None:
        dtype = data.dtype
        if 'int' in str(dtype): dtype = np.float32
            
    data = data / (np.percentile(abs(data), soft_clip) if soft_clip
                   else np.max(abs(data)))
    if eps: data *= 1. - eps
    if soft_clip: data = np.clip(data, -1.0, 1.0)

    return data.astype(dtype, copy=False)


def softmax(data, axis=None):
    """Scale exp(data) to sum to unit along axis."""

    edata = np.exp(data)
    return edata / np.sum(edata, axis=axis)[:, None].swapaxes(-1, axis)


def sigmoid(data):
    """Sigmoid activation function."""

    return 1 / (1 + np.exp(-data))


def logit(data, eps=1e-8):
    """Inverse of the sigmoid function."""

    return -np.log(1 / (data + eps) - 1 + eps)


def elu(data, alpha=1.0, copy=True):
    """Exponential LU activation function."""

    if copy: result = data.copy()
    else: result = data
    
    mask = data < 0
    result[mask] = alpha * (np.exp(data[mask]) - 1.0)

    return result


def celu(data, alpha, copy=True):
    """Continuously differentiable exponential LU activation function."""

    if copy: result = data.copy()
    else: result = data
    
    mask = data < 0
    result[mask] = alpha * (np.exp(data[mask] / alpha) - 1.0)

    return result


def ielu(data, copy=True, eps=1e-20):
    """Inverse exponential LU activation function."""

    if copy: result = data.copy()
    else: result = data
    
    mask = data < 0
    result[mask] = np.log(data[mask] + 1.0 + eps)

    return result


def llu(data, copy=True):
    """
    Linear-log activation function; linear inside of +/-1.0,
    log outside of it.
    """

    if copy: result = data.copy()
    else: result = data
    
    mask = data > 1.0
    result[mask] = np.log(data[mask]) + 1.0

    mask = data < -1.0
    result[mask] = -np.log(-data[mask]) - 1.0

    return result


def illu(data, copy=True):
    """Inverse of llu."""

    if copy: result = data.copy()
    else: result = data

    mask = data > 1.0
    result[mask] = np.exp(data[mask] - 1.0)

    mask = data < -1.0
    result[mask] = -np.exp(-data[mask] - 1.0)

    return result


def sroot(data, power=0.5):
    """
    'Signed' square-root (default power = 0.5):
    raised abs(data) to power, then multiply by sign(data).
    """

    result = np.abs(data)**power
    return np.sign(data) * result
