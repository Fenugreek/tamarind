from __future__ import division
"""
Implement some econometrics.

Copyright 2015 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import numpy
from numpy import sum

def gini_coeff(count_income):
    """
    Compute Gini coefficient. 0.0 == equality. 1.0 == inequality.

    count_income:
    list of 2-tuples, (# of people, mean income), sorted by mean income.
    """

    data = numpy.array(count_income)
    data[:, 0] = data[:, 0] / sum(data[:, 0])
    data[:, 1] = data[:, 1] / sum(data[:, 0] * data[:, 1])

    cumulative = 0
    result = 0
    for i in range(len(data)):
        cumulative += .5 * data[-1-i, 0]
        if i: cumulative += .5 * data[-i, 0]
        result += cumulative * data[-1-i, 0] * data[-1-i, 1]

    return 1 - 2 * result


def theil_index(count_income):
    """
    Compute Theil index, after normalizing count and total income to 1.0.
    0.0 == equality. 1.0++ == inequality.
    
    count_income:
    list of 2-tuples, (# of people, mean income), sorted by mean income.
    """

    data = numpy.array(count_income)
    data[:, 0] = data[:, 0] / sum(data[:, 0])
    data[:, 1] = data[:, 1] / sum(data[:, 0] * data[:, 1])

    net = sum(data[:, 0] * data[:, 1])
    mean = net / sum(data[:, 0])

    return sum(data[:, 0] * data[:, 1] * numpy.log(data[:, 1] / mean)) / net
