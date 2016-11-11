"""
Some utilities for integer manipulation.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

def squarest_factors(integer, tile=[1, 1]):
    """
    Return the two factors A, B of given positive integer I such that
        A * B = I
    and |A*a - B*b| is as small as possible.

    Return (1, integer) if integer is prime.
    """

    if integer < 1: raise ValueError('Input must be positive integer.')

    area = integer * tile[0] * tile[1]
    rows = int(area**.5 / tile[0])
    while rows > 1 and integer % rows:
        rows -= 1

    return rows, integer / rows
