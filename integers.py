"""
Some utilities for integer manipulation.

Copyright 2016 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

def squarest_factors(integer):
    """
    Return the two factors A, B of given positive integer I such that
        A * B = I
    and |A - B| is as small as possible.

    Return (1, integer) if integer is prime.
    """

    if integer < 1: raise ValueError('Input must be positive integer.')

    test = int(integer**.5)
    while test > 1 and integer % test:
        test -= 1

    return test, integer / test
