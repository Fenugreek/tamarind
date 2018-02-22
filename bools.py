"""
Some utilities for bools manipulation.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

def fields_test(dictionary, conditions):
    """
    Return +1 if all conditions are satisfied; 0 if at least one (but not all)
    conditions are satisfied; and -1 no conditions are satisfied.

    conditions:
    dictionary with keys corresponding to keys in dictionary, and values which are
    tuples of the form (+2|+1|0|-1|-2|None, val).
    +2 meaning dictionary[key] >  val,
    +1 meaning dictionary[key] >= val,
     0 meaning dictionary[key] == val,
    -1 meaning dictionary[key] <= val,
    -2 meaning dictionary[key] <  val,
    None meaning dictionary[key] != val.
    """

    count = 0
    net = 0
    for key, cond_value in list(conditions.items()):
        count += 1
        cond, value = cond_value
        field_value = dictionary[key]
        if cond == 1:
            result = field_value >= value
        elif cond == -1:
            result = field_value <= value
        elif cond == 0:
            result = field_value == value
        elif cond == 2 :
            result = field_value >  value
        elif cond == -2:
            result = field_value <  value
        elif cond == 0:
            result = field_value != value
        else: raise AssertionError('Bad condition ' + str(cond))
        net += result

    if net == count: return 1
    elif net > 0: return 0
    else: return -1
