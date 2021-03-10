"""
Some utilities for string manipulation.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import re
import numpy

format_expression = re.compile('%([+-]*)(\d+)\.*\d*\w')

def fmt(string, format, align=None):
    """
    Convert numeric format, e.g. %9.6f, to %9s and return correspondingly
    formatted string.

    Will also convert mixed strings. e.g. '($%6.3fM)' to '($%6sM)'.
    """
    match = format_expression.search(format)
    if not match: return string
    padding = match.span()[0] + len(format) - match.span()[1]
    length = int(match.group(2)) + padding

    if align is None: align = match.group(1)
    return ('%'+align+str(length)+'s') % string


def fmt_length(format):
    """Return number of characters represented by format."""

    match = format_expression.search(format)
    if not match: return None
    padding = match.span()[0] + len(format) - match.span()[1]
    return int(match.group(2)) + padding


def fmt_matrix(mat, format, labels=None):
    """
    Return a string that represents a pretty-print of a 2D array.

    mat:    input 2D array
    format: format string to use for each element of array.
    labels: (optional) label strings for row/columns, to print as a header.
    """
    

    result = ''
    if labels is not None:
        label_len = max(len(l) for l in labels)
        label_len = max(label_len, fmt_length(format))
        label_fmt = '%' + str(label_len) + 's'
        result += ' '.join(label_fmt % l for l in [''] + list(labels)) + '\n'

    for i in range(len(mat)):
        if labels is not None: result += label_fmt % labels[i] + ' '
        result += ' '.join(format % f for f in mat[i]) + '\n'
        
    return result


def fmt_list(vals, fmt, separator=' '):
    """
    Return a string that represents a separator joined format prints of list items.

    vals:      input list
    fmt:       format string to use for each element of list.
    separator: place in between formatted strings.
    """

    if fmt[0] != '{':
        if fmt[0] != ':': fmt = ':' + fmt
        fmt = '{' + fmt + '}'

    if numpy.isscalar(vals): vals = [vals]
    if len(vals) == 0: return ''
    result = fmt.format(vals[0])
    if len(vals) == 1: return result

    for v in vals[1:]:
        result += separator
        result += fmt.format(v)
        
    return result


def infer_type(arg_str):
    """
    If the string arg_str looks numeric, return int or float, else return str.
    """

    try:
        result = int(arg_str)
    except:
        try:
            result = float(arg_str)
        except:
            return str
        return float
    return int


def convert_type(arg_str):
    """
    If the string arg_str looks numeric, return value as int or float, else:
    if 'True' or 'False': return True or False
    else: return arg_str.
    """

    try:
        result = int(arg_str)
    except:
        try:
            result = float(arg_str)
        except:
            if arg_str == 'True': return True
            if arg_str == 'False': return False
            return arg_str

    return result

    
def args2dict(arg_strs):
    """
    Given a list of command-line arguments, like
        ['level=INFO', 'layers=2', 'rate=0.5'],
    return dict like
        {'level': 'INFO', 'layers': 2, 'rate': .5}
    with appropriate type conversions for the values inferred
    (in above example, retain as string, convert to int, convert to float).

    If commas present after the '=' delimiter, value is a list.
    """
    
    kwargs = {}
    for option in arg_strs or []:
        key, value = option.split('=')
        if ',' in value:
            kwargs[key] = [convert_type(v) for v in value.split(',')]
        else:
            kwargs[key] = convert_type(value)

    return kwargs


def args2listdict(arg_strs):
    """
    Allow for arguments that do not have '=', returning them as a list.
    So a tuple is returned -- a list plus a dict.
    See doc string for args2dict.
    """

    args, kwargs = [], []
    for option in arg_strs or []:
        if '=' in option:
            kwargs.append(option)
            continue
        if ',' in option:
            args.append(convert_type(v) for v in option.split(','))
        else:
            args.append(convert_type(option))

    return args, args2dict(kwargs)
