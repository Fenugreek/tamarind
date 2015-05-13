"""
Some utilities for string manipulation.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import re

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
