"""
Some utilities for string manipulation.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import re
import numpy, unidecode

format_expression = re.compile('%([+-]*)(\d+)\.*\d*\w')

def slugify(input_str, lower=True):
    """Transform input string <text> to URL friendly name."""
    output = unidecode.unidecode(input_str)
    if lower:
        output = output.lower()
    return re.sub(r'[\W_]+', '-', output)


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
    if 'True' or 'False' or 'None': return True or False or None.
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
            if arg_str == 'None': return None
            return arg_str

    return result

    
def args2dict(arg_strs, **kwargs):
    """
    Given a list of command-line arguments, like
        ['level=INFO', 'layers=2', 'rate=0.5'],
    return dict like
        {'level': 'INFO', 'layers': 2, 'rate': .5}
    with appropriate type conversions for the values inferred
    (in above example, retain as string, convert to int, convert to float).

    If commas present after the '=' delimiter, value is a list.

    If more than one '=' present in an argument, return a nested dict as follows:
        ['age=Deepak=12']
    returns
        {'age': {'Deepak': 12}}
        
    kwargs supplied are returned by default.
    """
    
    if type(arg_strs) == str:
        arg_strs = [arg_strs]

    for option in arg_strs or []:
        key, value = option.split('=', maxsplit=1)
        if '=' in value:
            key2, value = value.split('=')
            key = [key, key2]

        if ',' in value:
            value = [convert_type(v) for v in value.split(',')]
        else:
            value = convert_type(value)

        if type(key) == list:
            if key[0] in kwargs:
                kwargs[key[0]][key[1]] = value
            else:
                kwargs[key[0]] = {key[1]: value}
        else:
            kwargs[key] = value

    return kwargs


def args2listdict(arg_strs, **kwargs):
    """
    Allow for arguments that do not have '=', returning them as a list.
    So a tuple is returned -- a list plus a dict.
    See doc string for args2dict.

    kwargs supplied are returned by default.    
    """

    if type(arg_strs) == str:
        arg_strs = [arg_strs]

    args, kwargs_list = [], []
    for option in arg_strs or []:
        if '=' in option:
            kwargs_list.append(option)
            continue
        if ',' in option:
            args.append(convert_type(v) for v in option.split(','))
        else:
            args.append(convert_type(option))

    return args, args2dict(kwargs_list, **kwargs)


def lines2dict(lines, **kwargs):
    """
    Convert lines of text, such as that output by pprint.pprint(),
    back to a dict.

    kwargs supplied are returned by default.
    """
    string = lines[0].rstrip()
    for i in range(1, len(lines)):
        string += lines[i].rstrip()
    return {**eval(string), **kwargs}


def abbrev(string, length, pfx=None, sfx=None, spanner='...'):
    """
    Abbreviate <string> if longer than <length> as
    '<string[:pfx]><spanner><string[-sfx:]>'

    Soecifying one of <pfx> or <sfx> will infer the other.
    If neither given, a <sfx> that is 2 * <pfx> will be chosen.
    """
    if length >= len(string):
        return string

    len_span = len(spanner)
    if pfx is not None:
        if sfx is not None:
            assert(pfx + sfx + len_span == length)
        else:
            sfx = length - len_span - pfx
    elif sfx is not None:
        pfx = length - len_span - sfx
    else:
        pfx = (length - len_span) // 3
        sfx = length - len_span - pfx

    return string[:pfx] + spanner + string[-sfx:]
