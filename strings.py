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
