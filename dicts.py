"""Some utilities for dict analysis manipulation."""


def diff(basedict, newdict, ignore_missing=False, convert_bool=False):
    """
    Return dict of keys to (existing, new) values that represents the
    difference between <newdict> and <basedict>.
    """
    diff = {}
    for key, value in newdict.items():
        if ignore_missing and key not in basedict:
            continue
        existing = basedict.get(key)
        if convert_bool and type(value) == str:
            if value == 'true': value = True
            elif value == 'false': value = False
        if value != existing:
            diff[key] = (existing, value)
    return diff
