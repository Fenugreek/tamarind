"""Utility functions for manipulating JSON content."""

import json


def load_jsonl(filename_or_fh, filter_kv=None, filter_keys=None):
    """
    Load a JSON Lines file, returning a list of Python objects.

    Parameters
    ----------
    filename_or_fh : str or file handle
        Path to a .jsonl file, or an already-opened file handle

    filter_kv : tuple of ([key | keys], value), optional
        If provided, only return dicts that contain the given key or nested
        keys (provided as a tuple) with the given non-None value.
        Non-dict objects are excluded.

    filter_keys : list of {key | keys}, optional
        If provided, only return dicts that contain the given [nested] key[s].
        Non-dict objects are excluded.

    Currently only one of the above two filters may be specified.
    'None' is not a supported value when using above filters.
    
    Returns
    -------
    list
        Parsed objects from each line.
    """

    if hasattr(filename_or_fh, 'read'):
        fh = filename_or_fh
    else:
        fh = open(filename_or_fh, encoding='utf-8', errors='ignore')

    if filter_kv is not None:
        if filter_keys is not None:
            raise ValueError('Cannot specify both filter_kv and filter_keys options.')
        keys, value = filter_kv
    elif filter_keys is not None:
        keys, value = filter_keys, None
    else:
        keys = None
    if keys is not None:
        if type(keys) != tuple:
            keys = (keys,)
        max_idx = len(keys) - 1

    results = []
    for line in fh:
        if not (line := line.strip()):
            continue
        obj = json.loads(line)
        if keys is None:
            results.append(obj)
            continue

        record = obj
        for idx, key in enumerate(keys):
            if not isinstance(record, dict) or (val := record.get(key)) is None:
                break
            if idx == max_idx:
                if value is None or (val == value):
                    results.append(obj)
            else:
                record = val

    return results

