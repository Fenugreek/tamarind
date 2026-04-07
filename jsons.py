"""Utility functions for manipulating JSON content."""

import json


def load_jsonl(filename_or_fh, filter_kv=None):
    """
    Load a JSON Lines file, returning a list of Python objects.

    Parameters
    ----------
    filename_or_fh : str or file handle
        Path to a .jsonl file, or an already-opened file handle.
    filter_kv : tuple of ([key | keys], value), optional
        If provided, only return dicts that contain the given key or nested
        keys (provided as a tuple) with the given non-None value.
        Non-dict objects are excluded.

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
        keys, value = filter_kv
        if type(keys) != tuple:
            keys = (keys,)
        max_idx = len(keys) - 1

    results = []
    for line in fh:
        if not (line := line.strip()):
            continue
        obj = json.loads(line)
        if filter_kv is None:
            results.append(obj)
            continue
        
        for idx, key in enumerate(keys):
            if not isinstance(obj, dict) or (val := obj.get(key)) is None:
                break
            if idx == max_idx:
                if val == value:
                    results.append(obj)
            else:
                obj = val

    return results

