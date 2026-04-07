"""Utility functions for manipulating JSON content."""

import json


def load_jsonl(filename_or_fh, filter_kv=None):
    """
    Load a JSON Lines file, returning a list of Python objects.

    Parameters
    ----------
    filename_or_fh : str or file handle
        Path to a .jsonl file, or an already-opened file handle.
    filter_kv : tuple of (key, value), optional
        If provided, only return dicts that contain the given key with the
        given value. Non-dict objects are also excluded when filtering.

    Returns
    -------
    list
        Parsed objects from each line.
    """
    def _read_lines(fh):
        results = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if filter_kv is not None:
                if not isinstance(obj, dict):
                    continue
                if obj.get(filter_kv[0]) != filter_kv[1]:
                    continue
            results.append(obj)
        return results

    if hasattr(filename_or_fh, 'read'):
        return _read_lines(filename_or_fh)

    with open(filename_or_fh) as fh:
        return _read_lines(fh)
