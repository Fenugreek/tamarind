'''Utilities for counting appearances of items, bucketing by categories.'''
from collections import defaultdict
from copy import deepcopy

def counts(data, key, fields=[], label_all='ALL', label_none='NONE'):
    """
    Given a list of dicts, return a dict keyed by field values of <key>
    that contains counts of appearances.
    """
    default = dict((f, defaultdict(lambda : 0)) for f in fields)
    default[label_all] = 0
    results = defaultdict(lambda : deepcopy(default))

    for row in data:
        val = row.get(key, label_none) or label_none
        entry = results[val]
        entry[label_all] += 1
        for f in fields:
            val = row.get(f, label_none) or label_none
            entry[f][val] += 1

    return results
