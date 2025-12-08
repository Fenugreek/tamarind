"""Some utilities for sqlitedict analysis and manipulation."""
from sqlitedict import SqliteDict
from collections import OrderedDict, defaultdict, Counter

def invert(sql_fname, exclude=None, include=None):
    """
    Reverse the dict implicit in the sqlite dict.
    Note: If keys are not unique, this is likely not a good call.
    """
    result = {}
    for table in SqliteDict.get_tablenames(sql_fname):
        if include and table not in include: continue
        if exclude and table in exclude: continue
        for key in SqliteDict(sql_fname, table).keys():
            result[key] = table
    return result


def table_sets(sql_fname):
    """
    Return dict with tablenames as keys and a set of table keys as the values.
    """
    result = {}
    for table in SqliteDict.get_tablenames(sql_fname):
        result[table] = set(SqliteDict(sql_fname, table).keys())
    return result


def table_counts(sql_fname, field=None, key=None, **kwargs):
    """
    Return counts of number of records for each table.
    If field is given, group by field.
    If key function is given, group by key(field).
    """
    tables = SqliteDict.get_tablenames(sql_fname)
    tables.sort()

    result = {}
    for table in tables:
        sd = SqliteDict(sql_fname, tablename=table, **kwargs)
        if not field:
            result[table] = len(list(sd.keys()))
        else:
            vals = []
            for row in sd.values():
                if type(row) == dict:
                    val = row.get(field)
                    if key: val = key(val)
                else:
                    val = None
                vals.append(val)
            result[table] = Counter(vals)

    return result


def append(sql_fname, add_sql_fname, tables=None, overwrite=False, **kwargs):
    """
    Add entries in <add_sql_fname> to <sql_fname>.

    tables:
    For only this table / list of tables.

    overwrite:
    If true, overwrite existing entries.
    """

    if tables is None:
        tables = SqliteDict.get_tablenames(add_sql_fname)
        tables.sort()
    elif type(tables) == str:
        tables = [tables]

    added = 0
    for table in tables:
        sd = SqliteDict(sql_fname, tablename=table, **kwargs)
        add_sd = SqliteDict(add_sql_fname, tablename=table, **kwargs)
        if overwrite:
            for key, val in add_sd.items():
                sd[key] = val
                added += 1
        else:
            for key, val in add_sd.items():
                if key not in sd:
                    sd[key] = val
                added += 1
        sd.commit()

    return added
