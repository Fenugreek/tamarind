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
    Returns number of entries added, and list of pre-existing keys that were skipped.

    tables:
    For only this table / list of tables.

    overwrite:
    If true, overwrite existing entries.

    kwargs:
    kwargs to pass to SqliteDict constructor. Useful if encoded.
    """

    if tables is None:
        tables = SqliteDict.get_tablenames(add_sql_fname)
        tables.sort()
    elif type(tables) == str:
        tables = [tables]

    added, preexisting = 0, []
    for table in tables:
        sd = SqliteDict(sql_fname, tablename=table, **kwargs)
        add_sd = SqliteDict(add_sql_fname, tablename=table, **kwargs)
        if overwrite:
            for key, val in add_sd.items():
                sd[key] = val
                added += 1
        else:
            for key, val in add_sd.items():
                if key in sd:
                    preexisting.append((table, key))
                else:
                    sd[key] = val
                    added += 1
        sd.commit()

    return added, preexisting


def add_columns(sql_fname, add_sql_fname, columns, tables=None, overwrite=False, **kwargs):
    """
    Add columnar values in <add_sql_fname> to <sql_fname> where same keys exist.
    Returns number of entries added, and dict of (table, keys) which had pre-existing values
    (that were skipped or overwritten, as per overwite flag), and
    dict of (table, keys) of entries in source that did not exist in target.

    tables:
    For only this table / list of tables.

    overwrite:
    If true, overwrite existing entries.
    """

    if type(columns) == str:
        columns = [columns]

    if tables is None:
        tables = SqliteDict.get_tablenames(add_sql_fname)
        tables.sort()
    elif type(tables) == str:
        tables = [tables]
    existing_tables = [t.lower() for t in SqliteDict.get_tablenames(sql_fname)]
    add_tables = [t.lower() for t in SqliteDict.get_tablenames(add_sql_fname)]

    added, missing = defaultdict(int), defaultdict(list)
    preexisting = dict((c, defaultdict(list)) for c in columns)
    for table in tables:
        table = table.lower()
        if table not in add_tables:
            continue
        add_sd = SqliteDict(add_sql_fname, tablename=table, **kwargs)
        if table not in existing_tables:
            missing[table] = list(add_sd.keys())
            continue

        sd = SqliteDict(sql_fname, tablename=table, **kwargs)
        table_entry_added = added[table]
        for key, entry in add_sd.items():
            column_vals = {}
            for col in columns:
                if val := entry.get(col):
                    column_vals[col] = val
            if not column_vals:
                continue

            if (row := sd.get(key)) is None:
                missing[table].append(key)
                continue

            row_entry_added = added[table]
            for col, val in column_vals.items():
                if row.get(col):
                    preexisting[col][table].append(key)
                    if not overwrite:
                        continue
                row[col] = val
                added[table] += 1
            if row_entry_added != added[table]: # updated
                sd[key] = row

        if table_entry_added != added[table]: # updated
            sd.commit()

    return added, preexisting, missing


def write_dict(input_dict, sql_fname):
    """
    Write <input_dict> to sql_fname, with keys as tablenames.
    Values of <input_dict> are assumed to be dicts too, and their keys are used
    as keys into the table.
    """
    for table_name, entry in input_dict.items():
        sql = SqliteDict(sql_fname, tablename=table_name, autocommit=False)
        for key, value in entry.items():
            sql[key] = value
        sql.commit()


def read_dict(sql_fname):
    """
    Return a dict with tablenames as keys and contents of each table as values
    (dicts themselves).
    """
    result = {}
    for table_name in SqliteDict.get_tablenames(sql_fname):
        result[table_name] = dict(SqliteDict(sql_fname, tablename=table_name))
    return result
