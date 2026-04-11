"""Some utilities for dict analysis manipulation."""
import csv
from functools import reduce
import operator
from collections import defaultdict

def gets(basedict, keys):
    """
    Given a list of keys, return corresponding list of values.
    """
    return [basedict.get(k) for k in keys]


def get_nested(basedict, keys):
    """
    Given a list of nested keys, return basedict[keys[0]][keys[1]][...]
    """
    return reduce(operator.getitem, keys, basedict)


def dict_list(records, key, field=None):
    """
    Given a list of records and a key field, return a dict with values of key field
    as keys and corresponding records as lists of values.

    If field is not None, returned values are record[field]. If field is not found
    in a record or the value is None, the record is skipped.

    If key not found in a record or record[key] is None, that record is skipped.
    """

    output = defaultdict(list)
    for entry in records:
        if (identifier := entry.get(key)) is None:
            continue
        if field is not None:
            if field in entry:
                output[identifier].append(entry[field])
        else:
            output[identifier].append(entry)

    return dict(output) # Remove defaultdict


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


def nodiff(basedict, newdicts, fields=None):
    """
    Check if newdicts (list of dicts)  have same values of basedict for given fields.
    If fields is None, all keys of basedict are checked.
    """
    if fields is None:
        fields = basedict.keys()

    for f in fields:
        val = basedict.get(f)
        for newdict in newdicts:
            if newdict.get(f) != val:
                return False

    return True


def csv_to_dict(filename, index_field):
    """
    Read a CSV file and return a dictionary indexed by the specified field.
    
    Args:
        filename (str): Path to the CSV file
        index_field (str): Name of the field to use as dictionary keys
    
    Returns:
        dict: Dictionary where keys are values from index_field and values are row dictionaries
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        KeyError: If the index_field is not found in the CSV headers
        ValueError: If there are duplicate values in the index field
    """
    result_dict = {}
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if index_field exists in headers
            if index_field not in reader.fieldnames:
                raise KeyError(f"Index field '{index_field}' not found in CSV headers: {reader.fieldnames}")
            
            for row in reader:
                key = row[index_field]
                
                # Check for duplicate keys
                if key in result_dict:
                    raise ValueError(f"Duplicate value '{key}' found in index field '{index_field}'")
                
                result_dict[key] = row
                
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{filename}' not found")
    
    return result_dict
