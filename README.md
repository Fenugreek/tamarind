Panel Data and Numpy Utils
====================

Python and numpy based library for housing, examining and modeling tabular panel data. Some additional numpy utilities are implemented.

### Notable modules:
- **datab** :
numpy recarray subclass, for housing tabular data. Keeps track of formatting of the fields, and has methods for data inspection and disk I/O.

- **stats** :
Keep track of (optionally weighted) descriptive statistics of incoming data, handling nans. Handles bivariate data, and two-dimensional data.

- **regress** :
Perform multivariate weighted linear regression, handling nans and returning
associated statistics.

- **strings** :
Utilities for string manipulation not found in the standard library, including numpy-based formatting and unidecode support.

- **dicts** :
Utilities for dict analysis and manipulation, including diffing two dicts and CSV I/O.

- **bools** :
Utilities for boolean logic on dict fields, e.g. multi-condition tests.

- **integers** :
Utilities for integer manipulation, e.g. finding the squarest factor pair of an integer.

- **structs** :
Useful data structures missing from standard Python, including a `Trie` (prefix tree) implementation.

- **functions** :
Useful numeric/array utility functions missing from numpy/scipy.

Installation
------------
From the directory where you downloaded the files, run the following command-line to install the library:

```
 $ python setup.py install
```

Or simply place all files into a directory called `tamarind/` somewhere in your `$PYTHONPATH`.

**Dependencies** : numpy, scipy.

Synopsis
---------------
Under construction.
