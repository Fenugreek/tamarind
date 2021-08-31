"""
numpy recarray subclass, for housing tabular data. Keeps track of
formatting of the fields, and has methods for data inspection and disk I/O.

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os, sys, re, gzip, io
import numpy
from . import strings, arrays, logging

class Datab(numpy.ndarray):
    """
    numpy recarray subclass, for housing tabular data. Keeps track of
    formatting of the fields, and has methods for data inspection and disk I/O.

    ATTRIBUTES

    obj.spec:
    A list of tuples, each tuple representing a field in the rec array.
    The first two elements of each tuple indicate the name of the field and its
    datatype, e.g. ('quantity', int). The third element, the format spec, e.g. '%6d'.
    The fourth, the default value in case of missing data, e.g. -1.

    obj.field_spec:
    A dict with field names as keys and the above mentioned tuples as values.

    obj.index (optional):
    A dict with field values (for specified field) as keys and the array indices
    as values, so elements in the recarray can be looked up via hashing a key.

    obj.identifier (optional):
    Used to store the name of a special column; defaults to index option.

    obj.empty_record:
    An empty record with default values (fourth element of tuples) filled, so new
    missing data can be easily instantiated.

    obj.logger:
    logging object for printing error messages.
    """

    # formatting and missing value defaults, if not specified in spec.
    field_defaults = {'float': ('%9.6f', numpy.nan),
                      'int': ('%9d', 0),
                      'bool': ('%1d', False),
                      'str': (None, '')}

    @staticmethod
    def _read_from_file(filename, spec=None, separator=None, match_pattern=None,
                        select_field_values=None, skip_field_values=None,
                        select_fields=[], skip_fields=[], logger='warning'):
        """
        Utility function for constructing object from file on disk. Understands
        .gz extension. Returns data as list of tuples, and the inferred spec.

        spec:
        if not given, inferred from file header (must be in Datab format).

        separator:
        regexp for splitting each line in the file into columns. Defaults to whitespace.

        match_pattern:
        ignore lines in the file that don't match given pattern.
        NOTE: lines beginning with '#' always ignored.

        select_field_values:
        list of (<field>, set([str1, str2, ...])) where rows are to be loaded only if
        the value for each <field> is in [str1, str2, ...]
        
        skip_field_values:
        list of (<field>, set([str1, str2, ...])) where rows are to be loaded only if
        the value for each <field> is not in [str1, str2, ...]
        
        select_fields:
        list of fields to restrict loading.
        
        skip_fields:
        list of fields to skip loading.

        logger:
        Log error messages using this logging object; if a string, construct one
        with this loglevel.
        """
        
        if select_fields and skip_fields:
            raise AssertionError('Cannot specify both select_fields and skip_fields options')

        if 'file' in str(type(filename)):
            stream = filename
        else:
            if not os.path.isfile(filename):
                if os.path.isfile(filename + '.gz'): filename += '.gz'
                else:
                    logger.warning('Returning None because file does not exist: %s', filename)
                    return None, None
            if filename[-3:] == '.gz': stream = gzip.open(filename)
            else: stream = open(filename)
        lines = stream.readlines()

        file_separator = None
        if spec is None: spec, file_separator = Datab._get_spec(lines)
        if spec is None: raise AssertionError('No spec from file ' + filename)
        if file_separator is not None:
            if separator is not None and separator != file_separator:
                logger.warning('Given separator %s differs from separator in file header for file %s.',
                               separator, tokens[1], filename)
            separator = file_separator

        # parse specified inclusion and exclusion criteria
        field_index = dict((field_spec[0], count)
                           for count, field_spec in enumerate(spec))
        inclusion = [(field_index[fv[0]], fv[1]) for fv in select_field_values or []]
        exclusion = [(field_index[fv[0]], fv[1]) for fv in skip_field_values or []]

        # if user specifies fields to keep/skip, note their positions
        include_fields = []
        if select_fields:
            for f in select_fields:
                if f in field_index: include_fields.append(field_index[f])
        elif skip_fields:
            for field, count in list(field_index.items()):
                if field not in skip_fields: include_fields.append(count)
        if select_fields or skip_fields:
            if not include_fields: raise AssertionError('No fields in file after applying select/skip fields')
            spec = [spec[i] for i in include_fields]

        if match_pattern:
            match_expression = re.compile(match_pattern)
            lines = list(filter(match_expression.search, lines))

        data = []
        for line in lines:
            if line[0] == '#': continue
            row_data = line.split(separator)

            if select_field_values:
                include = True
                for criterion in inclusion:
                    if row_data[criterion[0]] not in criterion[1]:
                        include = False
                        break
                if not include: continue

            if skip_field_values:
                include = True
                for criterion in exclusion:
                    if row_data[criterion[0]] in criterion[1]:
                        include = False
                        break
                if not include: continue

            if include_fields: row_data = [row_data[i] for i in include_fields]

            data.append(tuple(row_data))

        return spec, data


    @staticmethod
    def _get_spec(lines):
        """
        Given lines from a file, determine and return the Datab spec and
        field separator.
        """
        
        spec_t = [[], [], []]
        separator = None
        while len(lines) and lines[0][0] == '#':
            tokens = lines.pop(0).split()
            if tokens[0] == '#fields': spec_t[0] = tokens[1:]
            elif tokens[0] == '#types': spec_t[1] = tokens[1:]
            elif tokens[0] == '#formats': spec_t[2] = tokens[1:]
            elif tokens[0] == '#separator': separator = tokens[1]

        spec = None
        if len(spec_t[0]):
            if len(spec_t[1]) and len(spec_t[1]) != len(spec_t[0]):
                raise AssertionError('File has bad header spec: ' + str(spec_t[1]))
            if len(spec_t[2]) and len(spec_t[2]) != len(spec_t[0]):
                print(spec_t[2])
                raise AssertionError('File has bad header spec: ' + str(spec_t[2]))
            spec_a = numpy.array(spec_t).transpose()
            spec = [tuple(row) for row in spec_a]

        return spec, separator


    @staticmethod
    def dict2tuple(rows, spec):
        results = []
        fields = [s[0] for s in spec]
        for row in rows:
            results.append(tuple([row.get(f) for f in fields]))

        return results

    
    def __new__(subtype, filename_or_data=None, spec=None, add_spec=None, shape=None,
                index=None, identifier=None, sort=None, reverse=False,
                select_fields=[], skip_fields=[],
                select_field_values=None, skip_field_values=None,
                process_Nones=False, default_missing=False, None_OK=True,
                separator=None, match_pattern=None, logger='warning'):
        """
        filename_or_data:
        name of file containing the data OR
        a list of tuples, each tuple containing values for columnar fields OR
        a numpy recarray. If None, an empty object of given shape is returned.
        Files are expected to be in the following format --
        
        #fields symbol average_volume liquidity days min_close min_volume
        #types |S14 <f8 <f8 <i8 <f8 <f8
        #formats %-14s %9.0f %8.2f %3d %9.4f %9.0f
        AA+                 1190     0.08  31   56.0000       100
        ABL                 6415     0.09  48   12.8750       200

        spec:
        A list of tuples, each tuple representing a field in the rec array.
        The first two elements of each tuple indicate the name of the field and its
        datatype, e.g. ('quantity', int). The third element, the format spec, e.g. '%6d'.
        The fourth, the default value in case of missing data, e.g. -1.
        The latter two elements are optional.
        If the datatype is str, it is converted to 'S<N>' where <N> is the length
        of the longest value in the data.

        add_spec (optional):
        use if you want to allocate space in the array for uninitialized fields.
        Format follows spec (i.e., a list of tuples)

        index (optional):
        Field to use to construct and store a dict with field values (for
        specified field) as keys and the array indices as values.
        Can be a tuple of fields, in which case a multi-layered dict is used.

        identifier (optional):
        Used to store the name of a special column; defaults to index option.

        separator:
        if reading from file, split columns on this regexp.

        match_pattern:
        if reading from file, ignore rows that dont match this regexp

        select_fields:
        load data for these fields only.

        skip_fields:
        skip loading data for these fields.
        
        select_field_values:
        a list of (<field>, [str1, str2, ...]) where
        rows are to be loaded only if the value for <field> is one of
        str1, str2, ...
        
        skip_field_values:
        a list of (<field>, [str1, str2, ...]) where
        rows are to be loaded only if the value for <field> is not one of
        str1, str2, ...

        sort, reverse:
        return recarray sorted by this field, optionally in reverse order.
        
        None_OK:
        Return None if file not found; otherwise raise ValueError.

        process_Nones:
        if input rows contain Nones, stuff with empty_record.

        default_missing:
        Check input rows for missing values in fields that have defaults in given spec,
        and replace the missing values with the defaults.
        
        logger:
        Log error messages using this logging object; if a string, construct one
        with this loglevel.
        """

        if type(logger) == str: logger = logging.Logger('Datab', logger)

        if filename_or_data is None:
            if (shape is None) or (spec is None):
                raise ValueError('Must specify shape and spec if no filename/data')
            data = None
        else:
            type_str = str(type(filename_or_data))
            if 'str' in type_str or 'file' in type_str:
                file_spec, data = \
                           subtype._read_from_file(filename_or_data, spec, separator, match_pattern,
                                                   select_field_values, skip_field_values,
                                                   select_fields, skip_fields, logger=logger)
                if spec is None or select_fields or skip_fields: spec = file_spec
                if spec is None:
                    raise ValueError('No spec and could not determine from %s' %
                                     filename_or_data)
            else:
                data = filename_or_data
            if data is None:
                if None_OK: return None
                raise ValueError('Could not load %s' % filename_or_data)
            if type(data[0]) == dict:
                data = subtype.dict2tuple(data, spec)

        
        if spec is None:
            if hasattr(data, 'spec'):
                spec = data.spec
            elif isinstance(data, numpy.ndarray):
                spec = data.dtype.descr
        full_spec = [list(s) for s in spec]
        
        if add_spec is not None:
            if isinstance(data, numpy.ndarray):
                # Need to do this to allocate more space to each element.
                data = list(data)
            empty_values = []
            for field_spec in add_spec:
                full_spec.append(list(field_spec))
                type_str = str(field_spec[1])
                if len(field_spec) > 3: empty_values.append(field_spec[3])
                elif 'S' in type_str: empty_values.append(Datab.field_defaults['str'][1])
                elif 'i' in type_str: empty_values.append(Datab.field_defaults['int'][1])
                elif 'b' in type_str: empty_values.append(Datab.field_defaults['bool'][1])
                else: empty_values.append(Datab.field_defaults['float'][1])

        empty_record = []
        for i, field_spec in enumerate(full_spec):
            if field_spec[1] == str:
                # Find length N of longest value in data, and set spec to S<N>
                N = max([len(d[i]) for d in data])
                field_spec = [field_spec[0], 'S' + str(N)] + list(field_spec[2:])
                full_spec[i] = tuple(field_spec)
            if len(field_spec) > 3:
                empty_record.append(field_spec[3])
                continue
            
            type_str = str(field_spec[1])
            if 'S' in type_str:
                defaults = list(Datab.field_defaults['str'])
                defaults[0] = '%-'+type_str.split('S')[-1]+'s'
            elif 'i' in type_str: defaults = Datab.field_defaults['int']
            elif 'b' in type_str: defaults = Datab.field_defaults['bool']
            else: defaults = Datab.field_defaults['float']
            empty_record.append(defaults[1])
            if len(field_spec) == 2:
                field_spec.append(defaults[0])
                full_spec[i] = tuple(field_spec)
        empty_record = tuple(empty_record)

        if type(data) == list:
            # can try to speed up the following
            if process_Nones:
                if add_spec is not None:
                    for count, row in enumerate(data):
                        if row is None: data[count] = empty_record
                        else: data[count] = tuple(list(row) + empty_values)
                else:
                    for count, row in enumerate(data):
                        if row is None: data[count] = empty_record
            elif add_spec is not None:                    
                for count, row in enumerate(data):
                    data[count] = tuple(list(row) + empty_values)
            if default_missing:
                defaults = []
                for count, field_spec in enumerate(spec):
                    if len(field_spec) > 3: defaults.append((count, field_spec[3]))
                if len(defaults):
                    for count, row in enumerate(data):
                        for f_count, f_default in defaults:
                            val = row[f_count]
                            if val is None or val == '' or val == '\n':
                                new_row = list(data[count])
                                new_row[f_count] = f_default
                                data[count] = tuple(new_row)

        our_dtype = numpy.dtype([tuple(i[:2]) for i in full_spec])
        empty_record = numpy.array([tuple(empty_record)], dtype=our_dtype)

        if type(data) == list:
            data = numpy.array(data, dtype=our_dtype)
        elif data is None:
            data = numpy.empty(shape, dtype=our_dtype)
            data.fill(empty_record[0])
        if shape == None: shape = numpy.shape(data)

        if sort: data = numpy.sort(data, order=sort)
        if reverse: data = data[::-1]

        # [Deepak: Following comment and __new__ call from numpy example doc]
        
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = numpy.ndarray.__new__(subtype, shape, our_dtype, data,)

        obj.spec = [(s[0], d[1], s[2])
                    for s, d in zip(full_spec, obj.dtype.descr)]
        obj.field_spec = dict([(i[0], i) for i in obj.spec])
        obj.empty_record = empty_record

        obj.identifier = identifier
        if index: obj.build_index(index)
        else: obj.index = None
                            
        obj.logger = logger

        return obj


    def build_index(self, key):
        if numpy.isscalar(key):
            self.index = dict([(ID, count)
                              for count, ID in enumerate(self[key])])
        else: self.index = arrays.index_array(self, key, arg=True)
        if self.identifier is None: self.identifier = key
        

    # Boiler-plate factory method for correct subclassing/CPickling.
    def __reduce__(self):
        object_state = list(numpy.ndarray.__reduce__(self))
        subclass_state = (self.spec, self.field_spec, self.index, self.identifier)
        object_state[2] = (object_state[2], subclass_state)
        return tuple(object_state)
    
    # Boiler-plate factory method for correct subclassing/CPickling.
    def __setstate__(self,state):
        nd_state, own_state = state
        numpy.ndarray.__setstate__(self, nd_state)
        self.spec, self.field_spec, self.index, self.identifier = own_state

    # Boiler-plate factory method for correct subclassing/CPickling.
    def __array_wrap__(self, out_arr, context=None):
        return numpy.ndarray.__array_wrap__(self, out_arr, context)

    # Boiler-plate factory method for correct subclassing/CPickling.
    def __array_finalize__(self, obj):
        # [Deepak: Following comment from numpy example doc]
        
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for our new fields, because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.spec = getattr(obj, 'spec', None)
        self.field_spec = getattr(obj, 'field_spec', None)
        self.index = getattr(obj, 'index', None)        
        self.identifier = getattr(obj, 'identifier', None)        


    def get(self, key, field=None, default=None):
        """
        Locate record identified by key; return value of <field> in that record.
        If <field> is none, return whole record. If record not found, return default.
        """
        
        if self.index is None: raise KeyError('Object has no index.')
        if numpy.isscalar(key):
            if key not in self.index: return default
            if field is None: return self[self.index[key]]
            else: return self[self.index[key]][field]
        else:
            sub_index = self.index
            for sub_key in key:
                if sub_key not in sub_index: return default
                sub_index = sub_index[sub_key]
            if field is None: return self[sub_index]
            else: return self[sub_index][field]
            
        
    def output(self, indices=None, fields=None, exclude=[], rename=[],
               select_values=None, skip_values=None, 
               sort=None, reverse=False, sliced=None, unique=None,
               filename=None, append=False, fh=None, stringify=False,
               print_spec=None, print_header=None, line_space=0, delimiter=' '):
        """
        Print to screen, or file filename or file handle fh, the records
        given by indices (default all).

        File format is as follows --
        
        #fields symbol average_volume liquidity days min_close min_volume
        #types |S14 <f8 <f8 <i8 <f8 <f8
        #formats %-14s %9.0f %8.2f %3d %9.4f %9.0f
        AA+                 1190     0.08  31   56.0000       100
        ABL                 6415     0.09  48   12.8750       200

        fields:
        list of fields, to selectively print, in that order.

        exclude:
        list of fields to exclude from printing.

        rename:
        list of (field, new_name) pairs to rename fields in header of output

        sort:
        sort records by this field before printing.

        reverse:
        reverse records to be printed.

        sliced:
        select records based on this slice tuple.
        
        unique:
        print only the first occurrence for each unique value for this field.

        filename:
        print to this file.

        fh:
        print to this filehandle.

        stringify:
        return output as string.
        
        append:
        append to file if file already exists.
        
        print_spec:
        Whether to print a three-line header or not. Defaults to true if writing
        to a new file, false otherwise.

        print_header:
        Whether to print a one-line header or not. Defaults to true if printing
        to standard output, false otherwise.

        select_values:
        (<field>, set([str1, str2, ...])) where rows are to be printed only if
        the value for <field> is in [str1, str2, ...]
        
        skip_values:
        (<field>, set([str1, str2, ...])) where rows are to be printed only if
        the value for <field> is not in [str1, str2, ...]

        line_space:
        Print an empty line every these many lines.
        """

        if filename:
            mode = 'w'
            if append and os.path.exists(filename):
                mode = 'a'
            elif print_spec is None and print_header is None: print_spec = True
            if filename[-3:] == '.gz':
                if mode == 'a': raise AssertionError('Can not append to .gz file ' + filename)
                fh = gzip.open(filename, 'wb')
            else: fh = open(filename, mode)
        elif fh is None and print_header is None and print_spec is None:
            print_header = True

        if stringify:
            assert filename is None and fh is None, 'Too many output options.'
            fh = io.StringIO()
            
        if fields is None: fields = [s[0] for s in self.spec]
        elif numpy.isscalar(fields): fields = [fields]
        if exclude is not None:
            if numpy.isscalar(exclude): exclude = [exclude]
            for skip_field in exclude:
                if skip_field in fields: fields.remove(skip_field)

        field_formats = [self.field_spec[f][2] for f in fields]
        field_is_str = [self.field_spec[f][2][-1] == 's' for f in fields]

        if print_spec or print_header:
            print(self.header(fields=fields, spec=print_spec, rename=rename, delimiter=delimiter), file=fh)

        if indices is None: records = self
        else: records = self[indices]        
        if sort is not None:
            srt_idx = arrays.argsort(records, order=sort, reverse=reverse)
            records = records[srt_idx]
        elif reverse: records = records[::-1]

        if sliced is not None:
            if numpy.isscalar(sliced): sliced = [sliced]
            records = records[slice(*sliced)]
            
        uniques = {}
        line_count = 0
        for record in records:
            if select_values and record[select_values[0]] not in select_values[1]: continue
            if skip_values and record[skip_values[0]] in skip_values[1]: continue
            if unique:
                if record[unique] in uniques: continue
                uniques[record[unique]] = True
                
            strings = []
            for field, fmt, is_str in zip(fields, field_formats, field_is_str):
                value = record[field]
                if is_str:
                    if value == '': value = '#NA'
                    else: value = value.decode()
                strings.append(fmt % value)
            if line_space and line_count and line_count % line_space == 0: print('')
            print(delimiter.join(strings), file=fh)
            line_count += 1

        if stringify: return fh.getvalue()
        if fh is not None: fh.flush()
        if filename: fh.close()


    def header(self, spec=False, fields=None, exclude=[], rename=[], delimiter=' '):
        """
        Return header, as the string that would be output by the output() method.
        """

        if fields is None: fields = [s[0] for s in self.spec]
        field_name = dict([(field, field) for field in fields])
        if len(rename):
            for field_rename in rename:
                if field_rename[0] not in field_name: continue
                field_name[field_rename[0]] = field_rename[1]

        for skip_field in exclude:
            if skip_field in fields: fields.remove(skip_field)

        if spec:
            lines = ['#fields ' + ' '.join([field_name[f] for f in fields]),
                     '#types ' + ' '.join([self.field_spec[i][1] for i in fields]),
                     '#formats ' + ' '.join([self.field_spec[i][2] for i in fields])]
            return '\n'.join(lines)
        else:
            return delimiter.join([strings.fmt(field_name[f], self.field_spec[f][2])
                             for f in fields])

        
    def stringify(self, record_or_index, fields=None, exclude=[]):
        """
        Return element, passed in as a record or index, as the string that
        would be output by the output() method.
        """

        record = record_or_index
        if type(record_or_index) != numpy.void:
            record = self[record_or_index]
            
        if fields is None: fields = [s[0] for s in self.spec]
        for skip_field in exclude:
            if skip_field in fields: fields.remove(skip_field)
        
        strings = []
        for field in fields:
            field_spec = self.field_spec[field]
            value = record[field]
            if field_spec[1][0] == 'S': value = value.decode()
            strings.append(field_spec[2] % value)
        return ' '.join(strings)
