"""
Keep track of (optionally weighted) descriptive statistics of incoming data,
handling nans. 

Copyright 2013 Deepak Subburam

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""


import sys
import numpy
from . import datab as db
from . import arrays, logging, strings


class Sparse(object):
    """
    Keep track of summary statistics, but do not store the values.

    SYNOPSIS
    
    s = Sparse(weighted=True, logger='debug')
    s.update(data, weights=w)
    stats = s.compute()
    OR
    s = Sparse(data, weights=w)
    stats = s.compute()

    OPTIONS

    weighted:
    Whether weights accompany datapoints. Defaults to False.

    logger (default: 'info'):
    If this is a logger object, use that for debug/error messages. If this
    is a string, initialize a logger object with the corresponding log level.

    store_last:
    Store values processed by last call to update().
    
    ATTRIBUTES

    statistics:
    statistics computed so far. Needs call to obj.compute() to refresh.

    last_update:
    values processed by last call to update().

    NOTES

    For weighted objects, statistics{d_count} contains sum(w)^2 / sum(w^2),
    and is a measure of the diversity of data points.
    """

    def __init__(self, *data, **opts):
        if 'logger' in opts:
            self.logger = opts['logger']
        else: self.logger = 'info'
        if type(self.logger) == str:
            self.logger = logging.Logger(self.__class__.__name__, self.logger)
        self.count = 0
        self.size = 0
        self.sum_xw = 0.0
        self.sum_xxw = 0.0
        self.sum_w = 0.0
        self.sum_ww = 0.0

        self.weighted = False
        if 'weighted' in opts: self.weighted = opts['weighted']
        if 'weights' in opts and opts['weights'] is not None:
            self.weighted = True
        if 'negative_weights' in opts and opts['negative_weights'] is not None:
            self.weighted = True
        if 'weights' not in opts: opts['weights'] = None
        if 'negative_weights' not in opts: opts['negative_weights'] = None

        if 'store_last' in opts and opts['store_last']: self.last_update = True
        else: self.last_update = None
        
        self.statistics = {'mean': None,
                           'mean_square': None,
                           'sum_square': None,
                           'variance': None,
                           'std_dev': None,
                           't-stat': None,
                           'std_err': None,
                           'count': 0,
                           'size': 0,
                           'sum': None,
                           'min': None,
                           'max': None,
                           }
        if self.weighted:
            self.statistics['mean_wt'] = None
            self.statistics['d_count'] = 0.0

        if len(data) > 1:
            raise TypeError('Too many arguments to constructor.')
        elif len(data):
            if 'IDs' not in opts: opts['IDs'] = None
            if self.weighted is None:
                self.weighted = ('weights' in opts) and (opts['weights'] is not None)
            self.update(data[0], weights=opts['weights'],
                        negative_weights=opts['negative_weights'], IDs=opts['IDs'])
                

    def update(self, values, weights=None, IDs=None, negative_weights=None):
        """
        values, [negative_]weights and IDs may either be all arrays or all scalars.

        If negative_weights are specified, values get multiplied by the sign of
        the corresponding negative_weight, and weights get set to
        abs(negative_weights).
        
        datapoints with either the value or the weight being nan are ignored,
        as are datapoints with weight <= 0. Though these datapoints affect
        the 'size' statistic (but not the 'count' statistic).

        values, weights and IDs are returned (useful to get back defaults/masked
        versions of the inputs).
        """

        values = arrays.nice_array(values, logger=self.logger)
        mask = values.mask.copy()

        if self.weighted:
            if negative_weights is not None:
                if weights is not None: raise AssertionError('Can not specify both weights and negative weights')
                weights = abs(negative_weights)
                values = values.copy()*numpy.sign(negative_weights)
            elif weights is None:
                raise AssertionError('Weighted statistics object received no weights in update.')
            weights = arrays.nice_array(weights, shape=values.shape, logger=self.logger)
            mask |= weights.mask
            # Following contortion to avoid bogus
            #    "RuntimeWarning: Invalid value encountered in less_equal"
            mask[~mask] = (weights[~mask] <= 0)
            fweights = weights.flatten()
        else:
            if weights is not None:
                raise AssertionError('Unweighted statistics object received weights in update.')
            fweights = numpy.ma.ones(values.size, dtype=float)            
            
        fweights.mask = mask.flatten()
        fvalues = values.flatten()
        fvalues.mask = fweights.mask

        if IDs is None: IDs = numpy.array(list(range(fvalues.size)), dtype=int) + self.size
        elif not isinstance(IDs, numpy.ndarray): IDs = numpy.array(IDs)

        self.size += fvalues.size
        count = fvalues.count()
        if count == 0:
            if self.last_update is not None: self.last_update = ([], [], [])
            return 

        min_index = numpy.ma.argmin(fvalues)
        max_index = numpy.ma.argmax(fvalues)
        if self.count == 0:
            self.statistics['min'] = (fvalues[min_index], IDs.flat[min_index])
            self.statistics['max'] = (fvalues[max_index], IDs.flat[max_index])
        else:
            if fvalues[min_index] < self.statistics['min'][0]:
                self.statistics['min'] = (fvalues[min_index], IDs.flat[min_index])
            if fvalues[max_index] > self.statistics['max'][0]:
                self.statistics['max'] = (fvalues[max_index], IDs.flat[max_index])

        self.count += count
        self.sum_xw += numpy.ma.sum(fvalues * fweights)
        self.sum_xxw += numpy.ma.sum(fvalues * fvalues * fweights)
        self.sum_w += numpy.ma.sum(fweights)
        self.sum_ww += numpy.ma.sum(fweights * fweights)

        if self.last_update is not None:
            self.last_update = (fvalues, fweights, IDs.flat)


    def compute(self):

        statistics = self.statistics
        statistics['count'] = self.count
        statistics['size'] = self.size
        if self.count == 0: return self.statistics
        statistics['mean'] = self.sum_xw / self.sum_w
        statistics['sum'] = statistics['mean'] * self.count
        statistics['mean_square'] = self.sum_xxw  / self.sum_w
        statistics['sum_square'] = statistics['mean_square'] * self.count
        d_count = self.count
        if self.weighted:
            self.statistics['mean_wt'] = self.sum_w / self.count
            d_count = self.sum_w ** 2 / self.sum_ww
            statistics['d_count'] = d_count
        if self.count == 1: return self.statistics
        
        statistics['variance'] = \
            statistics['mean_square'] - statistics['mean'] ** 2
        if statistics['variance'] < 0.0: return self.statistics
        statistics['std_dev'] = numpy.sqrt(statistics['variance'])
        statistics['std_err'] = statistics['std_dev'] / numpy.sqrt(d_count)

        if statistics['std_err'] <= 0.0: return self.statistics        
        statistics['t-stat'] = statistics['mean'] / statistics['std_err']
        return self.statistics


    @classmethod
    def stats(cls, data, weights=None, axis=None, step=1, sliced=None, select=None,
              overlay=None, split=None, buckets=None, group=None,
              labels=None, label_index=None, label_all=None, label_other='Other', 
              negative_weights=None, IDs=None,
              datab=None, name=None, formats=None, **opts):
        """
        Calls Class(data).compute(), handling complexities in the form of data.

        data can be two dimensional, and axis can be 0 or 1. In this case,
        a list of statistics-records is returned, in Datab form (unless datab=False).

        overlay:
        run stats only for records selected by this mask.

        split:
        run stats for all records, records selected by this mask, and for
        the others, returning a 3-tuple of results. Does not work with axis option.

        buckets:
        run stats for all records, and for records selected by each of the masks
        in this list of masks. Does not work with axis option.

        group:
        bucket stats by values in this field.

        sliced:
        run stats for records selected by this slice.

        select:
        run stats for records selected by this boolean mask.

        step:
        When axis option is specified, clump these many rows together for each row
        stat to be computed. This can optionally be a list of steps, in which case
        each clump can have variable number of rows.

        label_all:
        Relevant only when axis or split/buckets option present. If not None,
        compute stats over entire dataset, in addition to for each index of the
        axis or split/buckets, and place results in an entry of output with this
        label.

        label_other:
        Relevant only when buckets option present. If not None, compute stats
        over part of dataset not in any bucket, in addition to for each bucket,
        and place results in an entry of output with this label.

        labels:
        list to use to add labels to each entry of output. Relevant only when
        there are multiple lines of output.

        label_index:
        like labels, except use label_index[::step].

        name:
        in the header, label the key column with this string.

        formats:
        Default formatting e.g. %7.2f for numeric fields in Datab spec.
        """

        if group is not None:
            if buckets is not None and split is not None:
                raise AssertionError('group, buckets and split options not supported together.')
            label_other = None
            labels, buckets = [], []
            for group_name in numpy.unique(group):
                labels.append(group_name)
                buckets.append(group == group_name)
            if name is None: name = 'group'

        if split is not None:
            if buckets is not None:
                raise AssertionError('group, buckets and split options not supported together.')
            buckets = [split]
            if labels is None:
                labels = ['True']
                label_other = 'False'
            else:
                label_other = labels[1]
                labels = [labels[0]]
            if name is None: name = 'condn'
        elif buckets is not None:
            if labels is None:
                labels = [str(d + 1) for d in range(len(buckets))]
            if name is None: name = 'bucket'

        data = arrays.nice_array(data)
        if weights is not None: weights = arrays.nice_array(weights)
        if negative_weights is not None:
            if weights is not None: raise AssertionError('Can not specify both weights and negative weights')
            weights = abs(negative_weights)
            data = data.copy()*numpy.sign(negative_weights)

        if axis is None and numpy.isscalar(step) and step == 1:
            data, weights, IDs = \
                arrays.select([data, weights, IDs],
                              sliced=sliced, overlay=overlay, select=select)
            if buckets is None:
                results = cls(data, weights=weights, IDs=IDs, **opts).compute()
                if datab is True: return Datab([results], formats=formats)
                else: return results

            if label_all:
                all_labels = [label_all]
                results = [cls.stats(data, weights=weights, IDs=IDs, formats=formats,
                                     **opts)]
            else: all_labels, results = [], []

            if label_other: other = numpy.ones(numpy.shape(data), dtype=bool)
            buckets = arrays.select(buckets,
                                    sliced=sliced, overlay=overlay, select=select)
            all_labels.extend(labels)

            for b in buckets:
                results.append(cls.stats(data, weights=weights, IDs=IDs, overlay=b,
                                         formats=formats, **opts))
                if label_other: other[b] = False
            if label_other:
                all_labels.append(label_other)
                results.append(cls.stats(data, weights=weights, IDs=IDs,
                                         overlay=other, formats=formats, **opts))

            if datab is False: return results
            else: return Datab(results, labels=all_labels, name=name, formats=formats)

        if buckets is not None:
            raise AssertionError('split/buckets option not supported with axis/step option.')

        data, weights, IDs = arrays.select([data, weights, IDs],
                                           sliced=sliced, overlay=overlay, select=select)

        if cls != Multivariate:
            if axis is not None and (axis > 1 or axis < 0 or data.ndim != 2):
                raise ValueError('Got unsupported axis option value that is ' +
                                 'not 0 or 1; or data is not two-dimensional.')
            if axis == 0:
                data = data.transpose()
                if overlay is not None: overlay = overlay.transpose()
                if IDs is not None: IDs = IDs.transpose()
                if weights is not None: weights = weights.transpose()
        elif axis is not None and axis != 0:
            raise ValueError('Axis option value 0 is the only one supported for Multivariate stats.')

        if weights is not None and weights.ndim == 1 and data.ndim == 2:
            if len(weights) != numpy.shape(data)[1]:
                raise ValueError('shape mismatch: 1D weights cannot be broadcast to shape of values')
            sys.stderr.write('stats.stats: Broadcasting 1D weights for 2D values.\n')
            weights = arrays.extend(weights, numpy.shape(data)[0]).T

        if label_all is not None:
            results = [cls(data, weights=weights, IDs=IDs, **opts).compute()]
            all_labels = [label_all]
        else:
            results = []
            all_labels = []

        start_idx = 0
        count = 0
        while start_idx < len(data):
            if numpy.isscalar(step): end_idx = start_idx + step
            else: end_idx = start_idx + step[min(count, len(step)-1)]
            
            row_data, row_weights, row_IDs = \
                arrays.select([data, weights, IDs], sliced=(start_idx, end_idx, 1))

            results.append(cls.stats(row_data, weights=row_weights, IDs=row_IDs))

            if labels is not None and len(labels): all_labels.append(labels[count])
            elif label_index is not None: all_labels.append(label_index[start_idx] + '-')
            else: all_labels.append(str(start_idx) + '-')

            start_idx = end_idx
            count += 1

        if datab is False: return results
        else: return Datab(results, labels=all_labels, name=name or 'key', formats=formats)


    @classmethod
    def summary(cls, data, weights=None, include=None, exclude=None, fields=None,
                all=False, filename=None, line_space=0, stringify=False, **opts):
        """
        Convenience wrapper that, roughly speaking, calls cls.stats(data)
        and then prints results in nice tabulated form, using stats.Datab.

        See cls.stats() for documentation on supported options.
        """
        stats_data = cls.stats(data, weights=weights, datab=True, **opts)
        if len(stats_data) == 1 and 'name' not in opts:
            #Get rid of redundant key column.
            if exclude is None: exclude = ['key']
            else: exclude.append('key')
        return stats_data.output(include=include, exclude=exclude, fields=fields, all=all,
                                 filename=filename, line_space=line_space, stringify=stringify)

    
    @classmethod
    def loop_summary(cls, data, weights=None, include=None, exclude=None, fields=None,
                     labels=None, name='var', formats=None, all=False,
                     stringify=False, **opts):
        """
        Calls summary() in a loop for multiple responses/predictors.
        """

        if type(data) != tuple:
            raise ValueError('Data needs to be tuple so looping can happen.')

        output = []
        all_labels = []
        for count, layer in enumerate(data):
            w = weights[count] if type(weights) == tuple else weights
            output.append(cls.stats(layer, weights=w, **opts))
            if labels is not None and len(labels): all_labels.append(labels[count])
            else: all_labels.append(count)

        output = Datab(output, labels=all_labels, name=name, formats=formats)
        return output.output(include=include, exclude=exclude, fields=fields, all=all,
                             stringify=stringify)
        

class Full(Sparse):
    """
    Store the values, so as to produce median and percentile values.
    """

    default_percentiles = [[0.99, '99th%le'],
                           [0.95, '95th%le'],
                           [0.90, '90th%le'],
                           [0.75, '75th%le'],
                           [0.50, 'median'],
                           [0.25, '25th%le'],
                           [0.10, '10th%le'],
                           [0.05, '5th%le'],
                           [0.01, '1st%le']]

    def __init__(self, *args, **opts):
        self.data = None
        Sparse.__init__(self, store_last=True, *args, **opts)
        for percentile in Full.default_percentiles:
            self.statistics[percentile[1]] = None
        self.statistics['mad'] = None


    def update(self, values, weights=None, IDs=None, negative_weights=None):
        Sparse.update(self, values, weights=weights, IDs=IDs,
                      negative_weights=negative_weights)
        if numpy.isscalar(self.last_update): return
        values, weights, IDs = self.last_update
        if not len(values): return
        mask = values.mask | weights.mask 
        # Following contortion to avoid bogus
        #    "RuntimeWarning: Invalid value encountered in less_equal"
        mask[~mask] = (weights[~mask] <= 0)

        valid_values = values[~mask]
        if not len(valid_values): return

        indices = numpy.ma.argsort(valid_values)
        update_data = \
            numpy.array(list(zip(valid_values[indices], weights[~mask][indices],
                            IDs[~mask][indices])),
                        dtype=numpy.dtype([('value', values[0].dtype),
                                           ('weight', weights[0].dtype),
                                           ('ID', IDs[0].dtype)]))
        if self.data is None:
            self.data = update_data
        else:
            insert_indices = self.data['value'].searchsorted(update_data['value'])
            self.data = numpy.insert(self.data, insert_indices, update_data)


    def compute_percentiles(self, percentiles=None):

        if self.data is None: return
        if percentiles is None: percentiles = Full.default_percentiles
        data = self.data
        cumulative_weight = numpy.cumsum(data['weight'])
        sum_weight = cumulative_weight[-1]
        mean_weight = sum_weight / len(data)
        if sum_weight <= 0.0: return
        
        for entry in percentiles:
            wanted_center = sum_weight * entry[0] + mean_weight / 2
            right_index = numpy.searchsorted(cumulative_weight, wanted_center)
            if right_index == 0:
                self.statistics[entry[1]] = (data[0][0], data[0][2])
            elif right_index == len(data):
                self.statistics[entry[1]] = (data[-1][0], data[-1][2])
            else:
                left_index = right_index - 1
                left_distance = wanted_center - cumulative_weight[left_index]
                right_distance = cumulative_weight[right_index] - wanted_center
                value = (right_distance * data[left_index][0] +
                         left_distance * data[right_index][0]) / \
                         (left_distance + right_distance)
                if right_distance < left_distance: ID_index = right_index
                else: ID_index = left_index
                self.statistics[entry[1]] = (value, data[ID_index][2])


    def compute(self):
        if self.data is None or self.sum_w <= 0.0: return self.statistics
        statistics = Sparse.compute(self)

        data = self.data
        if len(data) == 1:
            self.statistics['median'] = [data['value'], data['ID']]
            return self.statistics

        self.compute_percentiles()

        deviations = Full(weighted=True)
        deviations.update(numpy.abs(data['value'] - self.statistics['median'][0]),
                          data['weight'])
        deviations.compute_percentiles(percentiles=[[0.5, 'mad']])
        self.statistics['mad'] = deviations.statistics['mad']
        
        return self.statistics


class Multivariate(Sparse):
    """
    Each value is an constant-sized array of variables; covariances between the
    variables are computed.

    SYNOPSIS
    
    s = Multivariate(nvars=2, weighted=True, logger='debug', names=['pred', 'resp']) 
    s.update(data, weights=w)
    stats = s.compute()
    OR
    s = Multivariate(data, weights=w)
    stats = s.compute()

    OPTIONS

    nvars: Number of variables in each datapoint.
    weighted: Whether weights accompany datapoints. Defaults to False
    store_last: Store values processed by last call to update().
    store_all: Store values processed by all calls to update().

    ATTRIBUTES

    statistics:
    statistics computed so far. Needs call to obj.compute() to refresh.

    last_update, all_update:
    values processed by last/all call(s) to update(), if store_last/all specified
    during object construction.
    
    Sparse[i]:
    Sparse statistics object for each of the variables.

    NOTES

    statistics{multiple_ij} containts mean(x_i * x_j) / mean(x_j * x_j).
    """
    
    def __init__(self, *data, **opts):

        if 'logger' in opts:
            self.logger = opts['logger']
        else: self.logger = 'info'
        if type(self.logger) == str:
            self.logger = logging.Logger(self.__class__.__name__, self.logger)

        if len(data) > 1:
            raise SyntaxError('Too many arguments to constructor.')
        elif len(data):
            data = data[0]
            if not isinstance(data, numpy.ndarray): data = numpy.array(data)
            if data.ndim == 1: data = data[:, numpy.newaxis]
            if 'nvars' in opts and opts['nvars'] != numpy.shape(data)[-1]:
                raise ValueError('Number of columns in data incompatible with nvars option.')
            self.nvars = numpy.shape(data)[-1]
        elif 'nvars' in opts:
            if opts['nvars'] < 1: raise ValueError('Number of nvars must be at least 1.')
            else: self.nvars = opts['nvars']
        else: 
            raise ValueError('Need to specify nvars option.')

        if 'names' in opts:
            self.names = opts['names']
            if len(self.names) != self.nvars:
                raise AssertionError('Number of names supplied should match nvars.')
            del opts['names']
        else: self.names = ['var'+str(i) for i in range(self.nvars)]
        
        self.size = 0
        self.count = 0
        self.sum_w = 0.0
        self.sum_ww = 0.0
        self.count_ij = numpy.zeros((self.nvars, self.nvars), dtype=int)
        self.sum_ijw = numpy.zeros((self.nvars, self.nvars))
        self.sum_wij = numpy.zeros((self.nvars, self.nvars))
        self.sum_wwij = numpy.zeros((self.nvars, self.nvars))
        
        self.weighted = False
        if 'weighted' in opts:
            self.weighted = opts['weighted']
        if 'weights' in opts and opts['weights'] is not None:
            self.weighted = True
        if 'negative_weights' in opts and opts['negative_weights'] is not None:
            self.weighted = True
        if 'weights' not in opts: opts['weights'] = None
        if 'negative_weights' not in opts: opts['negative_weights'] = None
        
        if 'store_last' in opts and opts['store_last']:
            self.last_update = True
        else: self.last_update = None

        if 'store_all' in opts and opts['store_all']:
            self.all_update = []
        else: self.all_update = None

        self.statistics = {'mean_ij': numpy.zeros((self.nvars, self.nvars)) + numpy.nan,
                           'variance_ij': numpy.zeros((self.nvars, self.nvars)) + numpy.nan,
                           'std_dev_ij': numpy.zeros((self.nvars, self.nvars)) + numpy.nan,
                           'count': 0,
                           'size': 0,
                           'sum_ij': numpy.zeros((self.nvars, self.nvars)) + numpy.nan,
                           'correlation_ij': numpy.zeros((self.nvars, self.nvars)) + numpy.nan,
                           'multiple_ij': numpy.zeros((self.nvars, self.nvars)) + numpy.nan,
                           'count_ij': numpy.zeros((self.nvars, self.nvars), dtype=int),
                           'nvars': self.nvars,
                           }
        self.Sparse = [Sparse(weighted=self.weighted) for i in range(self.nvars)]

        if self.weighted:
            self.statistics['mean_wt'] = None
            self.statistics['d_count'] = 0.0
            self.statistics['mean_wt_ij'] = numpy.zeros((self.nvars, self.nvars)) + numpy.nan
            self.statistics['d_count_ij'] = numpy.zeros((self.nvars, self.nvars))

        if len(data): self.update(data, weights=opts['weights'],
                                  negative_weights=opts['negative_weights'])

    
    def update(self, values, weights=None, negative_weights=None):
        """
        Can update one datapoint at a time (in which case values is an array
        and weights must be a scalar), or a set (in which case values are
        rows of a 2D array, and weights is a 1D array).

        If negative_weights are specified, values get multiplied by the sign of
        the corresponding negative_weight, and weights get set to
        abs(negative_weights).
        """

        values = arrays.nice_array(values, logger=self.logger,
                                   shape=(numpy.size(values) / self.nvars, self.nvars))

        if self.weighted:
            if negative_weights is not None:
                if weights is not None: raise AssertionError('Can not specify both weights and negative weights')
                negative_weights = arrays.nice_array(negative_weights, shape=len(values),
                                                     logger=self.logger)
                weights = abs(negative_weights)
                values = values.copy()*numpy.sign(negative_weights)[:, numpy.newaxis]
            elif weights is None:
                raise AssertionError('Weighted statistics object received no weights in update.')
            else:
                weights = arrays.nice_array(weights, shape=len(values), logger=self.logger)
        else:
            if weights is not None:
                raise AssertionError('Unweighted statistics object received weights in update.')
            weights = numpy.ma.ones(len(values))
            
        for i in range(self.nvars):
            if self.weighted: self.Sparse[i].update(values[:, i], weights)
            else: self.Sparse[i].update(values[:, i])

            for j in range(i):
                valid = ~(values.mask[:, i] | values.mask[:, j] | weights.mask)
                self.count_ij[i, j] += numpy.sum(valid)
                self.sum_ijw[i, j] += numpy.sum(values[:, i] * values[:, j] * weights)
                self.sum_wij[i, j] += numpy.sum(weights[valid])
                self.sum_wwij[i, j] += numpy.sum(weights[valid] ** 2)
                
        self.size += len(weights)
        valid = numpy.any(~values.mask, axis=1) & (~weights.mask)
        self.count += numpy.sum(valid)
        self.sum_w += numpy.sum(weights[valid])
        self.sum_ww += numpy.sum(weights[valid] ** 2)

        if self.last_update is not None: self.last_update = (values, weights, valid)
        if self.all_update is not None: self.all_update.append((values, weights, valid))
    

    def compute(self):

        statistics = self.statistics
        statistics['count'] = self.count
        statistics['size'] = self.size
        if self.count == 0: return statistics

        sparse_stats = [s.compute() for s in self.Sparse]

        if self.weighted:
            statistics['mean_wt'] = self.sum_w / self.count
            statistics['d_count'] = self.sum_w ** 2 / self.sum_ww
        if self.count == 1: return statistics

        stat_fields = ['count', 'variance', 'std_dev']
        if self.weighted: stat_fields.extend(['mean_wt', 'd_count'])

        for i in range(self.nvars):

            for stat in stat_fields:
                statistics[stat + '_ij'][i, i] = sparse_stats[i][stat]

            statistics['correlation_ij'][i, i] = 1.0
            statistics['multiple_ij'][i, i] = 1.0
            statistics['mean_ij'][i, i] = sparse_stats[i]['mean_square']
            statistics['sum_ij'][i, i] = sparse_stats[i]['sum_square']
            
            for j in range(i):
                statistics['count_ij'][i, j] = self.count_ij[i, j]
                if self.count_ij[i, j] == 0: continue
                
                statistics['mean_ij'][i, j] = self.sum_ijw[i, j] / self.sum_wij[i, j]
                statistics['sum_ij'][i, j] = \
                    statistics['mean_ij'][i, j] * statistics['count_ij'][i, j]
                statistics['variance_ij'][i, j] = statistics['mean_ij'][i, j] - \
                    sparse_stats[i]['mean'] * sparse_stats[j]['mean']
                statistics['std_dev_ij'][i, j] = \
                    numpy.sqrt(abs(statistics['variance_ij'][i, j])) * \
                    numpy.sign(statistics['variance_ij'][i, j])
                
                statistics['correlation_ij'][i, j] =  \
                    statistics['variance_ij'][i, j] / \
                    numpy.sqrt(statistics['variance_ij'][i, i] *
                               statistics['variance_ij'][j, j])
                
                if statistics['mean_ij'][j, j] > 0:
                    statistics['multiple_ij'][i, j] = \
                    statistics['mean_ij'][i, j] / statistics['mean_ij'][j, j]
                if statistics['mean_ij'][i, i] > 0:
                    statistics['multiple_ij'][j, i] = \
                    statistics['mean_ij'][i, j] / statistics['mean_ij'][i, i]

                if self.weighted:
                    statistics['mean_wt_ij'][i, j] = \
                        self.sum_wij[i, j] / self.count_ij[i, j]
                    statistics['d_count_ij'][i, j] = \
                        self.sum_wij[i, j] ** 2 / self.sum_wwij[i, j]

                for stat in stat_fields + ['correlation', 'mean', 'sum']:
                    statistics[stat + '_ij'][j, i] = statistics[stat + '_ij'][i, j]
                
        return self.statistics


    def print_ij(self, statistic, names=None, format=None):
        """Print table of given cross-statistic."""

        values = self.statistics[statistic]
        if names is None: names = self.names
        if format is None:
            stat = statistic.replace('_ij', '')
            if stat in Datab.spec_index: format = Datab.spec[Datab.spec_index[stat]][2]
            else: format = '%9.4f'

        print(' '.join([strings.fmt(string, format) for string in [' '] + names]))
        for i in range(self.nvars):
            print(strings.fmt(names[i], format), end=' ')
            for j in range(self.nvars):
                print(format % values[i, j], end=' ')
            print('')

        
    def datab_ij(self, statistic, names=None, format=None):
        """Return datab object containing table of given cross-statistic."""

        if names is None: names = self.names
        if format is None:
            stat = statistic.replace('_ij', '')
            if stat in Datab.spec_index: format = Datab.spec[Datab.spec_index[stat]][2]
            else: format = '%9.4f'
        spec = [('factor', 'S6', '%-6s')] + [(n, float, format) for n in names]

        values = self.statistics[statistic]
        entries = [tuple([names[i]] + list(row)) for i, row in enumerate(values)]
        return db.Datab(entries, spec)


class Datab(db.Datab):
    """
    Store and print computed statistics-records in nice tabular form.

    Datab.spec:
    preferred order of statistics, and typical formatting.

    Datab.spec_index:
    dict going from statistic to index in Datab.spec.

    Datab.default_output_fields:
    shortlisted statistics to print by default.

    See datab.Datab for other class/object fields and definitions.
    """
    
    # entries in spec should be ordered as they are to be columned.
    spec = (('size', int, '%7d'), ('count', int, '%7d'),
            ('d_count', float, '%8.1f'), ('mean_wt', float, '%7.3f'),
            ('sum', float, '%8.1f'), ('mean', float, '%8.4f'),
            ('std_dev', float, '%8.4f'), ('t-stat', float, '%7.3f'),
            ('median', float, '%8.4f'), ('mad', float, '%8.4f'),
            ('25th%le', float, '%8.4f'), ('75th%le', float, '%8.4f'),
            ('correlation_ij', float, '%7.4f'), ('multiple_ij', float, '%10.6f'),
            ('std_err', float, '%8.6f'), ('variance', float, '%8.6f'),
            ('min', float, '%8.3f'), ('max', float, '%8.3f'),
            )
    spec_index = dict([(f[0], i) for i, f in enumerate(spec)])
    matrix_stat = {'correlation_ij': {'name': 'corr',
                                      'symmetric': True},
                   'multiple_ij': {'name': 'coeff',
                                   'symmetric': False}}

    # the below should be in order the fields appear in spec.
    default_output_fields = ('count', 'd_count', 'mean_wt',
                             'mean', 'std_dev', 'correlation_ij', 't-stat',
                             'median', 'mad', 'min', 'max')


    def __new__(subtype, results, labels=[], name='key', formats=None, **datab_args):
        """
        results argument is a list of statistics-records. If corresponding list 
        of strings, labels, is given, a column called <name> is created
        and stored in the Datab object, which will also be constructed
        with index=<name> option.
        """

        first_result = None
        for r in results:
            if r is not None:
                first_result = r
                break
        if not first_result: return None

        if not len(labels):
            labels = [str(d) for d in range(len(results))]
        else: labels = [str(l) for l in labels]
        
        indices = []
        for stat in list(first_result.keys()):
            if stat not in Datab.spec_index: continue
            indices.append(Datab.spec_index[stat])
        statistics = numpy.array(Datab.spec)[numpy.sort(indices)]

        stats_data, none_indices = [], []
        key_len = len(name)
        for count, result in enumerate(results):
            if result is None:
                stats_data.append(None)
                none_indices.append(count)
                continue
            row_stats = [labels[count]]
            key_len = max(key_len, len(row_stats[-1]))

            for stat in [s[0] for s in statistics]:
                if stat not in Datab.matrix_stat:
                    if result[stat] is None or numpy.ndim(result[stat]) == 0:
                        row_stats.append(result[stat])
                    # for (min|max, arg) stats, just store the min/max value
                    else: row_stats.append(result[stat][0])
                else:
                    for i in range(first_result['nvars']):
                        for j in range(i):
                            row_stats.append(result[stat][i, j])
                            if not Datab.matrix_stat[stat]['symmetric']:
                                row_stats.append(result[stat][j, i])
            stats_data.append(tuple(row_stats))

        my_spec = [(name, 'S'+str(key_len), '%-'+str(key_len)+'s')]            
        default_fields = []
        for spec in statistics:
            if spec[0] not in Datab.matrix_stat:
                my_spec.append(spec)
                if spec[0] in Datab.default_output_fields:
                    default_fields.append(spec[0])
            else:
                nm = Datab.matrix_stat[spec[0]]['name']
                for i in range(first_result['nvars']):
                    for j in range(i):
                        my_spec.append(('_'.join([nm, str(i), str(j)]),
                                        spec[1], spec[2]))
                        if spec[0] in Datab.default_output_fields:
                            default_fields.append('_'.join([nm, str(i), str(j)]))
                        if not Datab.matrix_stat[spec[0]]['symmetric']:
                            my_spec.append(('_'.join([nm, str(j), str(i)]),
                                            spec[1], spec[2]))
                            if spec[0] in Datab.default_output_fields:
                                default_fields.append('_'.join([nm, str(j), str(i)]))
                        
        if formats is not None:
            for s in my_spec:
                if s[1] == float and s[0] != 'd_count': s[2] = formats

        obj = super(Datab, subtype).__new__(subtype, stats_data, my_spec, index=name,
                                            **datab_args)
        obj.default_output_fields = default_fields

        for index in none_indices:
            obj[name][index] = labels[index]
        
        return obj
                

    def output(self, include=None, exclude=None, all=False, **kwargs):
        """
        Print statistics-records nicely. If fields=[field1, field2, ...] not
        specified, use Datab.default_output_fields.

        all:
        print all statistics fields.

        include:
        include this statistic field(s) on top of what's going to be printed.

        exclude:
        exclude this statistic field(s) from what's going to be printed.
        """

        if numpy.isscalar(exclude): exclude = [exclude]
        if numpy.isscalar(include): include = [include]

        if 'fields' in kwargs and kwargs['fields'] is not None:
            super(Datab, self).output(**kwargs)
            return
        
        if self.identifier is not None and \
               (exclude is None or self.identifier not in exclude):
                kwargs['fields'] = [self.identifier]
        else: kwargs['fields'] = []

        if all: fields = [s[0] for s in self.spec]
        else:
            fields = self.default_output_fields
            if include is not None:
                spec_index = dict([(f[0], i) for i, f in enumerate(self.spec)])
                field_indices = [spec_index[f] for f in fields]
                insert_indices = numpy.searchsorted(field_indices,
                                                    [spec_index[f] for f in include])
                fields = numpy.array(fields, dtype='S'+str(max([len(s[0])
                                                                for s in self.spec])))
                fields = numpy.insert(fields, insert_indices, include)

        for field in fields:
            if exclude is None or field not in exclude: kwargs['fields'].append(field)

        return super(Datab, self).output(**kwargs)
        


def summary(*args, **kwargs):
    return Full.summary(*args, **kwargs)

    
def loop_summary(*args, **kwargs):
    return Full.loop_summary(*args, **kwargs)

    
def stats(*args, **kwargs):
    return Full.stats(*args, **kwargs)


def bucketer(*data_field_splits, **kwargs):
    """
    Helper function for constructing bucket options. Pass output of this function
    to stats.summary(), etc like this --
    stats.summary(data['1_252'], data['weight'],
                  **stats.bucketer([pred1, 'var1', (-.1, 0, .1)], [pred2, 'var2', 0]))

    *data_field_splits:
    List of triples, (data, field_name, [split_val1, ...]).
    If the third element here is a scalar, it gets cast as a list with one element.
    If the third element here is None, bucketing is done evaluating data as True/False.
    If the first element here is a tuple, it is deemed to be a set of True/False arrays,
    and the third element is assumed to be a list of labels (if None, label with integers).

    **kwargs:
    Can handle the following options: label_all, label_other, formats.
    """
    
    def _recurse_buckets(overlays_labels, overlays, labels):
        if not overlays_labels: return overlays, labels
        
        new_overs, new_labels = [], []
        for layer_over, layer_label in zip(*overlays_labels.pop(-1)):
            for existing_over, existing_label in zip(overlays, labels):
                new_overs.append(existing_over & layer_over)
                new_labels.append(layer_label + '|' + existing_label)
                
        return _recurse_buckets(overlays_labels, new_overs, new_labels)

    
    label_all = kwargs.get('label_all', 'All')
    label_other = kwargs.get('label_other', 'Other')

    formats = kwargs['formats'] if 'formats' in kwargs else '%6.1f'
    fmt = formats + '_%-' + formats[1:]
    str_fmt = '%' + str(strings.fmt_length(formats)) + 's ' + \
              strings.fmt(' ', formats)
    name_fmt = '%-' + str(strings.fmt_length(formats) * 2 + 1) + 's'
               
    overlays_labels = []
    name = ''
    for d_f_s in data_field_splits:
        data, field, splits = d_f_s
        if numpy.isscalar(splits): splits = [splits]
        if name: name += '|'
        name += name_fmt % field
        
        overlays = [numpy.ones(numpy.shape(data[0] if type(data) == tuple else data),
                               dtype=bool)] if label_all else []
        labels = [str_fmt % label_all] if label_all else []

        if type(data) == tuple:
            # multiple boolean bucketing
            for i in range(len(data)):
                overlays.append(data[i])
                labels.append(name_fmt % (i if splits is None else splits[i]))
            overlays_labels.append((overlays, labels))
            continue
        
            
        if splits is None:
            # boolean bucketing
            overlays.append(data)
            labels.append(str_fmt % 'True')
            overlays.append(~data)
            labels.append(str_fmt % 'False')
            overlays_labels.append((overlays, labels))
            continue
        
        other_overlay = numpy.ones(numpy.shape(data), dtype=bool) if label_other else None
        for count, value in enumerate(splits):
            if count == 0:
                overlays.append(data < value)
                if label_other: other_overlay &= ~overlays[-1]
                labels.append(fmt % (-numpy.inf, value))
            if count > 0 and len(splits) > 1:
                overlays.append((data < value) & (data >= splits[count - 1]))
                if label_other: other_overlay &= ~overlays[-1]
                labels.append(fmt % (splits[count - 1], value))
            if count == len(splits) - 1:
                overlays.append(data >= value)
                if label_other: other_overlay &= ~overlays[-1]
                labels.append(fmt % (value, numpy.inf))
                
        if label_other:
            overlays += [other_overlay]
            labels += [str_fmt % label_other]
            
        overlays_labels.append((overlays, labels))

    overlays, labels = _recurse_buckets(overlays_labels, *overlays_labels.pop(-1))
    labels = [l.replace('-inf_', '    _') for l in labels]
    labels = [l.replace('_inf', '_   ') for l in labels]
    return {'buckets': overlays, 'labels': labels,
            'name': name, 'label_all': None, 'label_other': None}

