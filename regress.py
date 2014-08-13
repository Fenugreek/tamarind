"""
Perform multivariate weighted linear regression, handling nans and returning
associated statistics.
"""

from __future__ import division

import sys
import numpy, numpy.linalg
import stats, arrays, strings
from arrays import nice_array
import datab as db


def _format_inputs(responses, predictors, weights=None):

    if not len(responses): return numpy.array([]), numpy.array([]), numpy.array([])
    responses = nice_array(responses, shape=numpy.size(responses))
    predictors = nice_array(predictors, shape=(len(responses),
                                               numpy.size(predictors) / len(responses)))
    if weights is not None:
        weights = nice_array(weights, shape=numpy.shape(responses))

    return responses, predictors, weights


class Regress(object):
    """

    SYNOPSIS
    
    r = Regress(npreds=2, weighted=True, constant=True) 
    r.update(responses, predictors, weights=w)
    OR
    s = Regress(responses, predictors, weights=w)
    THEN
    results = r.compute()

    OPTIONS

    npreds: Number of predictors.
    weighted: Whether weights accompany datapoints. Defaults to False
    store_last (default False): Store values processed by last call to update().

    ATTRIBUTES

    statistics:
    Regression results so far. Needs call to obj.compute() to refresh.
    
    Multivariate:
    Multivariate statistics on the predictors and responses computed so far. Needs
    call to obj.compute() to refresh. Response is the last index.

    npreds, constant, weighted:
    No. of predictors, if there is an additional constant term, and whether weighted.
    """
    
    def __init__(self, *data, **opts):

        if 'weighted' in opts: self.weighted = opts['weighted']
        if 'weights' in opts and opts['weights'] is not None:
            self.weighted = True
            weights = opts['weights']
        else:
            if not hasattr(self, 'weighted'): self.weighted = False
            weights = None

        if len(data) > 2 or len(data) == 1:
            raise SyntaxError('Odd number of arguments to constructor.')
        elif len(data) == 2:
            responses, predictors, weights = _format_inputs(data[0], data[1], weights)
            if 'npreds' in opts and opts['npreds'] != numpy.shape(predictors)[-1]:
                raise ValueError('Number of columns in data incompatible with npreds option.')
            self.npreds = numpy.shape(predictors)[-1]
        elif 'npreds' in opts:
            if opts['npreds'] < 1: raise ValueError('Number of predictors must be at least 1.')
            else: self.npreds = opts['npreds']
        else: 
            raise ValueError('Need to specify npreds option.')

        
        if 'constant' in opts:
            self.constant = opts['constant']
            del opts['constant']
        else: self.constant = True
                
        if 'names' in opts and opts['names'] is not None:
            self.names = opts['names']
            if numpy.isscalar(self.names): self.names = [self.names]
            if len(self.names) != self.npreds:
                raise AssertionError('Number of names supplied should match npreds.')
            del opts['names']
        else: self.names = ['pred'+str(i) for i in range(self.npreds)]
        if self.constant: self.names = ['constant'] + self.names

        self.statistics = {'residual_squares': numpy.nan,
                           'residual': numpy.nan,
                           'R2': numpy.nan,
                           'count': 0,
                           'size': 0,
                           't-stat': None,
                           'covarianceB': None,
                           'coefficients': None,
                           'names': self.names}
        if self.weighted:
            self.statistics['mean_wt'] = numpy.nan
            self.statistics['d_count'] = 0.0

        if 'store_last' in opts: store_last = opts['store_last']
        else: store_last = False
        if len(data) and len(responses):
            self.Multivariate = stats.Multivariate(\
                numpy.ma.hstack((predictors, responses[:, numpy.newaxis])),
                weights=weights, store_last=store_last)
        else:
            self.Multivariate = stats.Multivariate(\
                nvars=self.npreds + 1, weighted=self.weighted, store_last=store_last)



    def update(self, responses, predictors, weights=None):
        """
        Dimension of responses must at least be one.
        Dimension of responses and weights must be the same.
        Dimension of predictors must be one greater than that of responses,
        unless npreds == 1.
        """

        responses, predictors, weights = \
            _format_inputs(responses, predictors, weights)
        self.Multivariate.update(
            numpy.ma.hstack((predictors, responses[:, numpy.newaxis])), weights)


    def compute(self, responses=None, predictors=None, weights=None,
                forecast=False, errors=False, use_last=True):
        """
        If constant=True when object initialized, first entry of arrays such as
        'coefficients', 't-stat' refer to the constant.

        If responses, predictors (and weights if object is weighted) are given,
        residual statistics are computed based on these. Otherwise, if use_last is
        True and if object was initialized with store_last=True, uses last update
        vales for the residual statistics.

        forecast, errors:
        Set these flags to store predicted values and the sigma around them in
        self.statistics['forecast'] and self.statistics['errors'] respectively.
        """

        m_stats = self.Multivariate.compute()
        
        statistics = self.statistics
        stat_fields = ['count', 'size']
        if self.weighted: stat_fields.extend(['mean_wt', 'd_count'])
        for stat in stat_fields: statistics[stat] = m_stats[stat]
        if statistics['count'] <= 1: return statistics

        XTX = m_stats['mean_ij'][:-1, :-1]
        XTY = m_stats['mean_ij'][:-1, -1]
        if self.constant:
            means = [s.statistics['mean'] for s in self.Multivariate.Sparse]
            XTY = numpy.insert(XTY, 0, means[-1])
            XTX = numpy.insert(XTX, 0, means[:-1], axis=0)
            XTX = numpy.insert(XTX, 0, [1] + means[:-1], axis=1)
        XTXi = numpy.linalg.inv(XTX)
        statistics['coefficients'] = numpy.sum(XTXi * XTY, axis=1)

        if responses is None or predictors is None:
            if use_last and self.Multivariate.last_update is not None:
                values, weights, valid = self.Multivariate.last_update
                responses, predictors = values[:, -1], values[:, :-1]
            else: return statistics

        if self.constant:
            predictors = numpy.ma.hstack((numpy.ma.ones((len(predictors), 1)), predictors))
            predictors.fill_value = numpy.nan
        predicted = numpy.ma.sum(predictors * statistics['coefficients'], axis=1)
        if forecast: statistics['forecast'] = predicted
            
        statistics['residual_squares'] = numpy.ma.average((responses - predicted)**2,
                                                          weights=weights)
        statistics['residual'] = numpy.sqrt(statistics['residual_squares'])
        residual_stats = stats.Sparse(responses - predicted, weights=weights).compute()
        if self.constant:
            # we are always guaranteed predicted values, so no mask necessary.
            orig_squares = m_stats['variance_ij'][-1, -1]
        else:
            # don't count datapoints where we have no predicted values;
            # also, don't subtract mean from response (otherwise -ve R2 possible)
            orig_squares = stats.Sparse(responses[~predicted.mask], weights=
                                          weights[~predicted.mask]).compute()['mean_square']
        if orig_squares is not None:
            statistics['R2'] = 1 - statistics['residual_squares'] / orig_squares
        
        statistics['covarianceB'] = XTXi * statistics['residual_squares']
        if errors:
            inter = numpy.array(numpy.mat(predictors.data) * statistics['covarianceB'])
            statistics['errors'] = numpy.sqrt(numpy.sum(inter * predictors.data, axis=1))

        coeff_count = numpy.shape(predictors)[-1]
        for f in 't-stat', 'sigma': statistics[f] = arrays.nans(coeff_count)
        for i in range(coeff_count):
            covariance = statistics['covarianceB'][i, i]
            if covariance <= 0.0: continue
            if self.weighted: d_count = m_stats['d_count_ij'][i, i]
            else: d_count = m_stats['count_ij'][i, i]
            statistics['sigma'][i] = numpy.sqrt(covariance)
            statistics['t-stat'][i] = statistics['coefficients'][i] * \
                                      numpy.sqrt(d_count / covariance)
        
        return statistics


class Datab(stats.Datab):
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
            ('R2', float, '%7.4f'), ('coefficients', float, '%10.6f'),
            ('t-stat', float, '(%7.3f)'), ('sigma', float, '[%9.6f]'),)
    spec_index = dict([(f[0], i) for i, f in enumerate(spec)])
    spec_d = dict([[s[0], s] for s in spec])

    # the below should be in order the fields appear in spec.
    default_output_fields = ('count', 'd_count', 'mean_wt', 'R2', 'coefficients')

    def __new__(subtype, results, names=None, name='key',
                formats=None, **datab_args):

        first_result = results[0]
        if names is None: names = first_result['names']

        row_stats = []
        key_len = len(name)
        for result in results:
            if 'label' in result:
                row_stats.append([str(result['label'])])
                key_len = max(key_len, len(row_stats[-1][0]))
            else: row_stats.append([])

        my_spec = []
        if 'label' not in first_result: name = None
        else: my_spec.append((name, 'S'+str(key_len), '%-'+str(key_len)+'s'))

        for s in Datab.spec:
            if s[0] == 't-stat' or s[0] == 'sigma': continue # handled with coefficients
            if s[0] not in first_result or first_result[s[0]] is None: continue
            if s[0] != 'coefficients':
                my_spec.append(s)
                for index, result in enumerate(results):
                    row_stats[index].append(result[s[0]])
                continue

            # s[0] == coefficients
            for count in range(len(first_result['coefficients'])):
                my_spec.append((names[count], s[1], formats or s[2]))
                for index, result in enumerate(results):
                    row_stats[index].append(result['coefficients'][count] if
                                            result['coefficients'] is not None else numpy.nan)

                for field in 't-stat', 'sigma':
                    if field in result and result[field] is not None:
                        my_spec.append((names[count]+field[0].upper(), float,
                                        Datab.spec_d[field][2]))
                        for index, result in enumerate(results):
                            row_stats[index].append(result[field][count] if
                                                    result[field] is not None else numpy.nan)


        obj = db.Datab.__new__(subtype, [tuple(r) for r in row_stats],
                               my_spec, index=name, **datab_args)

        obj.names = names
        
        obj.default_output_fields = []
        for field in Datab.default_output_fields:
            if field in obj.field_spec: obj.default_output_fields.append(field)
            if field == 'coefficients':
                for name in names:
                    obj.default_output_fields.append(name)
                    if name + 'T' in obj.field_spec:
                        obj.default_output_fields.append(name + 'T')
        obj.default_output_fields = numpy.array(obj.default_output_fields)

        return obj


    def output(self, **opts):

        renames = []
        for name in self.names:
            if name + 'T' in self.field_spec: renames.append([name + 'T', ' '])
        super(Datab, self).output(rename=renames, **opts)


def regress(responses, predictors, weights=None, constant=True,
            forecast=False, errors=False,
            axis=None, step=1, sliced=None, select=None,
            overlay=None, split=None, buckets=None, group=None,
            labels=None, label_index=None, label_all='All', label_other='Other', 
            datab=None, names=None, name=None, formats=None):
    """
    Wrapper around Regress(*args, **kwargs).compute(), handling some additional
    options.

    data can be two dimensional, and axis can be 0 or 1. In this case,
    a list of statistics-records is returned, in Datab form.

    overlay:
    run stats only for records selected by this mask.

    split:
    run stats for all records, records selected by this mask, and for
    the others, returning a 3-tuple of results. Does not work with axis
    option, or if data is a dict.

    buckets:
    run stats for all records, and for records selected by each of the masks
    in this list of masks. Does not work with axis option, or if data is a dict.

    group:
    bucket stats by values in this field.

    sliced:
    run stats for records selected by this slice.

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

    names:
    in the header, labeled the predictor columns with these strings.

    name:
    in the header, label the key column with this string.

    datab:
    Return results in datab format rather than as a list, if appropriate.
    Defaults to True.

    formats:
    If using datab format, use this to pretty print floats. Defaults to '%9.6f'.
    """

    if datab is None: datab = True
    if datab == True:
        # datab output cannot hold forecasts or errors per datapoint
        forecast = False
        errors = False

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

    responses, predictors, weights = \
        arrays.select([nice_array(responses), nice_array(predictors), nice_array(weights)],
                        sliced=sliced, overlay=overlay, select=select)

    results = []
    if label_all is not None:
        reg = Regress(responses, predictors, weights=weights,
                      constant=constant, names=names, store_last=True)
        results.append(reg.compute(forecast=forecast, errors=errors))

    if axis is None and numpy.isscalar(step) and step == 1:
        if buckets is None:
            if not datab: return results[0]
            else: return Datab(results, formats=formats)
        else:
            if label_all is not None: results[-1]['label'] = label_all
            if label_other: other = numpy.ones(numpy.shape(responses), dtype=bool)
            buckets = arrays.select(buckets,
                                      sliced=sliced, overlay=overlay, select=select)

            for b, label in zip(buckets, labels):
                respb, predb, wtb = arrays.select([responses, predictors, weights],
                                                    select=b)
                reg = Regress(respb, predb, weights=wtb,
                              constant=constant, names=names, store_last=True)
                results.append(reg.compute(forecast=forecast, errors=errors))
                results[-1]['label'] = label             
                if label_other: other[b] = False
                
            if label_other:
                respb, predb, wtb = arrays.select([responses, predictors, weights],
                                                    select=other)
                reg = Regress(respb, predb, weights=wtb,
                              constant=constant, names=names, store_last=True)
                results.append(reg.compute(forecast=forecast, errors=errors))
                results[-1]['label'] = label_other

            if datab is False: return results
            else: return Datab(results, name=name or 'key', formats=formats)
    else:
        if buckets is not None:
            raise AssertionError('split/buckets option not supported with axis/step option.')
        if label_all is not None: results[-1]['label'] = label_all
            
    if axis > 1 or axis < 0 or numpy.ndim(responses) != 2:
        raise IndexError('Got unsupported axis option value that is ' +
                         'not 0 or 1; or data is not two-dimensional')

    if axis == 0:
        responses = responses.transpose()
        predictors = predictors.transpose()
        if weights is not None: weights = weights.transpose()
    
    start_idx = 0
    count = 0
    while start_idx < len(responses):
        
        row_responses, row_predictors, row_weights = \
            arrays.select([responses, predictors, weights],
                            sliced=(start_idx, start_idx + step, 1))
        r = Regress(row_responses, row_predictors, weights=row_weights,
                    constant=constant, names=names, store_last=True)
        start_idx += step
        count += 1
        if not r.Multivariate.count: continue
        
        results.append(r.compute(forecast=forecast, errors=errors))
        if labels is not None and len(labels):
            results[-1]['label'] = labels[count - 1]
        elif label_index is not None:
            results[-1]['label'] = label_index[start_idx - step] + '-'
        else: results[-1]['label'] = str(start_idx) + '-'


    if datab is False: return results
    else: return Datab(results, name=name or 'key', formats=formats)


def summary(responses, predictors, weights=None, constant=True,
            axis=None, step=1, sliced=None, select=None,
            overlay=None, split=None, buckets=None, group=None,
            labels=None, label_index=None, label_all='All', label_other='Other', 
            names=None, name=None, formats=None, **opts):
    """
    Convenience wrapper that, roughly speaking, calls Regress.regress(*args, **kwargs)
    and then prints results in nice tabulated form, using Regress.Datab.

    See documentation for Regress.regress.
    """
    
    results = regress(responses, predictors, weights=weights,
                      forecast=False, errors=False,
                      constant=constant, axis=axis, step=step,
                      sliced=sliced, select=select, overlay=overlay,
                      split=split, buckets=buckets, group=group,
                      labels=labels, label_index=label_index, label_all=label_all, label_other=label_other,
                      datab=True, names=names, name=name, formats=formats)
    results.output(**opts)


def loop_summary(responses, predictors, weights=None, constant=True, names=None,
                 sliced=None, select=None, overlay=None,
                 labels=None, name=None, formats=None, **opts):
    """
    Calls summary() in a loop for multiple responses/predictors.
    """

    if type(responses) == tuple:
        if type(predictors) == tuple:
            for count, predictor in enumerate(predictors):
                loop_summary(responses, predictor, weights=weights, constant=constant,
                             names=names or ['pred'+str(count)],
                             sliced=sliced, overlay=overlay, select=select,
                             labels=labels, name=name, formats=formats, **opts)
            return
        lpredictors = [predictors for r in responses]
        lresponses = responses
        name = name or 'resp'
    elif type(predictors) == tuple:
        lresponses = [responses for r in predictors]
        lpredictors = predictors
        name = name or 'pred'
    else: raise ValueError('predictors or responses need to be tuple so looping can happen.')

    output = []
    count = 0
    for response, predictor in zip(lresponses, lpredictors):
        results = regress(response, predictor, weights=weights, constant=constant,
                          names=names, sliced=sliced, overlay=overlay, select=select,
                          datab=False)
        if labels is not None and len(labels): results['label'] = labels[count]
        else: results['label'] = len(output)
        output.append(results)
        count += 1

    output = Datab(output, name=name, formats=formats)
    output.output(**opts)
    
