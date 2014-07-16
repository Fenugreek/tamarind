"""
Find areas of high density in 2D data.
API follows that of scikit learn (e.g. sklearn.cluster.Kmeans).
"""

import numpy
from scipy.signal import convolve2d, get_window
from matplotlib import pyplot


def _get_window(desc, shape, size=None):

    window_shape = list(shape)
    for i in range(2):
        if shape[i] < 1: window_shape[i] = int(shape[i] * size)
        # round up to nearest odd integer
        if (window_shape[i] + 1) % 2: window_shape[i] = int(window_shape[i]) + 1

    if type(desc) == 'str': return get_window(desc, window_shape)

    # desc is like ('gaussian', 0.4).
    # In these cases, scipy.signal.get_window() doesn't handle 2D shapes.
    # So compute 1D windows first, and then multiply them together.
    if desc[1] < 1:
        desc = [(desc[0], int(desc[1] * s)) for s in window_shape]
    else: desc = [desc, desc]
    window = [get_window(d, s) for d, s in zip(desc, window_shape)]
    return numpy.mat(window[1]).T * numpy.mat(window[0])

    
class KWindows(object):
    """
    Find centers of high density, using local averaging and finding peaks (cf. heatmaps).
    
    Currently implemented only for 2D data, with inefficiencies in recomputing the
    convolution in some cases when unnecessary, and not using FFT.
    """

    def __init__(self, K=100, min_count=0.01, bins=100,
                 window=('gaussian' 0.4), shape=(0.1, 0.1)):

        self.params = {'bins': bins,
                       'window': window,
                       'shape': shape,
                       'K': K,
                       'min_count': min_count}
        
        self.window_ = _get_window(window, shape, bins)


    def fit(self, x1, x2, range=None):

        params = self.params
        bins = params['bins']
        window_center = [(s - 1) / 2 for s in self.window_.shape]
        
        self.histogram2d_ = numpy.histogram2d(x1, x2, bins=bins, range=range)
        bin_counts, bin_edges = self.histogram2d_[0], self.histogram2d_[1:]
        
        self.first_convolution_ = convolve2d(bin_counts, self.window_, mode='valid')
        max_idx = numpy.unravel_index(self.first_convolution_.argmax(),
                                      self.first_convolution_.shape)
        self.counts_ = [self.first_convolution_[max_idx]]
        binX, binY = max_idx[0] + window_center[0], max_idx[1] + window_center[1]
        self.bins_ = [(binX, binY)]
        self.centers_ = [(numpy.mean(bin_edges[0][binX : binX + 2]),
                          numpy.mean(bin_edges[1][binY : binY + 2]))]
        self.last_convolution_ = self.first_convolution_
        self.dense_mask_ = numpy.zeros(bin_counts.shape, dtype=bool)
        self.dense_mask_[max(0, binX - window_center[0]) : binX + window_center[0] + 1,
                         max(0, binY - window_center[1]) : binY + window_center[1] + 1] = \
                         True

        while len(self.centers_) < (params['K'] or bin_counts.size) and \
              self.counts_[-1] > params['min_count'] and \
              (bin_counts.size - numpy.sum(self.dense_mask_) >= self.window_.size):

            bin_counts = self.histogram2d_[0].copy()
            bin_counts[self.dense_mask_] = 0
            
            convolution = convolve2d(bin_counts, self.window_, mode='valid')
            # Don't find a center that's in a previously determined dense area
            masked = numpy.ma.array(convolution,
                                    mask=self.dense_mask_[window_center[0]:-window_center[0],
                                                          window_center[1]:-window_center[1]])
            max_idx = numpy.unravel_index(masked.argmax(), masked.shape)
            self.counts_.append(convolution[max_idx])
            binX, binY = max_idx[0] + window_center[0], max_idx[1] + window_center[1]
            self.bins_.append((binX, binY))
            self.centers_.append((numpy.mean(bin_edges[0][binX : binX + 2]),
                                  numpy.mean(bin_edges[1][binY : binY + 2])))
            self.last_convolution_ = convolution
            self.dense_mask_[max(0, binX - window_center[0]) : binX + window_center[0] + 1,
                             max(0, binY - window_center[1]) : binY + window_center[1] + 1] = \
                             True
        

    def plot_window(self, bin, *plotargs):

        bin_edges = self.histogram2d_[1:]
        window_center = [(s - 1) / 2 for s in self.window_.shape]

        left = bin_edges[0][max(0, bin[0] - window_center[0])]
        right = bin_edges[0][min(len(bin_edges[0]) - 2, bin[0] + window_center[0]) + 1]
        bottom = bin_edges[1][max(0, bin[1] - window_center[1])]
        top = bin_edges[1][min(len(bin_edges[1]) - 2, bin[1] + window_center[1]) + 1]

        pyplot.plot([left, left, right, right, left],
                    [bottom, top, top, bottom, bottom], *plotargs)


    def plot_windows(self, windows, *args, **kwargs):

        for w in windows:
            self.plot_window(self.bins_[w], *args)
            pyplot.text(self.centers_[w][0], self.centers_[w][1], str(w),
                        horizontalalignment='center', verticalalignment='center', **kwargs)
            
