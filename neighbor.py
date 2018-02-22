"""
Implement some nearest neighbors statistics.
"""

import numpy as np
from . import stats as T_stats

def reciprocal_weights(distances, scale=1, power=1):
    """
    Function to generate weights that tail off for more distant points.
    """
    
    return (1 / (1 + distances*scale))**power


def stats(response, distance_matrix, knn, weights=None, cutoff=None,
          reciprocal=None):
    """
    Return for each element of response:
    a) stats of distances of knn nearest neighbors,
    b) stats of response among knn nearest neighbors, and
    c) list of the knn nearest neighbors, subject to cutoff.

    weights:
    return weighted stats with these as weights of the neighbors.

    cutoff:
    ignore neighbors further than this, even if within knn nearest.

    reciprocal:
    weight further neighbors less, calling reciprocal_weights().
    """
    
    nearest = np.argsort(distance_matrix, axis=1)[:, 1:]
    r_stats, d_stats, neighbors = [], [], []
    for count, row in enumerate(nearest):
        distances = distance_matrix[count][row[:knn]]
        if cutoff is not None:
            cutoff_index = np.searchsorted(distances, cutoff, side='right')
            if cutoff_index < knn:
                if cutoff_index == 0:
                    d_stats.append(None)
                    r_stats.append(None)
                    neighbors.append(None)
                    continue
                distances = distances[:cutoff_index]
        neighbors.append(row[:len(distances)])
            
        if weights is not None:
            w = weights[row[:len(distances)]]
            if reciprocal: w *= reciprocal_weights(distances, *reciprocal)
        elif distance_weights: w = distance_weights(distances)
        else: w = None

        
        d_stats.append(T_stats.Full.stats(distances, weights=w))
        r_stats.append(T_stats.Full.stats(response[row[:len(distances)]], weights=w))

    return T_stats.Datab(d_stats, process_Nones=True), \
           T_stats.Datab(r_stats, process_Nones=True), neighbors


def distance_matrix(features, power=2, weights=None):
    """
    Compute Euclidean distance between feature vectors, optionally weighting
    each dimension.
    """
    
    recip = 1.0 / power
    if weights is not None:
        features = np.array(weights)[np.newaxis, :] / np.sum(weights) * features
    
    distances = np.empty((len(features), len(features)))
    for i, feature_i in enumerate(features):
        distances[i, i] = 0.0
        for j, feature_j in enumerate(features[i+1:]):
            distances[i, i+1+j] = np.sum(np.abs(feature_j - feature_i)**power)**recip
            distances[i+1+j, i] = distances[i, i+1+j]

    return distances
