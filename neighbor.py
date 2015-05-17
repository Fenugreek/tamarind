"""
Implement some nearest neighbors statistics.
"""
from __future__ import division
import numpy as np
import tamarind.stats

def rd_weights(weights, index=0, offset=0.5, clips=(.25, 4)):
    if offset: weights = weights + offset

    rd_weights = np.ones(len(weights))
    if len(weights) > index+1:
        rd_weights[index+1:] = weights[index] / weights[index+1:]
    if index > 0 and len(weights) > index:
        rd_weights[:index] = weights[index] / weights[:index]

    return np.clip(rd_weights, *clips)


def stats(response, distance_matrix, knn, weights=None, cutoff=None,
          distance_weights=None):

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
            if distance_weights: w *= distance_weights(distances)
        elif distance_weights: w = distance_weights(distances)
        else: w = None

        
        d_stats.append(tamarind.stats.Full.stats(distances, weights=w))
        r_stats.append(tamarind.stats.Full.stats(response[row[:len(distances)]], weights=w))

    return tamarind.stats.Datab(d_stats, process_Nones=True), \
           tamarind.stats.Datab(r_stats, process_Nones=True), neighbors


def distance_matrix(features, power=2, weights=None):

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
