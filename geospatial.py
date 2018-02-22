"""
Implement some utilities dealing with geography and geometry.
"""

import numpy as np


class Earth(object):
    """Constants relating to the Earth."""
    radius = 6371.0


def hsin(angle):
    """Haversine function."""
    return np.sin(angle / 2.0)**2


def gc_distance(latlong1, latlong2):
    """
    Return great circle distance between GPS co-ords, in kilometers.
    Uses haversine formula. See https://en.wikipedia.org/wiki/Great-circle_distance
    Assumes lat/long in radians.
    """

    arg = hsin(latlong1[0] - latlong2[0]) + np.cos(latlong1[0]) * np.cos(latlong2[0]) * \
          hsin(latlong1[1] - latlong2[1])

    return Earth.radius * 2 * np.arcsin(arg**0.5)


def distance_matrix(latlongs):
    """ Returns matrix of distances between input degree latlongs."""
    
    distances = np.empty((len(latlongs), len(latlongs)))
    for i, lat_long_i in enumerate(latlongs):
        distances[i, i] = 0.0
        for j, lat_long_j in enumerate(latlongs[i+1:]):
            distances[i, i+1+j] = gc_distance(np.radians(lat_long_i),
                                              np.radians(lat_long_j))
            distances[i+1+j, i] = distances[i, i+1+j]

    return distances


def point_in_polygon(poly, point):
    """
    Determine if point <x, y> is inside the given polygon (list of <x, y> vertices,
    in clockwise order), using the ray casting method.

    From http://geospatialpython.com/2011/01/point-in-polygon.html    
    """
    x, y = point
    n = len(poly)

    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y- p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

