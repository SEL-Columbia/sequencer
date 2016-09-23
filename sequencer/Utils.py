# -*- coding: utf-8 -*-
__author__ = 'Brandon Ogle'

from pandas import DataFrame
import pandas as pd
import networkx as nx
import numpy as np
from numpy import sin, cos, pi, arcsin, sqrt
import string
import collections

def prep_data(network, metrics, loc_tol=.5):
    """
    This block of code performs fuzzy matching to align the floating point coordinates
    from the network shapefile with the input metrics, the drops non matching records
    """  

    # convert the node names from coords to integers, cache the coords as attrs
    # but ONLY if the nodes are themselves collections (which is the default for
    # networkx shapefile import)
    # otherwise, assume the coords attribute exists
    if(len(network.nodes()) > 0 and
       isinstance(network.nodes()[0], collections.Iterable)):
        network = nx.convert_node_labels_to_integers(network, label_attribute='coords')
    
    # convert special characters to dot notation 
    metrics.columns = parse_cols(metrics)

    # create a dataframe with the network nodes with int label index, attrs as cols
    node_df = DataFrame(network.node).T

    # join (x,y) coords to create a 'metrics coords'
    metrics['m_coords'] = map(tuple, metrics[['X', 'Y']].values)
    
    # cast coords to tuples (hashable)
    node_df['coords'] = node_df['coords'].apply(tuple)
    
    # build a vector of all the coordinates in the metrics dataframe
    coords_vec = np.vstack(metrics['m_coords'].values)

    # fuzzy_match takes a coordinate pair and returns the approximate match from the metrics dataframe using float_tol
    def fuzzy_match(coord):
        dists = hav_dist(coords_vec, coord)
        idx, val = min_tuple(dists)
        if val < loc_tol:
            return coords_vec[idx]
        return []

    # map over the coords in the nodal dataframe returning the fuzzy match from metrics
    node_df['m_coords'] = node_df['coords'].apply(fuzzy_match)
    
    # cast the coordinates back to tuples (hashable) 
    node_df['m_coords'] = node_df['m_coords'].apply(tuple)
    
    # now that we have identical metric coords in both node_df and metrics join on that column
    metrics = pd.merge(metrics, node_df, on='m_coords', left_index=True).sort()

    # TODO: Remove fuzzy matching and accept nodes and edges from same file
    #drop duplicate matches
    find_closest = lambda x: x.index[np.argmin(hav_dist(np.vstack(x['coords']), x.name))]
    closest_match = metrics.ix[metrics.groupby('m_coords').apply(find_closest).values]
    
    # anything in node_df that failed to find a fuzzy_match is a 'Fake' node
    fake_nodes = node_df[~node_df.index.isin(closest_match.index)]
    # reset m_coords on fakes
    fake_nodes['m_coords'] = fake_nodes['m_coords'].apply(lambda x: ())
    
    # tack the fake nodes on to the matched metrics (all values are NULL, except coord)
    metrics = pd.concat([closest_match, fake_nodes]).sort()
    
    # finally assume Network edges are bi-directional
    network = network.to_undirected().to_directed()
    
    return network, metrics

def min_tuple(series):
    idx = np.argmin(series)
    return (idx, series[idx])

def hav_dist(vector,point):
    lat = vector[:, 0]
    lon = vector[:, 1]
    x, y = point
    return get_hav_distance(lat, lon, x, y)

def get_hav_distance(lat, lon, pcode_lat, pcode_lon):
    """
    Find the distance between a vector of (lat,lon) and the reference point (pcode_lat,pcode_lon).
    """
    rad_factor = pi / 180.0  # degrees to radians for trig functions
    lat_in_rad = lat * rad_factor
    lon_in_rad = lon * rad_factor
    pcode_lat_in_rad = pcode_lat * rad_factor
    pcode_lon_in_rad = pcode_lon * rad_factor
    delta_lon = lon_in_rad - pcode_lon_in_rad
    delta_lat = lat_in_rad - pcode_lat_in_rad
    # Next two lines is the Haversine formula
    inverse_angle = (np.sin(delta_lat / 2) ** 2 + np.cos(pcode_lat_in_rad) *
                     np.cos(lat_in_rad) * np.sin(delta_lon / 2) ** 2)
    haversine_angle = 2 * np.arcsin(np.sqrt(inverse_angle))
    earth_radius =  6371010 # meters
    return haversine_angle * earth_radius

def haversine_distance(first_point, second_point):
    """Calculate the Haversine distance between two points on Earth."""
    # Implementation details copied from scikit-learn
    # http://scikit-learn.org/
    # 0.17/modules/generated/sklearn.neighbors.DistanceMetric.html
    #
    # https://github.com/scikit-learn/scikit-learn/blob
    # /a0b8c5291901bc28666bf6486babbfc584cea49c/sklearn/neighbors
    # /dist_metrics.pyx#L992-L1000
    p1 = np.radians(first_point)
    p2 = np.radians(second_point)
    sin0 = np.sin(0.5 * (p1[0] - p2[0]))
    sin1 = np.sin(0.5 * (p1[1] - p2[1]))
    return 2 * 6371010 * np.arcsin(np.sqrt(
        sin0 * sin0 + np.cos(p1[0]) * np.cos(p2[0]) * sin1 * sin1
    ))

def euclidean_distance(first_point, second_point):
    """Calculate the Euclidean distance between two points."""
    # http://stackoverflow.com/a/1401828
    return np.linalg.norm((first_point, second_point))

def get_euclidean_dist(point, coords):
    return np.sqrt(np.sum((coords - point) ** 2, axis=1))

def parse_cols(df):
    columns = df.columns
    strip_chars = lambda x: '.' if x not in list(string.letters) + map(str, range(10)) else x 
    columns = [''.join(map(strip_chars, col)) for col in columns]
    return columns


