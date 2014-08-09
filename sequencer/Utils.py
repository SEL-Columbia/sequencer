# -*- coding: utf-8 -*-
__author__ = 'Brandon Ogle'

from pandas import DataFrame
import pandas as pd
import networkx as nx
import numpy as np

pd.options.mode.chained_assignment = None

def prep_data(network, metrics, prec=8):
    """Does its best to align your shapefile nodes with your csv"""
    
    # Transform node names from coords to ints and save coords as attr
    network = nx.convert_node_labels_to_integers(network, label_attribute='coords')
    # Make a DataFrame representation of the node data
    node_df = pd.DataFrame(network.node).T
    # Make note of the starting node names
    node_df['orig_name'] = node_df.index
    
    # Zip the metrics x, y column to create a coords column to join on
    metrics['coords'] = zip(metrics['X'], metrics['Y'])
    # If lucky this will align all of the data
    easy_match = pd.merge(node_df, metrics, on='coords', right_index=True)
    
    # You probably aren't lucky, so try to get the remaining matches
    # Any non matches will later be interpreted as `FAKE' nodes
    
    # Grab the complement of the easy_matches to search for reamining nodes
    remaining_universe = metrics.ix[metrics.index - easy_match.index]
    # Grab the nodes that weren't set in easy_match
    remaining_nodes = node_df[~node_df['coords'].isin(easy_match['coords'])]
    
    # Change precision to force matches
    remaining_nodes.loc[   :, 'coords'] = remaining_nodes['coords'].map(lambda x: np.around(x, prec))
    remaining_nodes.loc[   :, 'coords'] = remaining_nodes['coords'].map(tuple)

    remaining_universe.loc[:, 'coords'] = remaining_universe['coords'].map(lambda x: np.around(x, prec))
    remaining_universe.loc[:, 'coords'] = remaining_universe['coords'].map(tuple)
    
    # Attempt another match on remaining data
    hard_match = pd.merge(remaining_nodes, remaining_universe, on='coords', right_index=True)  

    # Join all the matches
    master = pd.concat([easy_match, hard_match]).drop_duplicates()
    
    # Anything still without a match?
    fakes = remaining_nodes[~remaining_nodes['coords'].isin(master['coords'])]
    master = pd.concat([master, fakes])

    # Reset the node names
    master = master.set_index('orig_name')
    master.index.name = 'node'
    master = master.sort()
    
    #TODO: assert network.node ≅ master.ix[:, :'coords'].T.to_dict() 

    # Finally assume the edges are bidirectional with this quick hack
    network = network.to_undirected()
    network = network.to_directed()

    return network, master
