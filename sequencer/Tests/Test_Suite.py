import numpy as np
import networkx as nx
import os

from pandas import DataFrame
import pandas as pd
from sequencer import NetworkPlan, Sequencer
from sequencer.Models import EnergyMaximizeReturn
from nose.tools import eq_ 

import sys
# Turn off progress bar and print statements
class catch_prints(object):
    def write(self, arg):
        pass
    def flush(self):
        pass

#sys.stdout = catch_prints()    

def gen_data():
    """
    generates test metrics and network, where the network is a 
    balanced graph of height 2 and branch factor 2
    """
    network = nx.balanced_tree(2, 2)

    metrics = DataFrame(network.node).T

    metrics['Demand'] =     [np.nan, 100, 50, 25, 12, 6, 3]
    metrics['Population'] = [np.nan, 100, 50, 25, 12, 6, 3]
    #level 0 
    metrics['coords'] = [np.array([125,10]) for x in  metrics.index]
    #level 2
    metrics['coords'].ix[1] = metrics['coords'].ix[0] + [-.5, -.25]
    metrics['coords'].ix[2] = metrics['coords'].ix[0] + [+.5, -.25]
    #level 3
    metrics['coords'].ix[3] = metrics['coords'].ix[1] + [-.25, -.25]
    metrics['coords'].ix[4] = metrics['coords'].ix[1] + [+.25, -.25]
    metrics['coords'].ix[5] = metrics['coords'].ix[2] + [-.25, -.25]
    metrics['coords'].ix[6] = metrics['coords'].ix[2] + [+.25, -.25]
    metrics['coords'] = metrics['coords'].apply(tuple)

    nx.set_node_attributes(network, 'coords', metrics.coords.to_dict())
    #nx.draw(network, nx.get_node_attributes(network, 'coords'))
    
    return metrics, network.to_directed()


def gen_data_with_fakes():
    """
    generate network and metrics where some of the network
    nodes do not have corresponding metrics records
    
    This should be sufficient for tests requiring fake nodes

    network looks like

       o     *
       |    / \
       *   *   *
      / \
     *   *

    where o is a fake node, * is not
    """
  
    # create disjoint graph with 2 trees, one rooted by a fake node
    network = nx.graph.Graph()
    edges = ((0, 1), (0, 2), (3, 4), (3, 5))
    network.add_edges_from(edges)

    # now add fake root to tree at 3
    network.add_edge(6, 3)

    # set coordinates
    base_coord = np.array([10, 10])
    coord_dict = {i: base_coord + [i*-1, i*-1] for i in range(6)}
    nx.set_node_attributes(network, 'coords', coord_dict)
    # and set fake node coordinates
    nx.set_node_attributes(network, 'coords', {6: np.array([10, 11])})

    # now set the metrics dataframe without the fake node
    metrics_data = {'Demand...Projected.nodal.demand.per.year': 
                    [100, 50, 25, 12, 6, 3],
                    'Population': [100, 50, 25, 12, 6, 3]}

    metrics = DataFrame(metrics_data)
    metrics['X'] = [ coord_dict[i][0] for i in range(6) ]
    metrics['Y'] = [ coord_dict[i][1] for i in range(6) ]

    return metrics, network
            
def test_sequencer_with_fakes():
    """
    Make sure we work with fake nodes
    """
    
    # for now, just make sure it runs without exceptions
    metrics, network = gen_data_with_fakes()
    nwp = NetworkPlan(network, metrics, prioritize='Population', proj='wgs4')
    model = EnergyMaximizeReturn(nwp)
    model.sequence()
    #todo:  check the result

 
class TestNetworkPlan(NetworkPlan):
    
    def __init__(self):
        self._metrics, self._network = gen_data()
        self.proj = 'wgs4'
        self.priority_metric = 'Population'
        self.coord_values = self.coords.values()
        
        # Set the edge weight to the distance between those nodes                                                                                                             
        self._weight_edges()

        # Transform edges to a rooted graph                                                                                                                                   
        self.direct_network()
        
        # List of fake nodes
        self.fake_nodes = [0]

        # Fillna with 0
        self._metrics = self.metrics.fillna(0)

class TestSequencer(Sequencer):
    
    def nodal_demand(self, df):
        return df['Demand']
    def sequence(self):
        return DataFrame(list(self._sequence())).set_index('Sequence..Far.sighted.sequence')

def test_is_tree():
    """Ensures all roots have in_degree of 0 and leaves have in_degree of 1"""
    nwp = TestNetworkPlan()
    in_degree = nwp.network.in_degree()
    # Test that all roots have in_degree == 0
    ensure_roots = [in_degree[root] == 0 for root in nwp.roots]
    # Test that all leaves have in_degree == 1
    ensure_leaves = [in_degree[leaf] == 1 for leaf in (set(nwp.network.node.keys()) - set(nwp.roots))]

    eq_(all(ensure_roots + ensure_leaves), True)

def test_accumulate_demand():
    """Tests that the accumulated demand is correct"""
    
    nwp = TestNetworkPlan()
    # Build dictionary of accumulated values for each node
    acc_dicts =  {node : TestSequencer(nwp).accumulate(node) for node in nwp.network.node.keys()}
    # Dictionary of known accumulated demand computed manually
    demands = {0: (100 + 50 + 25 + 12 + 6 + 3), 
               1: (100 + 25 + 12), 
               2: ( 50 +  6 +  3), 
               3:25, 4:12, 5:6, 6:3}
    
    # Assert that accumulate method and manual computation are equal
    eq_(np.all([acc_dicts[node]['demand'] == demands[node] for node in nwp.network.node.keys()]), True)

def test_accumulate_cost():
    """Tests that the accumulates costs are correct"""

    nwp = TestNetworkPlan()
    # Build dictionary of accumulated values for each node
    acc_dicts = {node : TestSequencer(nwp).accumulate(node) for node in nwp.network.node.keys()}
    def get_distance(f, t):
        return nwp._distance(f, t)

    # Manually compute downstream distances
    costs = {0 : sum([get_distance(0, 1), get_distance(0, 2), 
                      get_distance(1, 3), get_distance(1, 4), 
                      get_distance(2, 5), get_distance(2, 6)]),
             1 : sum([get_distance(0, 1), get_distance(1, 3), get_distance(1, 4)]),
             2 : sum([get_distance(0, 2), get_distance(2, 5), get_distance(2, 6)]),
             3 : get_distance(1, 3),
             4 : get_distance(1, 4),
             5 : get_distance(2, 5),
             6 : get_distance(2, 6)}


    costs = {node : (acc_dicts[node]['cost'], costs[node]) for node in nwp.network.node.keys()}
    eq_(np.all(map(lambda tup: np.allclose(*tup), costs.values())), True)

def test_sequencer_follows_topology():
    """Tests that the sequencer doesn't skip nodes in the network"""
    nwp = TestNetworkPlan()
    model = TestSequencer(nwp)
    results = model.sequence()
    fnodes = results['Sequence..Upstream.id']
    node_seq_num = {node: seq_num for seq_num, node in 
                    results['Sequence..Vertex.id'].iteritems()}

    #For each from_node, assert that the sequencer has already pointed to it or its a root
    eq_(np.all([fnode in nwp.roots or node_seq_num[fnode] < seq_num 
                for seq_num, fnode in fnodes.iteritems()]), True)


def test_sequencer_compare():
    """
    Test an old output to ensure we don't regress
    """
    input_dir = "data/sumaila/input"
    csv_file = os.path.join(input_dir, "metrics-local.csv")
    shp_file = os.path.join(input_dir, "networks-proposed.shp")
    nwp = NetworkPlan.from_files(shp_file, csv_file, prioritize='Population')
    model = EnergyMaximizeReturn(nwp)

    model.sequence()

    expected_dir = "data/sumaila/expected_output"
    exp_csv_file = os.path.join(expected_dir, "sequenced-results.csv")
    exp_df = pd.read_csv(exp_csv_file)
    # exp_shp_file = os.path.join(expected_dir, "edges.shp")
    # expected_nwp = NetworkPlan(shp_file, exp_csv_file, prioritize='Population')

    # now compare results to expected
    #expected_net = expected_nwp.network
    compare_fields = ['Sequence..Vertex.id', 'Sequence..Far.sighted.sequence']

    # exp_node_dict = expected_net.nodes(data=True)
    # exp_node_tups = [tuple(map(d.get, compare_fields)) for d in exp_node_dict]
    exp_node_tups = map(tuple, exp_df[compare_fields].values)
    seq_node_tups = map(tuple, model.output_frame[compare_fields].values)
    exp_node_tups = filter(lambda tup: tup[0] > 0, exp_node_tups)
    seq_node_tups = filter(lambda tup: tup[0] > 0, seq_node_tups)
    seq_node_tups = map(lambda tup: tuple(map(int, tup)), seq_node_tups)
    # seq_node_tups = [tuple(map(seq_node_dict[d].get, compare_fields)) for d in seq_node_dict]

    assert sorted(exp_node_tups, key=lambda tup: tup[0]) == \
           sorted(seq_node_tups, key=lambda tup: tup[0]),\
           "expected nodes do not match sequenced"
