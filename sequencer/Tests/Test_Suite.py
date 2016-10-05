import numpy as np
import networkx as nx
import os

from pandas import DataFrame
import pandas as pd
from sequencer import NetworkPlan, Sequencer
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

    metrics = DataFrame()

    metrics['Demand'] =     [100, 50, 25, 12, 6, 3]
    metrics['Population'] = [100, 50, 25, 12, 6, 3]
    #level 0 
    base_coord = np.array([125, 10])
    coord_dict = {0: np.array([125, 10])}
    coord_dict[1] = coord_dict[0] + [-.5, -.25]
    coord_dict[2] = coord_dict[0] + [+.5, -.25]
    coord_dict[3] = coord_dict[1] + [-.25, -.25]
    coord_dict[4] = coord_dict[1] + [+.25, -.25]
    coord_dict[5] = coord_dict[2] + [-.25, -.25]
    coord_dict[6] = coord_dict[2] + [+.25, -.25]
                                   
    # assign x, y
    metrics['X'] = [coord_dict[i][0] for i in range(1, 7)]
    metrics['Y'] = [coord_dict[i][1] for i in range(1, 7)]
    nx.set_node_attributes(network, 'coords', coord_dict)
    #nx.draw(network, nx.get_node_attributes(network, 'coords'))
    
    return metrics, network.to_directed()


def gen_data_with_fakes():
    """
    generate network and metrics where some of the network
    nodes do not have corresponding metrics records
    
    This should be sufficient for tests requiring fake nodes

    network looks like (fake node starred, demand in parens)

                   6* 
                   |    
                   |
       0(100)      3(12)
      / \         / \   
     /   \       /   \  
    1(50) 2(25) 4(6)  5(3)

    Also returns edge_rank:  dict of edge -> rank 
    """
  
    # create disjoint graph with 2 trees, one rooted by a fake node
    network = nx.graph.Graph()
    edges = ((0, 1), (0, 2), (3, 4), (3, 5))
    network.add_edges_from(edges)

    # now add fake root to tree at 3
    network.add_edge(6, 3)

    # set coordinates (roughly match diagram above)
    base_coord = np.array([10, 10])
    fake_coord = np.array([20, 9])
    coord_dict = {0: base_coord, 
                  1: base_coord + [-1, 1], 
                  2: base_coord + [1, 1], 
                  3: fake_coord + [0, 1],
                  4: fake_coord + [-1, 2],
                  5: fake_coord + [1, 2],
                  6: fake_coord}

    nx.set_node_attributes(network, 'coords', coord_dict)
    # now set the metrics dataframe without the fake node
    metrics_data = {'Demand...Projected.nodal.demand.per.year': 
                    [100, 50, 25, 12, 6, 3],
                    'Population': [100, 50, 25, 12, 6, 3]}

    metrics = DataFrame(metrics_data)
    # Note, we skip fake node here
    metrics['X'] = [ coord_dict[i][0] for i in range(6) ]
    metrics['Y'] = [ coord_dict[i][1] for i in range(6) ]
    
    # assign expected ranks to nodes, edges (the sequence) 
    # note: 
    # - ranks are 1-based and originally assigned to nodes
    # - edges are assigned rank based on the "to" node
    # - fake nodes are skipped when assigning rank
    # (See Sequencer.sequencer._sequence for details)
    node_rank = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    edge_rank = {(0, 1): 2, (0, 2): 3, (6, 3): 4, (3, 4): 5, (3, 5): 6}
    return metrics, network, node_rank, edge_rank
            

def test_sequencer_with_fakes():
    """
    Make sure we work with fake nodes
    """
    
    # for now, just make sure it runs without exceptions
    metrics, network, node_rank, edge_rank = gen_data_with_fakes()
    nwp = NetworkPlan(network, metrics, prioritize='Population', proj='wgs4')
    model = Sequencer(nwp, 'Demand...Projected.nodal.demand.per.year')
    results = model.sequence()

    node_ids = results['Sequence..Vertex.id']
    sequence_ids = results['Sequence..Far.sighted.sequence']
    actual_node_rank = dict(zip(node_ids, sequence_ids))
    actual_edge_rank = {k: v['rank'] for k, v in 
                        model.networkplan.network.edge.iteritems()}
    assert node_rank == actual_node_rank,\
           "Node sequencing is not what was expected"
    assert edge_rank == actual_edge_rank,\
           "Edge sequencing is not what was expected"

 
def get_network_plan():
        
    metrics, network = gen_data()
    nwp = NetworkPlan(network, metrics, prioritize="Population", proj="wgs4")
    nwp.fake_nodes = [0]
    return nwp


def test_is_tree():
    """Ensures all roots have in_degree of 0 and leaves have in_degree of 1"""
    nwp = get_network_plan()
    in_degree = nwp.network.in_degree()
    # Test that all roots have in_degree == 0
    ensure_roots = [in_degree[root] == 0 for root in nwp.roots]
    # Test that all leaves have in_degree == 1
    ensure_leaves = [in_degree[leaf] == 1 for leaf in (set(nwp.network.node.keys()) - set(nwp.roots))]

    eq_(all(ensure_roots + ensure_leaves), True)

def test_accumulate_demand():
    """Tests that the accumulated demand is correct"""
    
    nwp = get_network_plan()
    # Build dictionary of accumulated values for each node
    acc_dicts =  {node : Sequencer(nwp, 'Demand').accumulate(node) for node in nwp.network.node.keys()}
    # Dictionary of known accumulated demand computed manually
    demands = {0: (100 + 50 + 25 + 12 + 6 + 3), 
               1: (100 + 25 + 12), 
               2: ( 50 +  6 +  3), 
               3:25, 4:12, 5:6, 6:3}
    
    # Assert that accumulate method and manual computation are equal
    eq_(np.all([acc_dicts[node]['demand'] == demands[node] for node in nwp.network.node.keys()]), True)

def test_accumulate_cost():
    """Tests that the accumulates costs are correct"""

    nwp = get_network_plan()
    # Build dictionary of accumulated values for each node
    acc_dicts = {node : Sequencer(nwp, 'Demand').accumulate(node) for node in nwp.network.node.keys()}
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
    nwp = get_network_plan()
    model = Sequencer(nwp, 'Demand')
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
    model = Sequencer(nwp, 'Demand...Projected.nodal.demand.per.year')

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
