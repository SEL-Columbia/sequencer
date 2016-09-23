# -*- coding: utf-8 -*-
__author__ = 'Brandon Ogle'

import pandas as pd
from functools import wraps
import networkx as nx
import numpy as np
import os
import logging
from sequencer.Utils import parse_cols

logger = logging.getLogger('sequencer')

def memoize(f):
    cache = {}
    
    # Provides access to this closures scope
    # In Python 3 this could be replaced with the nonlocal keyword
    class scope:
        last_prog = None
        
    @wraps(f)
    def memoizedFunction(*args, **kwargs):
        # Note:  only concerned with 2nd argument here
        # since it's only used in accumulate
        self = args[0]
        key = args[1]
        if key not in cache:
            cache[key] = f(*args, **kwargs)

            # Get the number of keys in the the cache and send to the progress meter
            if len(cache.keys()) != scope.last_prog:
                scope.last_prog = len(cache.keys())
            
        return cache[key]

    # Assigns a cache to the decorated func
    memoizedFunction.cache = cache

    return memoizedFunction

class Sequencer(object):
    
    def __init__(self, NetworkPlan):
        self.networkplan = NetworkPlan
        
        # Build a list of the fake nodes in the network
        self.fakes = self.networkplan.fake_nodes

        # Create a column containing the computed demand
        self.networkplan.metrics['nodal_demand'] = self.nodal_demand(self.networkplan.metrics)
        self.root_children = self.networkplan.root_child_dict()

    def _sequence(self):
        logger.info('Converting The Network to a HashMap (Warning: this is time consuming for large Networks)')
        # Convert the Network to a dict
        network = self.networkplan.network_to_dict()
        # The roots of the Network dict are the seed for the frontier
        frontier = network.keys()
        # Initialize a starting rank
        rank = 0  

        logger.info('Traversing The Input Network and Computing Decision Frontier')
        
        # While there exists nodes in the keys of the Network dict, continue to Sequence
        while frontier: 
            # Reset the max_ and choice nodes for the give state of the frontier
            max_ = choice = None
            # Iterate through the nodes in the frontier to find the max metric
            for node in frontier:
                # Get the accumulated values for the give node
                # The traversal is performed only in the first call due to Memoization
                accum_dict = self.accumulate(node)
                demand = accum_dict['demand']
                cost = accum_dict['cost']
                # Compute the metric
                if cost > 0:
                    metric = 1.0 * demand / cost
                else:
                    metric = np.inf

                if metric > max_:
                    # Update the metric and potential candidate
                    max_ = metric 
                    choice = node
                    choice_vars = accum_dict                    
            
            # Remove the candidate from the Network and shift its downstream neighbors to the keys
            for item in network.pop(choice):
                network.update(item) 

            # Update the frontier
            frontier = network.keys()
            
            # Build a row to be appended to the results dataframe
            choice_row =  {
                            'Sequence..Vertex.id'                   : choice,
                            'Sequence..Downstream.demand.sum.kwh'   : choice_vars['demand'],
                            'Sequence..Downstream.distance.sum.m'   : choice_vars['cost'],
                            'Sequence..Root.vertex.id'              : self.get_root(choice),
                            'Sequence..Upstream.id'                 : self.parent(choice),
                            'Sequence..Upstream.segment.distance.m' : self.upstream_distance(choice),
                            'Sequence..Decision.metric'             : 1.0 * choice_vars['demand'] / choice_vars['cost']
                          }

            
            if choice not in self.fakes:
                # Update the rank
                rank += 1
                choice_row['Sequence..Far.sighted.sequence'] = rank
                logger.debug('rank {} : node {} :: demand {} -> distance {} -> metric {}'.format(rank, choice,
                                                                                                 choice_vars['demand'], choice_vars['cost'],
                                                                                                 choice_vars['demand'] / choice_vars['cost']))
                yield choice_row
        
        # Clear the accumulate cache
        self.accumulate.cache.clear()

    def upstream_distance(self, node):
        """Computes the edge distance from a node to it's parent"""
        parent = self.parent(node)
        if parent is not None:
            return self.networkplan._distance(parent, node)
        return 0.0

    def sequence(self):
        """
        Compute the sequence (aka rank) of nodes and edges 
        
        This modifies the NetworkPlan member (so make a deep copy if you 
        need the original)
        """
        self.results = pd.DataFrame(self._sequence(), dtype=object).set_index('Sequence..Far.sighted.sequence')
        # Post process for output
        self._build_node_wkt()
        self._build_edge_wkt()
        self._clean_results()
    
        return self.output_frame

    def get_root(self, n):
        # Check the keys to see if the node even has an upstream root
        if n in self.root_children.keys():
            return None
        # If the node isn't a root, then iterate through the each root until we find the node
        for root, children in self.root_children.iteritems():
            if n in children:
                # Return the root the node belongs to
                return root
    
    @memoize
    def accumulate(self, n):
        """
        traverses the tree computing downstream aggregates and
        populating the cache
        """

        # Compute individual node variables
        demand = self.networkplan.metrics['nodal_demand'].ix[n]
        cost = self.upstream_distance(n)
        
        post_order_stack = []
        # create a post_order_traversal of downstream tree
        # then sum up accumulated values as we pop off 
        # un-ravel recursiveness since for large networks we hit a stack limit
        # 
        # Populate the stack 
        all_children = [(n, child) for child in self.networkplan.get_successors(n)]
        self.accumulate.cache[n] = {'demand': demand, 'cost': cost}
        while all_children:
            parent, child = all_children.pop()
            child_demand = self.networkplan.metrics['nodal_demand'].ix[child]
            child_cost = self.upstream_distance(child)
            self.accumulate.cache[child] = {'demand': child_demand, 'cost': child_cost}
            post_order_stack.append((parent, child))
            all_children += [(child, child_child) for child_child in self.networkplan.get_successors(child)]

        # now we have post_order_stack defined, pop it and sum cache back up the tree
        while post_order_stack:
            parent, child = post_order_stack.pop()
        
            # now add to parent
            self.accumulate.cache[parent]['demand'] += self.accumulate.cache[child]['demand']
            self.accumulate.cache[parent]['cost'] += self.accumulate.cache[child]['cost']

        
        # Return a dictionary of accumulated values
        return self.accumulate.cache[n]

    def output(self, path):

        out_results = 'sequenced-results.csv'
        out_shp = 'sequenced-network'
        
        # Write the shapefiles to path
        nx.write_shp(self.networkplan.network, os.path.join(path, out_shp))
        # Write the csv to path
        cast = {
                'Sequence..Vertex.id'            : int,
                'Sequence..Root.vertex.id'       : int ,
                'Sequence..Upstream.id'          : int,
                'Sequence..Far.sighted.sequence' : int}
        for k,v in cast.iteritems():
            self.output_frame[k] = self.output_frame[k].fillna(-9223372036854775807).astype(v)        
        self.output_frame.to_csv(os.path.join(path, 'temp.csv'), index=False, na_rep='NaN')
        
        with open(os.path.join(path, 'temp.csv')) as f:
            buff = f.read()
            while True:
                idx = buff.find('-9223372036854775807')
                if idx != -1:
                    buff = buff.replace('-9223372036854775807', 'NaN', idx)
                else:
                    break

        with open(os.path.join(path, out_results), 'w') as f:
            f.write(buff)
        os.remove(os.path.join(path, 'temp.csv'))
        # Trash the node shp files
        [os.remove(os.path.join(os.path.join(path, out_shp), x)) 
                for x in os.listdir(os.path.join(path, out_shp)) if 'node' in x]


    def _build_node_wkt(self):
        for node in self.networkplan.network.nodes():
            # Build Node WKT with Point coords
            self.networkplan.network.node[node]['Wkt'] = 'POINT ({x} {y})'.format(x=self.networkplan.coords[node][0],
                                                                                  y=self.networkplan.coords[node][1])
    def _build_edge_wkt(self):
        r = self.results
        # Iterate through the nodes and their parent
        for rank, fnode, tnode in zip(r.index, r['Sequence..Upstream.id'], r['Sequence..Vertex.id']):
            if fnode is not None:
                # Set the edge attributes with those found in sequencing
                self.networkplan.network.edge[fnode][tnode]['rank'] = int(rank)
                self.networkplan.network.edge[fnode][tnode]['distance'] = float(self.networkplan._distance(fnode, tnode))
                self.networkplan.network.edge[fnode][tnode]['id'] = int(tnode)
                fnode_coords = self.networkplan.coords[fnode]
                tnode_coords = self.networkplan.coords[tnode]
                
                # Build WKT Linestring with from_node and to_node coords
                self.networkplan.network.edge[fnode][tnode]['Wkt'] = 'LINESTRING ({x1} {y1}, {x2} {y2})'.format(x1=fnode_coords[0], y1=fnode_coords[1],
                                                                                                                x2=tnode_coords[0], y2=tnode_coords[1])
        # Filter empty edges
        edges = {(k1, k2): attr for k1, v in self.networkplan.network.edge.iteritems() if v!={}
                                for k2, attr in v.iteritems() if attr!={}}
        self.networkplan.network.edge = edges
        # Clear all edges from the networkx 
        self.networkplan.network.remove_edges_from(self.networkplan.network.edges())
        # Reset with the filtered edges
        self.networkplan.network.add_edges_from(edges.keys())
        # Set the atrributes
        attrs = set([v for i in edges.values() for v in i.keys()]) 
        for attr in attrs:
            nx.set_edge_attributes(self.networkplan.network, attr, pd.DataFrame(edges).ix[attr].to_dict())

    def parent(self, n):
        parent = (parent for parent in self.networkplan.get_predecessors(n))
        # Fake nodes will have no parent
        return next(parent, None)

    def _clean_results(self):
        """This joins the sequenced results on the metrics dataframe and reappends the dropped rows"""
        
        logger.info('Joining Sequencer Results on Input Metrics')
        # FIXME:  Remove this dependency on original_metrics
        orig = self.networkplan.original_metrics
        orig.columns = parse_cols(orig)
        self.networkplan.metrics.index.name = 'Sequence..Vertex.id'
        sequenced_metrics = pd.merge(self.networkplan.metrics.reset_index(), self.results.reset_index(), on='Sequence..Vertex.id')
        
        orig['m_coords'] = list(orig[['X', 'Y']].itertuples(index=False))
        cols_to_join_on = (sequenced_metrics.columns - orig.columns).tolist() + ['m_coords']
        union = pd.merge(orig, sequenced_metrics[cols_to_join_on], on='m_coords', how='outer')
        
        sorted_columns = orig.columns.tolist() + list(set(sequenced_metrics.columns) - set(orig.columns))
        self.output_frame = union[sorted_columns]
        self.output_frame = self.output_frame.drop(['m_coords', 'coords'], axis=1)
        self.output_frame['coords'] = list(self.output_frame[['X', 'Y']].itertuples(index=False))
                
        # Assert Output frame has same number of rows as input
        try:
            assert(len(orig.index) == len(self.output_frame.index))
        except:
            logger.error('Resulting Output does not match the number of rows in Input!')
        
        # Assert that Output has no duplicate coordinates THIS IS A BAD TEST AS THE PRECISION IS TRUNCATED
        try:
            assert(len(self.output_frame.index) == len(self.output_frame['coords'].unique()))
        except:
            logger.error('Duplicate rows detected in Output!')

        logger.info('DONE!')

        
    def nodal_demand(self, df):
        """Overload this method to compute your nodal demand"""
        raise NotImplemented()
