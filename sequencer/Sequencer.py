# -*- coding: utf-8 -*-
__author__ = 'Brandon Ogle'

import pandas as pd
from functools import wraps
import networkx as nx
import numpy as np
import os
import sys
import logging
from sequencer.Utils import parse_cols

logger = logging.getLogger('Sequencer  ')
logger.setLevel(logging.INFO)
    
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s : %(name)s [%(levelname)s] : %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def memoize(f):
    cache = {}
    
    # Provides access to this closures scope
    # In Python 3 this could be replaced with the nonlocal keyword
    class scope:
        last_prog = None
        
    @wraps(f)
    def memoizedFunction(*args):
        if args not in cache:
            cache[args] = f(*args)

            # Get the number of keys in the the cache and send to the progress meter
            if len(cache.keys()) != scope.last_prog:
                scope.last_prog = len(cache.keys())
                if args[0]._progress_meter(scope.last_prog) == 1:
                    # If the progress meter is already shown to be complete
                    # Go to next line in console                
                    sys.stdout.write('\n')
            
        return cache[args]

    # Assigns a cache to the decorated func
    memoizedFunction.cache = cache

    return memoizedFunction

class Sequencer(object):
    
    def __init__(self, NetworkPlan):
        self.networkplan = NetworkPlan
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
                metric = 1.0 * demand / cost

                if metric > max_:
                    # Update the metric and potential candidate
                    max_ = metric 
                    choice = node
                    choice_vars = accum_dict                    
            
            # Remove the candidate from the Network and shift its downstream neighbors to the keys
            for item in network.pop(choice):
                if type(item) != int:
                    network.update(item) 
                else:
                    network.update({item:[]})

            # Update the frontier
            frontier = network.keys()
            
            # Build a row to be appended to the results dataframe
            choice_row =  {
                            'Sequence..Vertex.id'                   : choice,
                            'Sequence..Downstream.demand.sum.kwh'   : choice_vars['demand'],
                            'Sequence..Downstream.distance.sum.m'   : choice_vars['cost'],
                            'Sequence..Root.vertex.id'              : self.get_root(choice),
                            'Sequence..Upstream.id'                 : self.parent(choice),
                            'Sequence..Upstream.segment.distance.m' : self.networkplan.distance_matrix[self.parent(choice), choice],
                            'Sequence..Decision.metric'             : 1.0 * choice_vars['demand'] / choice_vars['cost']
                          }

            # Only yield the row if it is not a fake node
            if choice_row['Sequence..Upstream.id'] is not None:
                # Update the rank
                rank += 1
                choice_row['Sequence..Far.sighted.sequence'] = rank
                logger.debug('rank {} : node {} :: demand {} -> distance {} -> metric {}'.format(rank, choice,
                                                                                                 choice_vars['demand'], choice_vars['cost'],
                                                                                                 choice_vars['demand'] / choice_vars['cost']))
                yield choice_row
        
        # Clear the accumulate cache
        self.accumulate.cache.clear()

    def sequence(self):
        self.results = pd.DataFrame(self._sequence()).set_index('Sequence..Far.sighted.sequence')
        
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
        """traverses the tree computing downstream aggregates"""
        # Compute individual node variables
        demand = self.networkplan.metrics['nodal_demand'].ix[n]
        parent = self.parent(n)
        cost = self.networkplan.distance_matrix[parent, n] if parent != None else 0.0
        # Compute the above variables for all child nodes
        downstream_vars = [self.accumulate(child) for child, edge in enumerate(self.networkplan.adj_matrix[n, :]) if edge]
        
        if downstream_vars:
            # Aggreagte all the variables that were found downstream
            agg_downstream = {k:sum(d[k] for d in downstream_vars) for k in downstream_vars[0]}  
            # Pull out the aggregated values
            demand += agg_downstream['demand']
            cost += agg_downstream['cost']

        # Return a dictionary of accumulated values
        return {'demand': demand, 'cost': cost}

    def output(self, path):

        out_results = 'sequenced-results.csv'
        out_shp = 'sequenced-network'
        
        # Write the shapefiles to path
        nx.write_shp(self.networkplan.network, os.path.join(path, out_shp))
        # Write the csv to path
        self.output_frame.to_csv(os.path.join(path, out_results), index=False, na_rep='NaN')
        
        # Trash the node shp files
        [os.remove(os.path.join(os.path.join(path, out_shp), x)) 
                for x in os.listdir(os.path.join(path, out_shp)) if 'node' in x]


    def _build_node_wkt(self):
        for node in self.networkplan.network.nodes():
            # Build Node WKT with Point coords
            self.networkplan.network.node[node]['Wkt'] = 'POINT ({x} {y})'.format(x=self.networkplan.coords[node][0],
                                                                                  y=self.networkplan.coords[node][0])
    def _build_edge_wkt(self):
        r = self.results
        # Iterate through the nodes and their parent
        for rank, fnode, tnode in zip(r.index, r['Sequence..Upstream.id'], r['Sequence..Vertex.id']):
            # Set the edge attributes with those found in sequencing
            self.networkplan.network.edge[fnode][tnode]['rank'] = rank
            self.networkplan.network.edge[fnode][tnode]['distance'] = self.networkplan.distance_matrix[fnode, tnode]
            self.networkplan.network.edge[fnode][tnode]['id'] = tnode
            fnode_coords = self.networkplan.coords[fnode]
            tnode_coords = self.networkplan.coords[tnode]
            
            # Build WKT Linestring with from_node and to_node coords
            self.networkplan.network.edge[fnode][tnode]['Wkt'] = 'LINESTRING ({x1} {y1}, {x2} {y2})'.format(x1=fnode_coords[0], y1=fnode_coords[1],
                                                                                                            x2=tnode_coords[0], y2=tnode_coords[1])
    
    def parent(self, n):
        parent = (parent for parent, edge in enumerate(self.networkplan.adj_matrix[:, n]) if edge)
        # Fake nodes will have no parent
        return next(parent, None)

    def _progress_meter(self, progress):
        # Clear the line
        sys.stdout.write('\r')
        
        # Divide the sequence progress by the number of nodes in the network minus the fakes
        completed = 1.0 * progress / len(self.networkplan.network.nodes())

        meter_ticks = completed * 100
        # At every 2.5% completion update the progress bar
        # If the percentage is less than 50% update the ticks before the percentage 
        before = ''.join('##' if int(meter_ticks) >  x else '  ' for x in np.arange(02.5,  50, 2.5))
        # If the percentage is greater than 50% update the ticks after the percentage 
        after  = ''.join('##' if int(meter_ticks) >= x else '  ' for x in np.arange(52.5, 100, 2.5))
        # Update the progress bar
        sys.stdout.write('[{b} {prog:.2f}% {a}]'.format(b    = before, 
                                               	        prog = np.around(meter_ticks, 2), 
                                              	        a    = after))
        sys.stdout.flush()
        
        return completed
    
    def _clean_results(self):
        """This joins the sequenced results on the metrics dataframe and reappends the dropped rows"""
        
        logger.info('Joining Sequencer Results on Input Metrics')
        
        #ToDo: This is INSANELY expensive, need to work on an alternative routine
        
        orig = pd.read_csv(self.networkplan.csv_p, header=1)
        orig.columns = parse_cols(orig)
        non_xy_cols = orig.columns - ['coords', 'X', 'Y']
        self.networkplan.metrics.index.name = 'Sequence..Vertex.id'
        sequenced_metrics = pd.merge(self.networkplan.metrics.reset_index(), self.results.reset_index(), on='Sequence..Vertex.id')
        tup_cond = lambda tup: (pd.isnull(orig[tup[0]])) if type(tup[1]) is not str and np.isnan(tup[1]) \
                                                         else (sequenced_metrics[tup[0]] == tup[1])
        index = lambda row: list(sequenced_metrics[reduce(lambda x, y: x & y, map(tup_cond, row.iteritems()))].index.values)
        rec_index = [index(row) for row in zip(*orig[non_xy_cols].T.iteritems())[1] if index(row)]
        joined = pd.concat([sequenced_metrics, orig.ix[list(zip(*rec_index)[0])]])
        joined['coords'] = joined.apply(lambda df: (df['X'], df['Y']), axis=1)
        
        self.output_frame = joined
        logger.info('DONE!')
        
        sorted_columns = orig.columns.tolist() + list(set(sequenced_metrics.columns) - set(orig.columns))
        self.output_frame = joined[sorted_columns]

    def nodal_demand(self, df):
        """Overload this method to compute your nodal demand"""
        raise NotImplemented()
