# -*- coding: utf-8 -*-
__author__ = 'Brandon Ogle'

import pandas as pd
from functools import wraps
import networkx as nx
import numpy as np
import os

def memoize(f):
    cache = {}
    @wraps(f)
    def memoizedFunction(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]

    memoizedFunction.cache = cache
    return memoizedFunction

class Sequencer(object):
    
    def __init__(self, NetworkPlan):
        self.networkplan = NetworkPlan
        # Create a column containing the computed demand
        self.networkplan.metrics['nodal_demand'] = self.nodal_demand(self.networkplan.metrics)
        self.root_children = self.networkplan.root_child_dict()
   
    def _sequence(self):
        network = self.networkplan.network_to_dict()
        frontier = network.keys()
        rank = 0  
        while frontier:
            max_ = choice = None
            for node in frontier:
                accum_dict = self.accumulate(node)
                demand = accum_dict['demand']
                cost = accum_dict['cost']

                metric = 1.0 * demand / cost
                if metric > max_:
                    max_ = metric 
                    choice = node
                    choice_vars = accum_dict
            
            for item in network.pop(choice):
                if type(item) != int:
                    network.update(item) 
                else:
                    network.update({item:[]})
            
            frontier = network.keys()
            
            choice_row =  {
                            'node'                                : choice,
                            'Sequence..Downstream.demand.sum.kwh' : choice_vars['demand'],
                            'Sequence..Downstream.distance.sum.m' : choice_vars['cost'],
                            'Sequence..Root.vertex.id'            : self.get_root(choice),
                            'Sequence..Upstream.id'               : self.parent(choice),
                            'Sequence..Decision.metric'           : 1.0 * choice_vars['demand'] / choice_vars['cost']
                          }

            if choice_row['Sequence..Upstream.id'] is not None:
                rank += 1
                choice_row['rank'] = rank
                yield choice_row

    def sequence(self):
        self.results = pd.DataFrame(self._sequence()).set_index('rank')
        self.rank_edges()
        return self.results

    def get_root(self, n):
        # check the keys to see if the node even has an upstream root
        if n in self.root_children.keys():
            return None
        for root, children in self.root_children.iteritems():
            if n in children:
                return root
    
    @memoize
    def accumulate(self, n):
        """traverses the tree computing downstream aggregates"""

        demand = self.networkplan.metrics['nodal_demand'].ix[n]
        parent = self.parent(n)
        cost = self.networkplan.distance_matrix[parent, n] if parent else 1.0

        downstream_vars = [self.accumulate(child) for child, edge in enumerate(self.networkplan.adj_matrix[n, :]) if edge]

        if downstream_vars:
            agg_downstream = {k:sum(d[k] for d in downstream_vars) for k in downstream_vars[0]}        
            demand += agg_downstream['demand']
            cost += agg_downstream['cost']

        return {'demand': demand, 'cost': cost}

    def output(self, path):
        
        for node in self.networkplan.network.nodes():
            self.networkplan.network.node[node]['Wkt'] = 'POINT ({x} {y})'.format(x=self.networkplan.coords[node][0],
                                                                                  y=self.networkplan.coords[node][0])
        out_metrics = 'sequenced-metrics.csv'
        out_results = 'sequenced-results.csv'
        out_shp = 'sequenced-network'
        nx.write_shp(self.networkplan.network, os.path.join(path, out_shp))
        self.networkplan.metrics.to_csv(os.path.join(path, out_metrics))
        self.results.to_csv(os.path.join(path, out_results))

    def rank_edges(self):
        r = self.results
        for rank, fnode, tnode in zip(r.index, r['Sequence..Upstream.id'], r['node']):
            self.networkplan.network.edge[fnode][tnode]['rank'] = rank
            self.networkplan.network.edge[fnode][tnode]['distance'] = self.networkplan.distance_matrix[fnode, tnode]
            fnode_coords = self.networkplan.coords[fnode]
            tnode_coords = self.networkplan.coords[tnode]

            self.networkplan.network.edge[fnode][tnode]['Wkt'] = 'LINESTRING ({x1} {y1}, {x2} {y2})'.format(x1=fnode_coords[0], y1=fnode_coords[1],
                                                                                                            x2=tnode_coords[0], y2=tnode_coords[1])
    
    def parent(self, n):
        parent = [parent for parent, edge in enumerate(self.networkplan.adj_matrix[:, n]) if edge]
        return parent[0] if parent else None

    def nodal_demand(self, df):
        """Overload this method to compute your nodal demand"""
        raise NotImplemented()
