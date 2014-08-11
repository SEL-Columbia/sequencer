# -*- coding: utf-8 -*-
__author__ = 'Brandon Ogle'

import pandas as pd
import networkx as nx
from functools import wraps

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

    def _sequence(self): 

        network = self.networkplan.network_to_dict()
        frontier = network.keys()
        rank = 0

        while frontier:
            max_ = choice = None
            for node in frontier:
                demand = self.accumulate(node)
                distance = self.distance(node)
                metric = 1.0 * demand / distance
                if metric > max_:
                    max_ = metric 
                    choice = node
            
            for item in network.pop(choice):
                if type(item) != int:
                    network.update(item) 
                else:
                    network.update({item:[]})
            
            frontier = network.keys()
    
            rank += 1
            yield {'rank':rank, 
                   'node':choice,
                   'demand':self.accumulate(choice), 
                   'distance_from_parent':self.distance(choice), 
                   'metric': 1.0 * self.accumulate(choice) / self.distance(choice)}
    
    @memoize 
    def accumulate(self, n): 
        """computes the aggregate downstream_demand"""
        demand = self.networkplan.metrics['nodal_demand'].ix[n]
        downstream_demand = sum([self.accumulate(child) for child, edge in 
            enumerate(self.networkplan.adj_matrix[n, :]) if edge])

        return demand + downstream_demand 

    @memoize
    def distance(self, n):
        parent = [i for i, edge in enumerate(self.networkplan.adj_matrix[:, n]) if edge]
        if parent:
            return self.networkplan._distance_matrix[parent, n][0]
        return 1

    def sequence(self):
        return pd.DataFrame(self._sequence()).set_index('rank')

    def nodal_demand(self, df):
        """Overload this method to compute your nodal demand"""
        raise NotImplemented()  
