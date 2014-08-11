# -*- coding: utf-8 -*-
__author__ = 'Brandon Ogle'

import pandas as pd
import networkx as nx

def memoize(f):
    cache = {} 
    def memoizedFunction(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]

    memoizedFunction.cache = cache
    return memoizedFunction

class Sequencer(object):
    
    def __init__(self, NetworkPlan):
        self.networkplan = NetworkPlan
    
    def sequence(self):
        
        # Create a column containing the computed demand
        self.networkplan.metrics['nodal_demand'] = self.nodal_demand(self.networkplan.metrics)

        network = self.networkplan.network_to_dict()
        frontier = network.keys()
        
        while frontier:
            max_ = choice = None
            for node in frontier:
                demand = self.accumulate(node)
                if demand > max_:
                    max_ = demand 
                    choice = node
            
            for item in network.pop(choice):
                if type(item) != int:
                    network.update(item) 
                else:
                    network.update({item:[]})
            
            frontier = network.keys()
            yield choice
    
    @memoize
    def accumulate(self, n):
        """computes the aggregate downstream_demand"""
        demand = self.networkplan.metrics['nodal_demand'].ix[n]
        downstream_demand = sum([self.accumulate(child) for child, edge in 
                            enumerate(self.networkplan.adj_matrix[n, :]) if edge])
        return demand + downstream_demand if downstream_demand else demand

    def nodal_demand(self, df):
        """Overload this method to compute your nodal demand"""
        raise NotImplemented()

