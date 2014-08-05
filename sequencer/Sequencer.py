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
        self.propagate_demand()

    def nodal_demand(self):
        raise NotImplemented()
    
    def propagate_demand(self):
        # Todo: If we ever hope to have a generic sequencer we need a 
        # better index to align on, the current implementation is hacky 
        # and requires hard coded column names making any abstracton obsolete

        # Create a column containing the computed demand
        self.networkplan.metrics['nodal_demand'] = self.nodal_demand(self.networkplan.metrics)
        # Filtering down the dataframe, to limit duplicates resulting from bad index
        is_grid = self.networkplan.metrics[self.networkplan.metrics['Metric > System'] == 'grid']
        for node in self.networkplan.network.nodes():
            node = self.networkplan.network.node[node]
            name = node['names']
            # Filter DataFrame down to results with matching name
            name_match = is_grid[is_grid['Name'] == name]
            if name == 'FAKE':
                node['nodal_demand'] = 0
            else:
               node['nodal_demand'] = name_match['nodal_demand'].values[0]

    def sequence(self):
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
        demand = self.networkplan.network.node[n]['nodal_demand']
        downstream_demand = sum([self.accumulate(child) for child, edge in 
                            enumerate(self.networkplan.adj_matrix[n, :]) if edge])
        return demand + downstream_demand if downstream_demand else demand
