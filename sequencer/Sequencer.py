import pandas as pd
import networkx as nx

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

    def _graph_to_nested_list(self, obj):
        """converts a dictionary representation of a network to a list"""
        if type(obj) is dict:
            return [obj.keys()] + self._graph_to_nested_list(obj.values())
        elif type(obj) is list:
                return [self._graph_to_nested_list(item) for item in obj]
        else:
            return [obj]
        
    def _flatten_list(self, nested_list):
        """Flatten an arbitrarily nested list."""
        while nested_list:
            sublist = nested_list.pop(0)

            if isinstance(sublist, list):
                nested_list = sublist + nested_list
            else:
                yield sublist
    
    def flatten_graph(self, network):
        return self._flatten_list(self._graph_to_nested_list(network))
        
    def sequence(self):
        # Todo: pass accumulate only roots when refactored for recursive call
        demand = {node: self.accumulate(node) for node in self.networkplan.network.nodes()}
        network = self.networkplan.network_to_dict()
        frontier = network.keys()
        
        while frontier:
            max_ = choice = None
            for node in frontier:
                if demand[node] > max_:
                    max_ = demand[node]
                    choice = node
                    
            [network.update({item:[]}) if type(item) == int else network.update(item) 
             for item in network.pop(choice)]
            
            frontier = network.keys()
            yield choice
    
    def accumulate(self, n):
        """computes the aggregate downstream demand"""
        # Todo: make a recursive call to prevent duplicate calculations
        return sum([self.networkplan.network.node[node]['nodal_demand'] 
         for node in list(self.flatten_graph(self.networkplan.downstream(n)))]) 
