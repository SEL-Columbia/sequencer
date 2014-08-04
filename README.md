NetworkPlan
===========

This module aims to replace [NetworkPlanner.R](https://github.com/sel-columbia/networkplanner.R), with an object 
oriented pythonic implementation. The python version also depends on NetworkX as opposed to the R counterparts igraph 
module. Furthermore, I hope to transition many of the graph algorithms to the scipy csgraph (adjacency matrix based) 
module for improved performance.

``` python
from sequencer import NetworkPlan, Sequencer

class DemoSequencer(Sequencer):
    """Demo Sequencer, to demonstrate the module"""

    # Overload the nodal_demand function with your metric
    def nodal_demand(self, df):
        # Nonsensical demand function
        return df['P_dem_ho'] + df['Metric > Maximum length of medium voltage line extension']

if __name__ == '__main__':
    
    csv = '/Users/blogle/Downloads/1643/metrics-local.csv'
    shp = '/Users/blogle/Downloads/1643/networks-proposed.shp'
    
    # Initialize a NetworkPlan from NetworkPlanner shapefile and csv
    mynetworkplan = NetworkPlan(shp, csv)
    # Initialize the Sequencer with the NetworkPlan
    mysequencer = DemoSequencer(mynetworkplan)

    print list(mysequencer.sequence())
```
