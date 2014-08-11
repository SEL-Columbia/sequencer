NetworkPlanner Sequencer
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

    # Sequences the network returning a dataframe
    print mysequencer.sequence()
```

<div class="output" style=""><div class="output_area"><div class="prompt output_prompt">Out[7]:</div><div class="output_subarea output_html rendered_html output_pyout"><div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>demand</th>
      <th>distance_from_parent</th>
      <th>metric</th>
      <th>node</th>
    </tr>
    <tr>
      <th>rank</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1  </th>
      <td> 1311992.655328</td>
      <td> 1.000000</td>
      <td> 1311992.655328</td>
      <td>  31</td>
    </tr>
    <tr>
      <th>2  </th>
      <td>  193245.823512</td>
      <td> 0.301721</td>
      <td>  640478.303476</td>
      <td>  67</td>
    </tr>
    <tr>
      <th>3  </th>
      <td>  906296.794866</td>
      <td> 1.686770</td>
      <td>  537297.057125</td>
      <td> 104</td>
    </tr>
    <tr>
      <th>4  </th>
      <td>  903510.987999</td>
      <td> 0.367407</td>
      <td> 2459154.571464</td>
      <td>   6</td>
    </tr>
    <tr>
      <th>5  </th>
      <td>  900725.181132</td>
      <td> 1.856837</td>
      <td>  485085.744979</td>
      <td>  56</td>
    </tr>
    <tr>
      <th>6  </th>
      <td>  794685.254751</td>
      <td> 0.990892</td>
      <td>  801989.450144</td>
      <td>  15</td>
    </tr>
    <tr>
      <th>7  </th>
      <td>  792672.077095</td>
      <td> 2.471885</td>
      <td>  320675.207324</td>
      <td>  99</td>
    </tr>
    <tr>
      <th>8  </th>
      <td>  547291.178251</td>
      <td> 2.835366</td>
      <td>  193023.145854</td>
      <td>  21</td>
    </tr>
    <tr>
      <th>9  </th>
      <td>  545588.312152</td>
      <td> 1.031583</td>
      <td>  528884.442069</td>
      <td>  40</td>
    </tr> 
  </tbody>
</table>
<p>132 rows × 4 columns</p>
</div></div></div></div>
