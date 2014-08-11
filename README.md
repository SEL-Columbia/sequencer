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
    
    # Initialize a NetworkPlan from NetworkPlanner shapefile and csv,
    # also a column to prioritize (defaults to population)
    
    mynetworkplan = NetworkPlan(shp, csv, prioritize='P_dem_ho')
    
    # Initialize the Sequencer with the NetworkPlan
    mysequencer = DemoSequencer(mynetworkplan)

    print mysequencer.sequence()
```
<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide" style=""></div><div class="output" style=""><div class="output_area"><div class="prompt output_prompt"></div><div class="output_subarea output_html rendered_html output_pyout"><div style="max-height:1000px;max-width:1500px;overflow:auto;">
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
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>105</th>
      <td>    3805.787912</td>
      <td> 1.304664</td>
      <td>    2917.063744</td>
      <td>  61</td>
    </tr>
    <tr>
      <th>106</th>
      <td>    5151.838874</td>
      <td> 1.794850</td>
      <td>    2870.345763</td>
      <td>  17</td>
    </tr>
    <tr>
      <th>107</th>
      <td>    8043.271517</td>
      <td> 2.826317</td>
      <td>    2845.848824</td>
      <td>  46</td>
    </tr>
    <tr>
      <th>108</th>
      <td>    6363.240814</td>
      <td> 1.787285</td>
      <td>    3560.282343</td>
      <td>  44</td>
    </tr>
    <tr>
      <th>109</th>
      <td>    6020.821377</td>
      <td> 2.348822</td>
      <td>    2563.336854</td>
      <td> 110</td>
    </tr>
    <tr>
      <th>110</th>
      <td>    6020.821377</td>
      <td> 2.706031</td>
      <td>    2224.964097</td>
      <td> 124</td>
    </tr>
        <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>132 rows × 4 columns</p>
</div></div></div></div><div class="btn output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

#Heirarchial Representaion
```python
import networkx as nx
figsize(30, 30)
pos = nx.pygraphviz(nwp.network, 'dot')
nx.draw(nwp.network, pos)
nx.draw_networkx_labels(nwp.network, pos)
```

![a link](https://github.com/SEL-Columbia/Sequencer/blob/master/Network.png)
