NetworkPlanner Sequencer
===========

This module aims to replace [NetworkPlanner.R](https://github.com/sel-columbia/networkplanner.R), with an object 
oriented pythonic implementation. The python version also depends on NetworkX as opposed to the R counterparts igraph 
module. 

#Quickstart
if you want to quickly get up and running, grab an electrification scenario from 
[NetworkPlanner](http://networkplanner.modilabs.org/scenarios) and follow along with the provided demo_sequencer.py

The required toolset for this example is the NetworkPlan object and a Sequencer Model

```python 
from sequencer import NetworkPlan
from sequencer.Models import EnergyMaximizeReturn
```

The NetworkPlan object takes a path to a shapefile containing nodes and edges and a corresponding metrics csv. 
```python
csv = '/Users/blogle/Downloads/3305/metrics-local.csv'
shp = '/Users/blogle/Downloads/3305/networks-proposed.shp'
```

Upon instantiating this object, the nodes are joined on the metrics file and all subgraphs (or components) are 
directed away from their roots. 'Fake' nodes are placeholders with no corresponding metric data used to 
represent the position of the existing infastructure (in this case a grid). If a subgraph contains a 'Fake' node, 
this is chosen as its root, otherwise its root is determined by prioritizing some metric.

```python
nwp = NetworkPlan(shp, csv, prioritize='Population')
```

The above NetworkPlan will direct away from the grid, or if the subgraph has no connection to the grid it will direct
away from the most populated nodes. The prioritize keyword argument defaults to 'population', so if this field is not 
in your metrics you will need to pass it an alternative parameter.

```python
model = EnergyMaximizeReturn(nwp)
model.sequence()
```

Finally instantiate the model with the NetworkPlan and call sequence. This computes a frontier of possible nodes 
accessible from the roots, and iteratively traveses the tree optimizing some heuristic. The sequence method builds a 
DataFrame containing the results of the traversal, namely the sequence rank for all nodes. 

Note: The results dataframe is accesible through```model.results```
<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide" style=""></div><div class="output" style=""><div class="output_area"><div class="prompt output_prompt">Out[12]:</div><div class="output_subarea output_html rendered_html output_pyout"><div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sequence..Decision.metric</th>
      <th>Sequence..Downstream.demand.sum.kwh</th>
      <th>Sequence..Downstream.distance.sum.m</th>
      <th>Sequence..Root.vertex.id</th>
      <th>Sequence..Upstream.id</th>
      <th>Sequence..Upstream.segment.distance.m</th>
      <th>Sequence..Vertex.id</th>
    </tr>
    <tr>
      <th>Sequence..Far.sighted.sequence</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1  </th>
      <td> 2780868.271316</td>
      <td> 28013000</td>
      <td>    10.073472</td>
      <td>  27</td>
      <td>  27</td>
      <td>   10.073472</td>
      <td>  25</td>
    </tr>
    <tr>
      <th>2  </th>
      <td>    7010.936648</td>
      <td> 44720000</td>
      <td>  6378.605633</td>
      <td> 505</td>
      <td> 505</td>
      <td>   16.171168</td>
      <td> 391</td>
    </tr>
    <tr>
      <th>3  </th>
      <td>     960.761193</td>
      <td>  1217000</td>
      <td>  1266.703952</td>
      <td> 392</td>
      <td> 392</td>
      <td>  537.213092</td>
      <td> 342</td>
    </tr>
    <tr>
      <th>4  </th>
      <td>     452.370301</td>
      <td>   330000</td>
      <td>   729.490860</td>
      <td> 392</td>
      <td> 342</td>
      <td>  155.021075</td>
      <td> 304</td>
    </tr>
    <tr>
      <th>5  </th>
      <td>     464.776402</td>
      <td>   267000</td>
      <td>   574.469785</td>
      <td> 392</td>
      <td> 304</td>
      <td>  574.469785</td>
      <td> 212</td>
    </tr>
    <tr>
      <th>...  </th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>519</th>
      <td>      15.164609</td>
      <td>   165000</td>
      <td> 10880.596829</td>
      <td> 424</td>
      <td> 210</td>
      <td> 6113.945502</td>
      <td> 228</td>
    </tr>
    <tr>
      <th>520</th>
      <td>      21.818252</td>
      <td>   104000</td>
      <td>  4766.651327</td>
      <td> 424</td>
      <td> 228</td>
      <td> 4766.651327</td>
      <td> 421</td>
    </tr>
    <tr>
      <th>521</th>
      <td>      13.653252</td>
      <td>   219000</td>
      <td> 16040.134490</td>
      <td> 424</td>
      <td> 210</td>
      <td> 3982.017803</td>
      <td> 486</td>
    </tr>
    <tr>
      <th>522</th>
      <td>      12.025095</td>
      <td>   145000</td>
      <td> 12058.116687</td>
      <td> 424</td>
      <td> 486</td>
      <td> 6212.158996</td>
      <td> 325</td>
    </tr>
    <tr>
      <th>523</th>
      <td>      11.460911</td>
      <td>    67000</td>
      <td>  5845.957691</td>
      <td> 424</td>
      <td> 325</td>
      <td> 5845.957691</td>
      <td>  42</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>


```python 
model.output('/Users/blogle/Downloads/3305/output/')
```
This will output a shapefile of edges and a csv containing the outer join of sequence results and the input metrics.

#Sequencer Models

There is a Sequencer base class to build your own models off of. After subclassing Sequencer, you must overload the 
nodal_demand method. This method should return a linear combination of n-columns in the input metrics that is then 
appended to Sequencer.NetworkPlan.metrics, that the sequencer will use to optimize the network. For an example of how
to roll out your own model, see sequencer/Models.py for examples.

```python 
class EnergyMaximizeReturn(Sequencer):
    """This class sequences a proposed electrification plan, optimizing for maximum Demand (kwh) / Distance (m)"""
    
    def nodal_demand(self, df):
        return df['Demand...Projected.nodal.demand.per.year']
    
    def _strip_cols(self):
      del (self.output_frame['nodal_demand'])

    def sequence(self):
        super(EnergyMaximizeReturn, self).sequence()
        self._strip_cols()
```

#Hierarchial Representaion
```python
import networkx as nx
figsize(30, 30)
pos = nx.pygraphviz(nwp.network, 'dot')
nx.draw(nwp.network, pos)
nx.draw_networkx_labels(nwp.network, pos)
```

![a link](https://github.com/SEL-Columbia/Sequencer/blob/master/Network.png)

##Installation 

The Sequencer depends on several modules including the scipy stack, Networkx and some various other libraries.
The required packages can be found in the requirements.txt, you can install them by hand or take advantage of your 
package manager. 

##Anaconda

The simplest way to setup an environment is via [anaconda](https://docs.continuum.io/anaconda/index).  
You'll need the python2.7 based install available [here](https://www.continuum.io/downloads).

You'll need to make sure anaconda is in your PATH before running conda commands.  

### For running

Setting up Sequencer for running (i.e. not for development) can be done via:
```bash
conda create -n [your env] -c sel sequencer
```

### For development
You can easily create a new env with the listed packages by running
```bash
conda create -n [your env] --file requirements.txt python=2.7
```

You can build the package using the [recipe](https://github.com/SEL-Columbia/conda-recipes)

If you plan to play with the sequencer interactively and want to output graphics like those above, you will need 
Pygraphviz. Pygraphviz is not in the default channel for conda however it can be pulled from binstar.org 
and built with 

```bash 
conda install -c https://conda.binstar.org/mhworth pygraphviz  
```
