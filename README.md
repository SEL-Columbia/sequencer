# Sequencer

This module aims to assign a priority or order to the components of a network for planning purpose.  This is done by converting the inputs into a directed acyclic graph and ordering the nodes by the cumulative "downstream" values defined by the input parameters.  

See [NetworkPlanner.R](https://github.com/sel-columbia/networkplanner.R) for similar work, but in R.  

## Development Setup

Clone this repository and cd into the directory

Setup a virtual environment (example here uses [anaconda](https://docs.continuum.io/anaconda/index))

```bash
conda create -n sequencer --file requirements.txt python=2.7
source activate sequencer
```

The repo contains a sample dataset in the data folder.  
You can run a test against it via the command line

```bash
./run_sequencer.py --metrics_path data/sumaila/input/metrics-local.csv --network_path data/sumaila/input/networks-proposed.shp --demand_field "Demand...Projected.nodal.demand.per.year" --prioritize_field "Population" --output_path output
```

Take a look at `run_sequencer.py` for an example of how to use the modules from Python

## Output

The main output is a table with the original metrics csv data (i.e. the nodes) plus "Sequence" fields as shown below

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

