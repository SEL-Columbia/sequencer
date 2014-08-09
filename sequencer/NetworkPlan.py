# -*- coding: utf-8 -*-
__author__ = 'Brandon Ogle'

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as graph
from sklearn.neighbors import DistanceMetric
import pandas as pd

from sequencer.Utils import prep_data

class NetworkPlan(object):
    """
    NetworkPlan containing NetworkPlanner proposed network and 
    accompanying nodal metrics
    
    Parameters
    ----------
    shp : file or string (File, directory, or filename to read).
    csv : string or file handle / StringIO.
    
    Example
    ----------
    NetworkPlan('/Users/blogle/Downloads/1643/networks-proposed.shp', 
                '/Users/blogle/Downloads/1643/metrics-local.csv')
    """
    
    def __init__(self, shp, csv, **kwargs):
         
        self.priority_metric = kwargs['prioritize'] if 'prioritize' in kwargs else 'population'

        # Load in and align input data
        self._network, self._metrics = prep_data( nx.read_shp(shp),
                                                  pd.read_csv(csv, header=1), 
                                                  prec = kwargs['precision'] 
                                                  if 'precision' in kwargs else 8
                                                )
        
        # Set the edge weight to the distance between those nodes
        self._weight_edges()
        
        # Transform edges to a rooted graph
        self.direct_network()
        
    @property
    def _distance_matrix(self):
        """Returns the computed distance matrix"""
        # TODO: add if projection is utm then use 'euclidean' else 'haversine'
        metric = DistanceMetric.get_metric('haversine')
        return metric.pairwise(self.coords.values(), self.coords.values())
        
    def _get_node_attr(self, node, attr):
        """Returns an attribute value from the metrics dataframe"""
        return self.metrics[attr].ix[node]
       
    def _depth_first_directed(self, graph):
        """Transforms a networks edges to direct away from the root""" 
        old_edges = graph.edges()
        dfs_edges = list(nx.traversal.dfs_edges(graph,
                        self._graph_priority(graph.nodes())))
        graph.remove_edges_from(old_edges)
        graph.add_edges_from(dfs_edges)
        return graph
    
    def _graph_priority(self, nodes):
        """returns the starting node to be used in directing the graph"""
        
        # get a view of the DataFrame without positional columns
        non_positional = self.metrics[self.metrics.columns - ['X', 'Y', 'coords']].ix[nodes]
        # find rows that are all null, these are the nodes representing the connection to existing infastructure
        fakes = non_positional[np.all(pd.isnull(non_positional) == True, axis=1)].index
        
        # There theoretically should only be one fake per subgraph
        if len(fakes) == 1:
            return fakes

        # If for some reason there is more, its likely due to poor indexes and just pick one
        elif len(fakes) > 1:
            print Warning('More than one fake node in subgraph, something may have gone horribly in aligning your data!')
            return np.random.choice(fakes)

        # If there is no fake node in the subgraph, its not close to infastructure and thus priority is given to MAX(priority metric)
        else:
            return self.metrics[self.priority_metric].ix[nodes].idxmax()

    def _weight_edges(self):
        """sets the edge weights in the graph using the distance matrix"""
        weights = {}
        for edge in self.network.edges():
            weights[edge] = self._distance_matrix[edge]
        nx.set_edge_attributes(self.network, 'weight', weights)
    
    def direct_network(self):
        """Decomposes a full graph into its components and directs them away from their roots"""
        graphs = [self._depth_first_directed(g) for g in self.get_subgraphs()]
        self._network = reduce(lambda a, b: nx.union(a, b), graphs)
        
    def downstream(self, n):
        """recursively builds a dictionary of child nodes from the input node"""
        children = [self.downstream(node) for node, edge in 
                    enumerate(self.adj_matrix[n, :]) if edge]
        return {n : children} if children else n

    def get_subgraphs(self):
        """returns the components from a directed graph"""
        return nx.weakly_connected_component_subgraphs(self.network)
    
    def network_to_dict(self):
        """returns a dictionary representation of the full graph"""
        return reduce(lambda x,y: x.update(y) or x, 
                      [self.downstream(root) for root in self.roots])         
    @property
    def roots(self):
        return [n for n, k in self.network.in_degree().iteritems() if k == 0]

    @property
    def coords(self):
        """returns the nodal coordinants"""
        return nx.get_node_attributes(self.network, 'coords')
        
    @property
    def adj_matrix(self):
        """returns the matrix representation of the graph"""
        return nx.adj_matrix(self.network).toarray()
    
    @property
    def network(self):
        """returns the DiGraph Object representation of the graph"""
        return self._network
    
    @property
    def metrics(self):
        """returns the nodal metrics Pandas DataFrame"""
        return self._metrics


def download_scenario(scenario_number, directory_name=None, username=None, password=None,
                      np_url='http://networkplanner.modilabs.org/'):

    # TODO: Figure out how to handle case that user didn't give login info
    # but the SCENARIO happens to be PRIVATE, can't think of a way to validate
    # until the zip file is downloaded.

    # If no dir specified, dump data to working directory
    if directory_name is None:
        directory_name = str(os.getcwd()) + '/' + str(scenario_number) + '/'

    # Standardize/ convert to absolute path
    directory_name = os.path.abspath(directory_name)

    # Create a Boolean flag indicating if the repo if private
    # error handling for only 1 NULL value for user & pass
    private = all([username is not None, password is not None])

    # If the scenario is public, yet there is a credential raise exception
    if not private and any([username is not None, password is not None]):
        raise Exception("Private scenario requires both username and password!" +
                        "Authentication for public scenarios can be omitted.")

    # Reconstructing url for the zip file
    full_url = np_url + 'scenarios/' + str(scenario_number) + '.zip'

    with requests.Session() as s:
        # If it is a private repo, then login to network planner
        if private:
            # Go to the login page
            login_page = np_url + "people/login_"
            # Send the login credentials
            payload = {'username': username, 'password': password}
            s.post(login_page, data=payload)
            print 'LOGGING IN...'

        scenario_data = s.get(full_url)

    # Read in the zipfile contents
    zip_folder = ZipFile(StringIO(scenario_data.content))

    def write_file(name):
        content = zip_folder.read(name)
        path = os.path.join(directory_name, name)
        subdir = '/'.join(path.split('/')[:-1])
        # Build directory should it not exist
        if not os.path.exists(subdir):
            print('creating {dir}'.format(dir=subdir))
            os.makedirs(subdir)

        # Open the file and write the zipped contents
        with open(path, 'wb') as f:
            f.write(content)

    # Write all the zipped files to disk
    map(write_file, zip_folder.namelist())

    csv = os.path.join(directory_name, 'metrics-local.csv')
    shp = os.path.join(directory_name, 'network-proposed.shp')

    return NetworkPlan(shp, csv)
