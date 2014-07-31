import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as graph
from sklearn.neighbors import DistanceMetric
import pandas as pd

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
    
    def __init__(self, shp, csv):
        # Load in metrics csv
        self._metrics = pd.read_csv(csv, header=1)
        # ToDo: need better way of aligning shp and csv data!!!
        # Round coordinates to 12 digits
        self._metrics[['X','Y']] = np.around(self._metrics[['X','Y']], 12)
        # Set coordinates to index
        self._metrics = self._metrics.set_index(['X', 'Y'])
        
        # Load in the shape file
        self._network = nx.read_shp(shp)
        # Pull the shape file coordinates
        coords = self._network.nodes()
        # Change the node id to integers
        self._network = nx.convert_node_labels_to_integers(self._network)
        # Make bidirectional edges
        self._network = self._network.to_undirected()
        self._network = self._network.to_directed()
        
        # zip the node ids with their coords
        coords = dict(zip(self._network.nodes(), coords))
        # build a dictionary of nodes to names
        names = dict()
        for k in coords.keys():
            name = self._get_node_name(coords[k])
            if not name.empty:
                names[k] = name[0]
            else:
                # Set fields with no matching index to FAKE
                names[k] = 'FAKE'

        # Set network node attrs for names and coords
        nx.set_node_attributes(self._network, 'names', names)
        nx.set_node_attributes(self._network, 'coords', coords)
        self._weight_edges()
        
        # Transform edges to a rooted graph
        self.direct_network()
        
    @property
    def _distance_matrix(self):
        """Returns the computed distance matrix"""
        # Todo: add if projection is utm then use 'euclidean' else 'haversine'
        metric = DistanceMetric.get_metric('haversine')
        return metric.pairwise(self.coords.values(), self.coords.values())
        
    def _get_node_name(self, x):
        """Returns the name of a node given the integer representation"""
        proposed = self.metrics[self.metrics['Metric > System'] == 'grid']
        return proposed.ix[np.around(x, 12)]['Name']

    def _get_node_attr(self, name, attr):
        """Returns an attribute value from the metrics dataframe"""
        return self.metrics[self.metrics['Name'] == name][attr]
    
    def _get_components(self):
        """returns the components from a directed graph"""
        return nx.weakly_connected_component_subgraphs(self.network)
    
    def _depth_first_directed(self, graph):
        """Transforms a networks edges to direct away from the root""" 
        # Todo: refactor to csgraph traversal to improve performance
        old_edges = graph.edges()
        dfs_edges = list(nx.traversal.dfs_edges(graph,
                        #Todo: fix to use graph root
                        np.random.choice(graph.nodes())))
        graph.remove_edges_from(old_edges)
        graph.add_edges_from(dfs_edges)
        return graph
    
    def direct_network(self):
        """Decomposes a full graph into its components and directs them away from their roots"""
        graphs = [self._depth_first_directed(g) for g in self._get_components()]
        self._network = reduce(lambda a, b: nx.union(a, b), graphs)
        
    def downstream(self, n):
        """recursively builds a dictionary of child nodes from the input node"""
        children = [self.downstream(node) for node, edge in 
                    enumerate(self.adj_matrix[n, :]) if edge]
        return {n : children} if children else n
    
    def network_to_dict(self):
        """returns a dictionary representation of the full graph"""
        return reduce(lambda x,y: x.update(y) or x, 
                      [self.downstream(root) for root in nwp.roots])
    
    def _weight_edges(self):
        """sets the edge weights in the graph using the distance matrix"""
        weights = {}
        for edge in self.network.edges():
            weights[edge] = self._distance_matrix[edge]
        nx.set_edge_attributes(self.network, 'weight', weights)
            
    @property
    def roots(self):
        """Returns the root nodes in the graph"""
        # ToDo: use fake nodes or max population in subgraph
        return [k for k,v in self.network.in_degree().iteritems() if v == 0]

    @property
    def coords(self):
        """returns the nodal coordinants"""
        return nx.get_node_attributes(self.network, 'coords')
    
    @property
    def node_names(self):
        """returns the node names"""
        return nx.get_node_attributes(self.network, 'names')
    
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

    @property
    def fake_nodes(self):
        """returns a list of fake nodes in the graph"""
        return [k for k, v in self.node_names.iteritems() if v == 'FAKE']