# -*- coding: utf-8 -*-utf
__author__ = 'Brandon Ogle'

import fiona
import numpy as np
import networkx as nx
import pandas as pd
import logging
import copy

from sequencer.Utils import prep_data, haversine_distance, euclidean_distance

logger = logging.getLogger('sequencer')

class NetworkPlan(object):
    """
    NetworkPlan containing NetworkPlanner proposed network and 
    accompanying nodal metrics
    
    """
    TOL = .5 # meters at the equator, tolerance is stricter towards the poles

    def __init__(self, network, metrics, **kwargs):
        self.priority_metric = kwargs.get('prioritize', 'population')
        self.proj = kwargs.get('proj', 'longlat')

        # FIXME:
        # Remove the dependency that sequencer has on the
        # original metrics file (this is terrible coupling)
        # see sequencer:_clean_results()
        self._original_metrics = metrics
        
        self._init_helper(network, metrics)

    def _init_helper(self, network, metrics):
        """
        All initialization (cleaning up metrics, network, etc)
        """

        # Load in and align input data
        logger.info('Aligning Network Nodes With Input Metrics')
        self._network, self._metrics = prep_data(network, 
                                                 metrics, 
                                                 loc_tol = self.TOL)

        self.coord_values = self.coords.values()

        # Set the edge weight to the distance between those nodes
        self._weight_edges()
        
        logger.info('Directing Network Away From Roots')
        # Transform edges to a rooted graph
        self.direct_network()

        # Assert that the netork is a tree
        self.assert_is_tree()

        # Build list of fake nodes
        self.fake_nodes = self.fakes(self.metrics.index)

        #Fillna values with Zero
        self._metrics = self.metrics.fillna(0)

    @classmethod 
    def from_files(cls, shp, csv, **kwargs):
        """
        Parameters
        ----------
        shp : file or string (File, directory, or filename to read).
        csv : string or file handle / StringIO.
        
        Example
        ----------
        NetworkPlan.from_files('networks-proposed.shp', 
                               'metrics-local.csv')
        """

        logger.info('Asserting Input Projections Match')

        # cls._assert_proj_match(shp, csv)
        # TODO: Developer should transform shapefile projection to match csv
        # Use fiona to open the shapefile as this includes the projection type
        
        # shapefile = fiona.open(shp)
        # Pass along the projection
        # if 'proj' in shapefile.crs:
            # kwargs['proj'] = shapefile.crs['proj']
 
        return cls(nx.read_shp(shp), pd.read_csv(csv), **kwargs)

    @classmethod
    def _assert_proj_match(self, shp, csv):
        """Ensure that the projections match before continuing"""
        # Use fiona to open the shapefile as this includes the projection type
        shapefile = fiona.open(shp)
        # read the first line of the csv to get the projection
        csv_proj = open(csv).readline()

        # Parse the Projection to a dictionary
        csv_proj = csv_proj.split('PROJ.4')[1]
        pairs = [x.split('=') for x in csv_proj.split(' +') if x != '' and '=' in x]
        csv_proj = {x[0]:x[1] for x in pairs}
        # Iterate through and ensure all values match in both projection dicts
        for key in csv_proj.keys():
            if key in csv_proj and key in shapefile.crs:
                try:
                    assert(str(csv_proj[key]) == str(shapefile.crs[key]))
                except:
                    logger.error("csv and shp Projections Don't Match")
                    raise AssertionError("csv and shapefile Projections Don't Match")

    def assert_is_tree(self):
        in_degree = self.network.in_degree()
        # Test that all roots have in_degree == 0
        ensure_roots = [in_degree[root] == 0 for root in self.roots]
        # Test that all leaves have in_degree == 1
        ensure_leaves = [in_degree[leaf] == 1 for leaf in (set(self.network.node.keys()) - set(self.roots))]
        
        assert(all(ensure_roots + ensure_leaves) == True)

    def _get_node_attr(self, node, attr):
        """Returns an attribute value from the metrics dataframe"""
        return self.metrics[attr].ix[node]
       
    def _depth_first_directed(self, graph):
        """Transforms a networks edges to direct away from the root"""
        
        # Figure out which subgraph this is
        sub = next((i+1 for i, g in enumerate(self.get_subgraphs()) if g==graph), None)
        # Log the Subgraph progress
        logger.info('Directing SUBGRAPH {} / {}'.format(sub, len(list(self.get_subgraphs()))))

        old_edges = graph.edges()
        dfs_edges = list(nx.traversal.dfs_edges(graph,
                        self._graph_priority(graph.nodes())))
        #This debug message could be cleaner
        logger.debug('mapping {} -> {}'.format(old_edges, dfs_edges))
        graph.remove_edges_from(old_edges)
        graph.add_edges_from(dfs_edges)
        
        logger.info('DONE!')
        return graph

    def fakes(self, nodes):
        """applies a filter to the input nodes, returning the subset representing fake nodes"""

        # get a view of the DataFrame without positional columns
        non_positional = self.metrics[self.metrics.columns - ['X', 'Y', 'coords', 'm_coords']].ix[nodes]
        # find rows that are all null, these are the nodes representing the connection to existing infastructure
        return non_positional[np.all(pd.isnull(non_positional) == True, axis=1)].index.values
   
    def _graph_priority(self, nodes):
        """returns the starting node to be used in directing the graph"""
       
        fakes = self.fakes(nodes)
        
        # There theoretically should only be one fake per subgraph
        if len(fakes) == 1:
            return fakes[0]

        # If for some reason there is more, its likely due to poor indexes and just pick one
        elif len(fakes) > 1:
            logger.error('More Than One Fake Node In Subgraph {}, \
                         Paths between fake nodes results in unreliable sequence!'.format(fakes))
            return np.random.choice(fakes)

        # If there is no fake node in the subgraph, its not close to infastructure and thus priority is given to MAX(priority metric)
        else:
            return self.metrics[self.priority_metric].ix[nodes].idxmax()

    def _distance(self, first_index, second_index):
        """Calculate the distance between two points given their indices."""
        distance_function = (
            haversine_distance if 'longlat' in self.proj else
            euclidean_distance
        )
        return distance_function(
            self.coord_values[first_index], self.coord_values[second_index]
        )

    def _weight_edges(self):
        """Set the edge weights in the graph using a distance function."""
        weights = {}
        logger.info('Using {} distance'.format(
            'haversine' if 'longlat' in self.proj else 'euclidean'
        ))

        no_bad_edges_found = True
        for edge in self.network.edges():
            distance = self._distance(edge[0], edge[1])
            if no_bad_edges_found and distance < self.TOL:
                no_bad_edges_found = False
                logger.error(
                    'Dataset contains edges less than {tolerance} meters! This'
                    ' can result in incorrect alignment of metrics and'
                    ' network, where fake nodes are incorrectly assigned'
                    'metrics. This error is resolved by buffering your input'
                    ' data.'.format(tolerance=self.TOL)
                )
            weights[edge] = distance
        nx.set_edge_attributes(self.network, 'weight', weights)
    
    def direct_network(self):
        """Decomposes a full graph into its components and directs them away from their roots"""
        #print list(self.get_subgraphs())
        graphs = [self._depth_first_directed(g) for g in self.get_subgraphs()]
        self._network = reduce(lambda a, b: nx.union(a, b), graphs)
        
    def downstream(self, n):
        """
        Builds a nested dict of downstream nodes for input node n
        """

        return self._downstream_helper(n)

    def _downstream_helper(self, n):
        """
        recursively builds a dictionary of child nodes from the input node
        
        """
        children = [self._downstream_helper(node) 
                    for node in self.get_successors(n)]
        return {n : children} if children else {n : []}

    def root_child_dict(self):
        root_child = {}
        for subgraph in self.get_subgraphs():
            nodes_degree = subgraph.in_degree()
            subgraph = copy.deepcopy(subgraph.node)
            for node, degree in nodes_degree.iteritems():
                if degree == 0:
                    break
            assert(nodes_degree[node] == 0)
            subgraph.pop(node)
            root_child[node] = subgraph.keys()
        return root_child

    def _get_subgraphs(self):
        self.subgraphs = list(nx.weakly_connected_component_subgraphs(self.network))

    def get_subgraphs(self):
        """returns the components from a directed graph"""
        if hasattr(self, 'subgraphs') is False:
            self._get_subgraphs()
        for sub in self.subgraphs:
            yield sub

    def get_predecessors(self, n):
        """wrap networkx call"""
        return self._network.predecessors(n)

    def get_successors(self, n):
        """wrap networkx call"""
        return self._network.successors(n)

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
        return nx.adj_matrix(self.network)
    
    @property
    def network(self):
        """returns the DiGraph Object representation of the graph"""
        return self._network
    
    @property
    def original_metrics(self):
        """returns the original (unprocessed) metrics data_frame"""
        return self._original_metrics

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
        logger.warn("Private scenario requires both username and password!" +
                        "Authentication for public scenarios can be omitted.")
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

    return NetworkPlan.from_files(shp, csv)
