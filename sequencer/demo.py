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

    print list(mysequencer.sequence()),  '\n'
    print mynetworkplan.network_to_dict()
