class Sequencer(object):
    
    def __init__(NetworkPlan, Sequence):
        self.networkplan = NetworkPlan
        
    def nodal_demand(self):
        raise NotImplemented()
    
    def accumulate(self):
        raise NotImplemented()