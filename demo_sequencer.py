from sequencer import NetworkPlan
from sequencer.Models import EnergyMaximizeReturn

csv = '/Users/blogle/Downloads/2994/metrics-local.csv'
shp = '/Users/blogle/Downloads/2994/networks-proposed.shp'

nwp = NetworkPlan(shp, csv, prioritize='Population')
model = EnergyMaximizeReturn(nwp)

results = model.sequence()
model.output('/Users/blogle/Desktop/output/')


