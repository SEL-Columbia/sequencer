from sequencer import NetworkPlan
from sequencer.Models import EnergyMaximizeReturn

csv = 'data/sumaila/input/metrics-local.csv'
shp = 'data/sumaila/input/networks-proposed.shp'

nwp = NetworkPlan.from_files(shp, csv, prioritize='Population')
model = EnergyMaximizeReturn(nwp)

results = model.sequence()
model.output('output')
