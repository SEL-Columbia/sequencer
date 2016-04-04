from sequencer import NetworkPlan
from sequencer.Models import EnergyMaximizeReturn

from os.path import join
source_folder = 'data/sumaila/input'

csv = join(source_folder, 'metrics-local.csv')
shp = join(source_folder, 'networks-proposed.shp')

nwp = NetworkPlan(shp, csv, prioritize='Population')
model = EnergyMaximizeReturn(nwp)

results = model.sequence()
target_folder = 'xyz'

from os import makedirs
try:
    makedirs(target_folder)
except OSError:
    pass

model.output(target_folder)
