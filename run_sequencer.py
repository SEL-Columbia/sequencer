# -*- coding: utf-8 -*-
"""
Run sequencer
"""

import argparse
import json
import os
import sys
import logging
import sequencer
from sequencer import NetworkPlan, Sequencer

logger = logging.getLogger('sequencer')

logger.info(
    "sequencer %s (Python %s)" %
    (sequencer.__version__,
     '.'.join(map(str, sys.version_info[:3]))))

   
def load_arguments(value_by_key):
    """
    Merge arguments from command line with args from json config
    
    shameless copy from infrastructure-planning
    """

    configuration_path = value_by_key.pop('configuration_path')
    working_path = value_by_key.pop('working_path')

    if configuration_path:
        if working_path and not os.path.isabs(configuration_path):
            configuration_path = os.path.join(working_path, configuration_path)
        g = json.load(open(configuration_path))
    else:
        g = {}

    # Command-line arguments override configuration arguments
    for k, v in value_by_key.items():
        if v is None:
            continue
        g[k] = v

    # Resolve relative paths using source_folder
    if working_path:
        for k, v in g.items():
            if k.endswith('_path') and v and not os.path.isabs(v):
                g[k] = os.path.join(working_path, v)
    return g

parser = argparse.ArgumentParser(
    description="Run sequencer on networkplanner output"
    " (metrics csv and network shp)")
parser.add_argument("--configuration_path", "-c",
    help="json configuration file")
parser.add_argument("--working_path", "-w",
    help="base directory which all paths are relative to")
parser.add_argument("--metrics_path", "-m", 
    help="metrics csv filename")
parser.add_argument("--network_path", "-n", 
    help="network shp filename")
parser.add_argument("--demand_field", "-d",
    default="Demand...Projected.nodal.demand.per.year",
    help="field name in metrics data representing demand")
parser.add_argument("--prioritize_field", "-p",
    default="Population", 
    help="field name in metrics data to select root nodes by")
parser.add_argument("--output_path", "-o",
    default=".", 
    help="directory where all output files will be written")

args = parser.parse_args()
params = load_arguments(args.__dict__)

if 'metrics_path' not in params:
    raise Exception("metrics_path parameter is required")
if 'network_path' not in params:
    raise Exception("network_path parameter is required")

nwp = NetworkPlan.from_files(params['network_path'], params['metrics_path'], prioritize=params['prioritize_field'])
sequencer = Sequencer(nwp, params['demand_field'])

results = sequencer.sequence()
sequencer.output(params['output_path'])
