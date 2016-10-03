# -*- coding: utf-8 -*-
"""
Run sequencer
"""

import argparse
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
    
parser = argparse.ArgumentParser(
    description="Run sequencer on networkplanner output"
    " (metrics csv and network shp)")
parser.add_argument("--working_directory", "-w",
    default=".", 
    help="base directory which input paths are relative to")
parser.add_argument("--metrics_file", "-m", 
    help="metrics csv filename")
parser.add_argument("--network_file", "-n", 
    help="network shp filename")
parser.add_argument("--demand_field", "-d",
    help="field name in metrics data representing demand")
parser.add_argument("--prioritize_field", "-p",
    default="Population", 
    help="field name in metrics data to select root nodes by")
parser.add_argument("--output_directory", "-o",
    default=".", 
    help="directory where all output files will be written")

args = parser.parse_args()

os.chdir(args.working_directory)

nwp = NetworkPlan.from_files(args.network_file, args.metrics_file, prioritize=args.prioritize_field)
sequencer = Sequencer(nwp, args.demand_field)

results = sequencer.sequence()
sequencer.output(args.output_directory)
