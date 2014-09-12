import argparse, os

from sequencer import NetworkPlan
from sequencer.Models import EnergyMaximizeReturn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Sequencer on MVMax derived scenario")
    parser.add_argument("-i", "--input-directory", default="data/input", 
                        help="the input directory")
    parser.add_argument("-o", "--output-directory", default="data/output", 
                        help="the output directory")


    args = parser.parse_args()
    csv_file = os.path.join(args.input_directory, "metrics-local.csv")
    shp_file = os.path.join(args.input_directory, "networks-proposed.shp")
                                                                                                                  

    nwp = NetworkPlan(shp_file, csv_file, prioritize='Population')
    model = EnergyMaximizeReturn(nwp)

    results = model.sequence()
    model.output(args.output_directory)
