import argparse
from pathlib import Path
from woundcompute import image_analysis as ia

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="the .yml user input file")
args = parser.parse_args()
input_file_str = args.input_file
input_file = Path(input_file_str)
time_all, action_all = ia.run_all(input_file)