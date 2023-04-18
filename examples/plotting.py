# plotting.py
# Authors: Stephan Meighen-Berger
# Example of how to plot the brightest event from promehteus output

import awkward as ak
import sys
sys.path.append('../')
from prometheus import Prometheus, config
from prometheus.plotting import plot_brightest
from prometheus.utils.geo_utils import 

def main():
    config["detector"]['detector specs file'] = '../prometheus/data/custom.txt'
    config["detector"]["file name"] = '../prometheus/data/custom-f2k'
    prometheus = Prometheus()
    data = ak.from_parquet('./output/custom_1337_meta_data.parquet')
    det = from_geo()
