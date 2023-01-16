# plotting.py
# Authors: Stephan Meighen-Berger
# Plot Prometheus events

import sys
sys.path.append('../')
from prometheus import Prometheus, config

def main():
    config["detector"]['detector specs file'] = '../prometheus/data/custom.txt'
    config["detector"]["file name"] = '../prometheus/data/custom-f2k'
    prometheus = Prometheus()
    data = ak.from_parquet('./output/custom_1337_meta_data.parquet')
    prometheus.plot(
        data,
        e_id=0,
        brightest_event=True,
        channel='total',
    )
