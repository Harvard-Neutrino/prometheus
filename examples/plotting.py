# plotting.py
# Authors: Stephan Meighen-Berger
# Plot Prometheus events

import sys
sys.path.append('../')
from hebe import HEBE, config

def main():
    config["detector"]['detector specs file'] = '../hebe/data/custom.txt'
    config["detector"]["file name"] = '../hebe/data/custom-f2k'
    hebe = HEBE()
    data = ak.from_parquet('./output/custom_1337_meta_data.parquet')
    hebe.plot(
        data,
        e_id=0,
        brightest_event=True,
        channel='total',
    )
