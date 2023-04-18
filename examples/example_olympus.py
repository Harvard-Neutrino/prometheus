# example.ipynb
# Authors: Stephan Meighen-Berger
# Shows how to generate some simple data sets
# imports
import sys
sys.path.append('../')
from prometheus import Prometheus, config
from jax.config import config as jconfig
import gc
import os
import tracemalloc
import warnings
# Ignore some jax warnings (for now)
warnings.simplefilter(action='ignore', category=FutureWarning)

jconfig.update("jax_enable_x64", True)
# Global flag to set a specific platform, must be used at startup.
jconfig.update('jax_platform_name', 'cpu')

def main(args=None):
    if len(args) == 1:
        rset = 1337
        print("Using default seed!")
    else:
        rset = int(args[1])
    print('CURRENT SET %d' % rset)
    config['run']['run number'] = rset
    config["run"]["random state seed"] = rset
    config['run']['nevents'] = 10
    # Injection parameters
    config["injection"]["name"] = "LeptonInjector"
    config["injection"]['LeptonInjector']['simulation']['is ranged'] = False  # Automatic
    config['injection']["LeptonInjector"]['simulation']['minimal energy'] = 1e4
    config['injection']["LeptonInjector"]['simulation']['maximal energy'] = 1e5
    # NUmber of modules to model at once
    # Smaller numbers make the simulation slower but less memory intensive
    config['photon propagator']['olympus']['simulation']['splitter'] =  10000 # 10000
    config["detector"]["geo file"] = '../resources/geofiles/demo_water.geo'
    prom = Prometheus()
    prom.sim()
    # del prom
    print(tracemalloc.get_traced_memory())

if __name__ == "__main__":
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    tracemalloc.start()
    main(sys.argv)
    tracemalloc.stop()
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
