# example.ipynb
# Authors: Stephan Meighen-Berger
# Shows how to generate some simple data sets
# imports
import sys
sys.path.append('../')
from prometheus import Prometheus, config
from jax.config import config as jconfig
import tracemalloc
import warnings
import numpy as np
# Ignore some jax warnings (for now)
warnings.simplefilter(action='ignore', category=FutureWarning)

jconfig.update("jax_enable_x64", True)
# Global flag to set a specific platform, must be used at startup.
jconfig.update('jax_platform_name', 'cpu')

def main(args=None):
    print('CURRENT SET %d' % 1337)
    config['run']['run number'] = 1337
    config["run"]["random state seed"] = 1337
    config['run']['nevents'] = 5
    # Injection parameters
    config["injection"]["name"] = "LeptonInjector"
    config["injection"]['LeptonInjector']['simulation']['is ranged'] = False  # Automatic
    timing_arr = []
    for en in [1e2, 1e3, 1e4, 1e5]:
        config['injection']["LeptonInjector"]['simulation']['minimal energy'] = en
        config['injection']["LeptonInjector"]['simulation']['maximal energy'] = en+1
        # NUmber of modules to model at once
        # Smaller numbers make the simulation slower but less memory intensive
        config['photon propagator']['olympus']['simulation']['splitter'] = 10000
        config["detector"]["geo file"] = '../resources/geofiles/demo_water.geo'
        prom = Prometheus()
        prom.sim()
        timing_arr.append(prom._timing_arr)
        del prom
    np.save('timingtest.npy', np.array(timing_arr))
        

if __name__ == "__main__":
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    tracemalloc.start()
    main()
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
