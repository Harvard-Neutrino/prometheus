# example.ipynb
# Authors: Stephan Meighen-Berger
# Shows how to generate some simple data sets
# imports
import sys
sys.path.append('../')
from prometheus import Prometheus, config
from jax.config import config as jconfig
import warnings
import numpy as np
# Ignore some jax warnings (for now)
warnings.simplefilter(action='ignore', category=FutureWarning)

jconfig.update("jax_enable_x64", True)
# Global flag to set a specific platform, must be used at startup.
jconfig.update('jax_platform_name', 'cpu')

def main(energy, rset):
    print('CURRENT SET %d' % rset)
    config['run']['run number'] = rset
    config["run"]["random state seed"] = rset
    config['run']['nevents'] = 10
    # Injection parameters
    config["injection"]["name"] = "LeptonInjector"
    config["injection"]['LeptonInjector']['simulation']['is ranged'] = False  # Automatic
    config['injection']["LeptonInjector"]['simulation']['minimal energy'] = energy
    config['injection']["LeptonInjector"]['simulation']['maximal energy'] = energy+1
    # NUmber of modules to model at once
    # Smaller numbers make the simulation slower but less memory intensive
    config['photon propagator']['olympus']['simulation']['splitter'] = 10000
    config["detector"]["geo file"] = '../resources/geofiles/demo_water.geo'
    prom = Prometheus()
    prom.sim()
    return prom._timing_arr
        

if __name__ == "__main__":
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    timing_arr = []
    for energy in [1e2, 1e3, 1e4, 1e5]:
        tmp_timing = []
        for i in range(50):
            try:
                tmp_timing.append(main(energy, i))
            except:
                print('For energy %.1e' % energy)
                print('Skipping %d' % i)
                continue
        timing_arr.append(np.mean(tmp_timing, axis=0))

    timing_arr = np.array(timing_arr)
    np.save('timings.npy', timing_arr)
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
