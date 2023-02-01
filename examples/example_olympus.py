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
    if args is None:
        args = 1337
        print("Using default seed!")
    else:
        rset = int(args[1])
    print('CURRENT SET %d' % rset)
    config['run']['run number'] = rset
    config["run"]["random state seed"] = rset
    config['run']['nevents'] = 10
    # Injection parameters
    config["injection"]["name"] = "LeptonInjector"
    config['injection']["LeptonInjector"]['simulation']['minimal energy'] = 1e4
    config['injection']["LeptonInjector"]['simulation']['maximal energy'] = 1e6
    # NUmber of modules to model at once
    # Smaller numbers make the simulation slower but less memory intensive
    config['photon propagator']['olympus']['simulation']['splitter'] = 4000
    # config['photon propagator']['name'] = 'olympus'
    config["detector"]["specs file"] = '../prometheus/data/pone_triangle-geo'
    prom = Prometheus()
    prom.sim()
    del prom
    gc.collect()
    # Getting all memory using os.popen()
    total_memory, used_memory, _ = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))

if __name__ == "__main__":
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    tracemalloc.start()
    main(sys.argv)
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
