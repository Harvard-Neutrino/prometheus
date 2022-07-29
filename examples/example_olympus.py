# example.ipynb
# Authors: Stephan Meighen-Berger
# Shows how to generate some simple data sets
# imports
import sys
sys.path.append('../')
from hebe import HEBE, config
from jax.config import config as jconfig
import gc
import os

jconfig.update("jax_enable_x64", True)

def main(args=None):
    if args is None:
        args = 1337
        print("Using default seed!")
    else:
        rset = int(args[1])
    print('CURRENT SET %d' % rset)
    config["general"]["random state seed"] = rset
    config["general"]["meta_name"] = 'meta_data_%d' % rset
    config['general']['clean up'] = True
    config['lepton injector']['simulation']['output name'] = "./output/custom_%d_output_LI.h5" % rset
    config['photon propagator']['storage location'] = './output/custom_%d_' % rset
    config['lepton injector']['simulation']['nevents'] = 10
    config['lepton injector']['simulation']['minimal energy'] = 30
    config['lepton injector']['simulation']['maximal energy'] = 40
    config['lepton injector']['simulation']["injection radius"] = 150
    config['lepton injector']['simulation']["endcap length"] = 200
    config['lepton injector']['simulation']["cylinder radius"] = 150
    config['lepton injector']['simulation']["cylinder height"] = 400
    config['detector']['injection offset'] = [0., 0., 0]
    config['photon propagator']['name'] = 'olympus'
    config["detector"]['new detector'] = True
    config["detector"]['detector specs file'] = '../hebe/data/custom.txt'
    config["detector"]["file name"] = '../hebe/data/custom-f2k'
    hebe = HEBE()

    hebe.sim()
    del hebe
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
    main(sys.argv)
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
