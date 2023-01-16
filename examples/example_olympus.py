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
import tracemalloc

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
    config['injection']["LeptonInjector"]['paths']['output name'] = "./output/custom_%d_output_LI.h5" % rset
    config['photon propagator']['storage location'] = './output/custom_%d_' % rset
    config['injection']["LeptonInjector"]['simulation']['nevents'] = 10
    config['injection']["LeptonInjector"]['simulation']['minimal energy'] = 1e4
    config['injection']["LeptonInjector"]['simulation']['maximal energy'] = 1e5
    config['injection']["LeptonInjector"]['simulation']["injection radius"] = 150
    config['injection']["LeptonInjector"]['simulation']["endcap length"] = 200
    config['injection']["LeptonInjector"]['simulation']["cylinder radius"] = 150
    config['injection']["LeptonInjector"]['simulation']["cylinder height"] = 400
    config['injection']["LeptonInjector"]['force injection params'] = True
    config['photon propagator']['olympus']['splitter'] = 1000
    config['detector']['injection offset'] = [0., 0., 0]
    config['photon propagator']['name'] = 'olympus'
    config["detector"]["detector specs file"] = '../hebe/data/pone_triangle-geo'
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
    tracemalloc.start()
    main(sys.argv)
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
