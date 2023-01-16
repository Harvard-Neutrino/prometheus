# Authors: Jeffrey Lazar, Stephan Meighen-Berger
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
    config["run"]["nevent"] = 2
    config["general"]["state seed"] = rset
    config["general"]["meta_name"] = 'meta_data_%d' % rset
    config['photon propagator']['storage location'] = './output/rset_%d_' % rset
    nevent = 2
    config['injection']["LeptonInjector"]["paths"]['install location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/lib64/"
    config['injection']["LeptonInjector"]["paths"]['xsec location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/LeptonInjector/resources/"
    config["injection"]['LeptonInjector']['simulation']['is ranged'] = False
    config["injection"]['LeptonInjector']['simulation']['output name'] = "./output/data_%d_output_LI.h5" % rset
    config["injection"]['LeptonInjector']['simulation']['minimal energy'] = 1e3
    config["injection"]['LeptonInjector']['simulation']['maximal energy'] = 1e4
    config["injection"]['LeptonInjector']['simulation']["injection radius"] = 800
    config["injection"]['LeptonInjector']['simulation']["endcap length"] = 800
    config["injection"]['LeptonInjector']['simulation']["cylinder radius"] = 800
    config["injection"]['LeptonInjector']['simulation']["cylinder height"] = 1000
    config["detector"]["specs file"] = '../hebe/data/icecube-geo'
    config['photon propagator']['name'] = 'PPC_CUDA'
    hebe = HEBE(userconfig=config)
    hebe.sim()

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
