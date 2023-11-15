# Authors: Jeffrey Lazar, Stephan Meighen-Berger
# Shows how to generate some simple data sets
# imports
import sys
sys.path.append('../')
from prometheus import Prometheus, config
import prometheus
#from jax.config import config as jconfig
import gc
import os

#jconfig.update("jax_enable_x64", True)

RESOURCE_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/resources/"

def main(simset):
    print('CURRENT SET %d' % simset)
    config["run"]["nevents"] = 20
    config["run"]["random state seed"] = simset
    config["run"]["outfile"] = f"./output/simset_{simset}_photons.parquet"
    config['injection']["LeptonInjector"]["paths"]['xsec dir'] = (
        f"{RESOURCE_DIR}/cross_section_splines/"
    )
    #config['injection']["LeptonInjector"]["paths"]['install location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/lib64/"
    config["detector"]["geo file"] = f"{RESOURCE_DIR}/geofiles/icecube.geo"
    config["injection"]['LeptonInjector']['simulation']['is ranged'] = False
    config["injection"]['LeptonInjector']['simulation']['output name'] = "./output/data_%d_output_LI.h5" % simset
    config["injection"]['LeptonInjector']['simulation']['minimal energy'] = 1e3
    config["injection"]['LeptonInjector']['simulation']['maximal energy'] = 1e4
    config['photon propagator']['name'] = 'PPC_CUDA'
    config["photon propagator"]["PPC_CUDA"]["paths"]["ppc_tmpdir"] = "./ppc_tmpdir"
    prometheus = Prometheus(userconfig=config)
    prometheus.sim()
    del prometheus
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
    simset = 925
    if len(sys.argv)==2:
        simset = int(sys.argv[1])
    main(simset)
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
