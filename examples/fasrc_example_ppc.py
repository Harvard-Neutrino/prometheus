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
    if len(args)==1:
        rset = 1337
        print("Using default seed!")
    else:
        rset = int(args[1])
    print('CURRENT SET %d' % rset)
    pname = 'TauMinus'
    config["general"]["random state seed"] = rset
    config["general"]["meta_name"] = 'meta_data_%d' % rset
    scratch_dir = "/n/holyscratch01/arguelles_delgado_lab/Everyone/jlazar/muons/data/"
    config['photon propagator']['storage location'] = f'{scratch_dir}/rset_{rset}_'
    nevent = 1
    config['lepton injector']['location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/lib64/"
    config['lepton injector']['xsec location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/LeptonInjector/resources/"
    config['lepton injector']['simulation']['output name'] = "./data_%d_output_LI.h5" % rset
    config['lepton injector']['simulation']['nevents'] = nevent
    config['lepton injector']['simulation']['final state 1'] = pname
    config['lepton injector']['simulation']['minimal energy'] = 1e7
    config['lepton injector']['simulation']['maximal energy'] = 1e7
    config['lepton injector']['simulation']['is ranged'] = False
    config['lepton injector']['simulation']["injection radius"] = 800
    config['lepton injector']['simulation']["endcap length"] = 800
    config['lepton injector']['simulation']["cylinder radius"] = 700
    config['lepton injector']['simulation']["cylinder height"] = 1000
    config['detector']['injection offset'] = [0., 0., -2000]
    config['lepton propagator']['lepton'] = pname
    config["detector"]["file name"] = '../hebe/data/icecube-f2k'
    photo_prop = "PPC_CUDA"
    config['photon propagator']['name'] = photo_prop
    config['photon propagator'][photo_prop]['ppc_tmpfile'] = f"/n/home12/jlazar/hebe/examples/.{rset}_event_hits.ppc.tmp"
    config['photon propagator'][photo_prop]['f2k_tmpfile'] = f"/n/home12/jlazar/hebe/examples/.{rset}_event_losses.f2k.tmp"
    config['photon propagator'][photo_prop]['location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA/"
    config['photon propagator'][photo_prop]['ppctables'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA/"
    config['photon propagator'][photo_prop]['ppc_exe'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA/ppc"
    config['run']['group name'] = 'VolumeInjector0'
    hebe = HEBE(userconfig=config)
    hebe.sim()
    for idx in range(nevent):
        print('Plotting')
        event = {'final_1':[hebe.results['final_1'][idx]], 'final_2':[hebe.results['final_2'][idx]]}
        hebe.ppc_event_plotting(event, fig_name=f'./{pname}_{rset}_{idx}.pdf', show_track=False, show_dust_layer=True)

    #hebe.sim()
    del hebe
    #gc.collect()
    ## Getting all memory using os.popen()
    #total_memory, used_memory, _ = map(
    #    int, os.popen('free -t -m').readlines()[-1].split()[1:])
    ## Memory usage
    #print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))

if __name__ == "__main__":
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    main(sys.argv)
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
