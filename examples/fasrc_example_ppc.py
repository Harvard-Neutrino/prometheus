# Authors: Jeffrey Lazar, Stephan Meighen-Berger
# Shows how to generate some simple data sets
# imports
import sys
sys.path.append('../')
from prometheus import Prometheus, config
from jax.config import config as jconfig

jconfig.update("jax_enable_x64", True)

ranged_leptons = "MuMinus MuPlus".split()
chared_leptons = "EMinus EPlus MuMinus MuPlus TauMinus TauPlus".split()

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    # Meta
    parser.add_argument(
        "-n",
        dest="n",
        type=int,
        default=10,
        help="number of events to simulate."
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        default=925,
        help="Seed for the random number generator."
    )
    # Injection
    parser.add_argument(
        "--padding",
        dest="padding",
        type=float,
        default=200,
        help="Padding when injecting in volume mode. Should be about an absorption length"
    )
    parser.add_argument(
        "--final_1",
        dest="final_1",
        type=str,
        default="MuMinus",
        help="Final particle 1 for LeptonInjector. See https://arxiv.org/abs/2012.10449 for more information \
        See https://github.com/icecube/LeptonInjector/blob/master/private/LeptonInjector/Particle.cxx#L44 \
        for allowed particles."
    )
    parser.add_argument(
        "--final_2",
        dest="final_2",
        type=str,
        default="Hadrons",
        help="Final particle 2 for LeptonInjector. See https://arxiv.org/abs/2012.10449 for more information. \
        See https://github.com/icecube/LeptonInjector/blob/master/private/LeptonInjector/Particle.cxx#L44  \
        for allowed particles."
    )
    parser.add_argument(
        "--emin",
        dest="emin",
        type=float,
        default=1e3,
        help="Minimum energy for the simulation."
    )
    parser.add_argument(
        "--emax",
        dest="emax",
        type=float,
        default=1e6,
        help="Maximum energy for the simulation."
    )
    parser.add_argument(
        "--gamma",
        dest="gamma",
        type=float,
        default=1.0,
        help="Spectral index for sampling"
    )
    # These LI options are now defunct. You need to do a bit more if you wanna play with them
    parser.add_argument(
        "--endcap_length",
        dest="endcap_length",
        type=float,
        default=900,
        help="LeptonInjector endcap length"
    )
    parser.add_argument(
        "--injection_radius",
        dest="injection_radius",
        type=float,
        default=900,
        help="LeptonInjector injection radius"
    )
    parser.add_argument(
        "--cylinder_radius",
        dest="cylinder_radius",
        type=float,
        default=700,
        help="LeptonInjector cylinder radius"
    )
    parser.add_argument(
        "--cylinder_height",
        dest="cylinder_height",
        type=float,
        default=1000,
        help="LeptonInjector cylinder height"
    )
    parser.add_argument(
        "--force_volume",
        dest="force_volume",
        action="store_true",
        default=False,
        help="Force volume injection."
    )
    # I/O
    parser.add_argument(
        "--geo_file",
        dest="geo_file",
        type=str,
        default="../prometheus/data/icecube-geo",
        help="F2k file describing the geometry of the detector"
    )
    parser.add_argument(
        "--output_prefix",
        dest="output_prefix",
        type=str,
        default="./output",
        help="Prefix for where the data should be stored. You must have write access"
    )
    parser.add_argument(
        "--injection",
        dest="injection",
        type=str,
        default="",
        help="Path to preexisting injection. If this is not null, we will skip LI"
    )
    parser.add_argument(
        "--ppc_tmpfile",
        dest="ppc_tmpfile",
        type=str,
        default="./.event_hits.ppc.tmp",
        help="Path where you wanna store PPC temporary files"
    )
    parser.add_argument(
        "--f2k_tmpfile",
        dest="f2k_tmpfile",
        type=str,
        default="./.event_losses.f2k.tmp",
        help="Path where you wanna store PROPOSAL temporary files"
    )
    return parser.parse_args()

def main(args):
    nevent = args.n
    seed = args.seed

    print('CURRENT SET %d' % seed)
    config["detector"]["specs file"] = args.geo_file
    config["detector"]["padding"] = args.padding
    config["general"]["random state seed"] = seed
    config["general"]["meta_name"] = 'meta_data_%d' % seed
    if args.final_1 in chared_leptons:
        clepton = args.final_1
    elif args.final_2 in chared_leptons:
        clepton = args.final_2
    else:
        clepton = None
    if clepton in ranged_leptons:
        if args.force_volume:
            print("Doing volume injection")
            config['run']['group name'] = 'VolumeInjector0'
            config['injection']["LeptonInjector"]['simulation']['is ranged'] = False
        else:
            print("Doing ranged injection")
            config['run']['group name'] = 'RangedInjector0'
            config['injection']["LeptonInjector"]['simulation']['is ranged'] = True
    else:
        print("Doing volume injection")
        config['run']['group name'] = 'VolumeInjector0'
        config['injection']["LeptonInjector"]['simulation']['is ranged'] = False
    if args.injection:
        config["injection"]["LeptonInjector"]["inject"] = False
        config['injection']["LeptonInjector"]['paths']['output name'] = args.injection
        config['run']['group name'] = 'RangedInjector0'
    else:
        config['run']["nevents"] = nevent
        config['injection']["LeptonInjector"]['simulation']['final state 1'] = args.final_1
        config['injection']["LeptonInjector"]['simulation']['final state 2'] = args.final_2
        config['injection']["LeptonInjector"]['simulation']['minimal energy'] = args.emin
        config['injection']["LeptonInjector"]['simulation']['maximal energy'] = args.emax
        config['injection']["LeptonInjector"]['simulation']["power law"] = args.gamma
        config['injection']["LeptonInjector"]['paths']['output name'] = f"{args.output_prefix}/data_{seed}_output_LI.h5"
        config['injection']["LeptonInjector"]["paths"]['install location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/lib64/"
        config['injection']["LeptonInjector"]["paths"]['xsec location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/LeptonInjector/resources/"
        config['injection']["LeptonInjector"]['paths']["lic name"] = (
            f"{args.output_prefix}/{args.final_1}_{args.final_2}_{seed}_LI_config.lic"
        )
    config["lepton propagator"]['name'] = "old proposal"
    config['general']['storage location'] = f'{args.output_prefix}/{args.final_1}_{args.final_2}_seed_{seed}_'
    photo_prop = "PPC_CUDA"
    config['photon propagator']['name'] = photo_prop
    config['photon propagator'][photo_prop]["paths"]['ppc_tmpfile'] = args.ppc_tmpfile.replace(".ppc", f"{seed}.ppc")
    config['photon propagator'][photo_prop]["paths"]['f2k_tmpfile'] = args.f2k_tmpfile.replace(".f2k", f"{seed}.f2k")
    config['photon propagator'][photo_prop]["paths"]['location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/"
    #config['photon propagator'][photo_prop]["paths"]['ppctables'] = "../PPC_tables/ic_accept_all/"
    config['photon propagator'][photo_prop]["paths"]['ppctables'] = "../PPC_tables/ic_default/"
    config['photon propagator'][photo_prop]["paths"]['ppc_exe'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/ppc"
    #config['photon propagator'][photo_prop]["simulation"]['supress_output'] = False
    prometheus = Prometheus(userconfig=config)
    prometheus.sim()
    del prometheus

if __name__ == "__main__":
    args = initialize_args()
    print(args.f2k_tmpfile)
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("Launching simulation")
    main(args)
    print("Finished call")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
