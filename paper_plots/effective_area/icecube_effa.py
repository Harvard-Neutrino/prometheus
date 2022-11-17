# Authors: Jeffrey Lazar, Stephan Meighen-Berger
# Shows how to generate some simple data sets
# imports
import sys
sys.path.append('../../')
from hebe import HEBE, config
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
        default=100,
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
        default=100,
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
        default=1e2,
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
        default="../../hebe/data/icecube-geo",
        help="F2k file describing the geometry of the detector"
    )
    parser.add_argument(
        "--output_prefix",
        dest="output_prefix",
        type=str,
        default="./output",
        help="Prefix for where the data should be stored. You must have write access"
    )
    return parser.parse_args()

def main(args):
    nevent = args.n
    seed = args.seed
    print('CURRENT SET %d' % seed)
    if args.final_1 in chared_leptons:
        clepton = args.final_1
        nclepton = args.final_2
    elif args.final_2 in chared_leptons:
        nclepton = args.final_1
        clepton = args.final_2
    else:
        raise ValueError("What's happening here")
    config["detector"]["detector specs file"] = args.geo_file
    config["detector"]["padding"] = args.padding
    config["general"]["random state seed"] = seed
    config["general"]["config location"] = f"{args.output_prefix}/config_{args.final_1}_{args.final_2}_seed_{seed}.json"
    config["general"]["meta_name"] = 'meta_data_%d' % seed
    if clepton in ranged_leptons:
        config['run']['group name'] = 'RangedInjector0'
        config['lepton injector']['simulation']['is ranged'] = True
    else:
        config['run']['group name'] = 'VolumeInjector0'
        config['lepton injector']['simulation']['is ranged'] = False
    config['lepton injector']['simulation']['output name'] = f"{args.output_prefix}/config_{args.final_1}_{args.final_2}_seed_{seed}.h5"
    config['lepton injector']['simulation']['nevents'] = nevent
    config['lepton injector']['simulation']['final state 1'] = args.final_1
    config['lepton injector']['simulation']['final state 2'] = args.final_2
    config['lepton injector']['simulation']['minimal energy'] = args.emin
    config['lepton injector']['simulation']['maximal energy'] = args.emax
    config['lepton injector']['simulation']["power law"] = args.gamma
    config['lepton injector']['location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/lib64/"
    config['lepton injector']['xsec location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/LeptonInjector/resources/"
    config['lepton propagator']["lepton"] = clepton
    config['photon propagator']['storage location'] = f'{args.output_prefix}/{args.final_1}_{args.final_2}_seed_{seed}_'
    photo_prop = "PPC_CUDA"
    config['photon propagator']['name'] = photo_prop
    config['photon propagator'][photo_prop]['ppc_tmpfile'] = args.ppc_tmpfile.replace(".ppc", f"{seed}.ppc")
    config['photon propagator'][photo_prop]['f2k_tmpfile'] = args.f2k_tmpfile.replace(".f2k", f"{seed}.f2k")
    config['photon propagator'][photo_prop]['location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/"
    config['photon propagator'][photo_prop]['ppctables'] = "/n/home12/jlazar/hebe/PPC_CUDA/"
    config['photon propagator'][photo_prop]['ppc_exe'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/ppc"
    #config['photon propagator'][photo_prop]['supress_output'] = False
    hebe = HEBE(userconfig=config)
    hebe.sim()
    del hebe

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
