# Authors: Jeffrey Lazar, Stephan Meighen-Berger
# Shows how to generate some simple data sets
# imports
import numpy as np
from typing import Union

import prometheus
from prometheus import Prometheus, config

RESOURCE_DIR = f"{'/'.join(prometheus.__path__[0].split('/')[:-1])}/resources/"

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    # Meta
    parser.add_argument(
        "-n",
        "--events",
        dest="n",
        type=int,
        default=5000,
        help="number of events to simulate."
    )
    parser.add_argument(
        "--run_n",
        type=int,
        required=True,
        help="Run number for the simulation. Will be used as a seed if None is provided"
    )
    parser.add_argument(
        "--sub_run_n",
        type=int,
        default=-1,
        help="Run number for the simulation. Will be used as a seed if None is provided"
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
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
    # I/O
    parser.add_argument(
        "--geo_file",
        dest="geo_file",
        type=str,
        default=f"{RESOURCE_DIR}/geofiles/icecube.geo",
        help="Prometheus geometry file for the detector"
    )
    parser.add_argument(
        "--earth_file",
        dest="earth_file",
        type=str,
        default=f"{RESOURCE_DIR}/earthparams/densities/PREM_mmc.dat",
        help="Prometheus geometry file for the detector"
    )
    parser.add_argument(
        "--output_prefix",
        dest="output_prefix",
        type=str,
        default="/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/prometheus_simulation/",
        help="Prefix for where the data should be stored. You must have write access"
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
    run_n = args.run_n
    seed = np.random.randint(0, 2**30 -1)
    if args.seed is not None:
        seed = args.seed
    nice_run_n = str(run_n).zfill(5)
    if args.sub_run_n >= 0:
        nice_run_n += f"-{str(args.sub_run_n).zfill(3)}"

    config['run']["nevents"] = nevent
    config["run"]["random state seed"] = seed
    config["run"]["run number"] = seed
    config["run"]["meta_name"] = 'meta_data_%d' % seed
    config['run']['storage prefix'] = (
        f'{args.output_prefix}/'
    )
    config["detector"]["geo file"] = args.geo_file
    config["detector"]["padding"] = args.padding

    # injection
    config['injection']['name'] = "SIREN"
    config['injection']["SIREN"]['paths']['injection file'] = (
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/IceCube_DB/analysis/output/IceCube_DipoleDIS_2.hdf5"
    )

    config["lepton propagator"]['name'] = "new proposal"
    config['photon propagator']['name'] = "PPC_CUDA"
    config['photon propagator']["PPC_CUDA"]["paths"]['outfile'] = (
        f"{args.output_prefix}/Generation_{nice_run_n}_photons.parquet"
    )
    config['photon propagator']["PPC_CUDA"]["paths"]['ppc_tmpfile'] = args.ppc_tmpfile.replace(".ppc", f"{seed}.ppc")
    config['photon propagator']["PPC_CUDA"]["paths"]['f2k_tmpfile'] = args.f2k_tmpfile.replace(".f2k", f"{seed}.f2k")
    config['photon propagator']["PPC_CUDA"]["paths"]['location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/"
    config['photon propagator']["PPC_CUDA"]["paths"]['ppctables'] = f"{RESOURCE_DIR}/PPC_tables/south_pole/"
    # Uncomment this line to not use IceCube's angular acceptance
    #config['photon propagator'][photo_prop]["paths"]['ppctables'] = "../resources/PPC_tables/ic_accept_all/"
    config['photon propagator']["PPC_CUDA"]["paths"]['ppc_exe'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/ppc"
    config['photon propagator']["PPC_CUDA"]["paths"]['ppc_exe'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/ppc"
    config["photon propagator"]["PPC_CUDA"]["paths"]["ppc_tmpdir"] = f"{args.output_prefix}/{nice_run_n}_ppc_tmp/"
    config['photon propagator']["PPC_CUDA"]["simulation"]['supress_output'] = False
    prometheus = Prometheus(userconfig=config)
    prometheus.sim()
    del prometheus

if __name__ == "__main__":
    args = initialize_args()
    main(args)
