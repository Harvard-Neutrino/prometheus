from prometheus import Prometheus, config
 
ranged_leptons = "MuMinus MuPlus".split()
chared_leptons = "EMinus EPlus MuMinus MuPlus TauMinus TauPlus".split()
 
def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    # Meta
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        default=925,
        help="Seed for the random number generator."
    )
    parser.add_argument(
        "--final_1",
        dest="final_1",
        type=str,
        required=True,
        help="Final particle 1 for LeptonInjector. See https://arxiv.org/abs/2012.10449 for more information \
        See https://github.com/icecube/LeptonInjector/blob/master/private/LeptonInjector/Particle.cxx#L44 \
        for allowed particles."
    )
    parser.add_argument(
        "-e",
        "--energy",
        dest="energy",
        type=float,
        default=1e3,
        help="Minimum energy for the simulation."
    )
    parser.add_argument(
        "--geo_file",
        dest="geo_file",
        type=str,
        default="/n/home12/jlazar/prometheus/resources/geofiles/demo_ice.geo",
        help="F2k file describing the geometry of the detector"
    )
    parser.add_argument(
        "--timing_outdir",
        dest="timing_outdir",
        type=str,
        default="",
        help="Prefix for where the data should be stored. You must have write access"
    )
    parser.add_argument(
        "--output_prefix",
        dest="output_prefix",
        type=str,
        default="./output",
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
    import numpy as np
    import prometheus
    prefix = f"timing_{args.final_1}_{args.energy}"
    nevent = 1000
    seed = args.seed
    config['run']["nevents"] = nevent
    config["run"]["random state seed"] = seed
    config["run"]["run number"] = seed
    config["detector"]["geo file"] = args.geo_file
    if args.final_1 in ranged_leptons:
        config["injection"]["LeptonInjector"]["simulation"]["is ranged"] = True
    else:
        config["injection"]["LeptonInjector"]["simulation"]["is ranged"] = False
    config["run"]["nevents"] = nevent
    config["injection"]["LeptonInjector"]["simulation"]["final state 1"] = args.final_1
    config["injection"]["LeptonInjector"]["simulation"]["final state 2"] = "Hadrons"
    config["injection"]["LeptonInjector"]["simulation"]["minimal energy"] = args.energy
    config["injection"]["LeptonInjector"]["simulation"]["maximal energy"] = args.energy
    config['injection']["LeptonInjector"]['simulation']['min zenith'] = 0
    config['injection']["LeptonInjector"]['simulation']['max zenith'] = 180
    config['injection']["LeptonInjector"]['paths']['injection file'] = (
        f"{args.timing_outdir}/h5_files/{prefix}_LI_output.h5"
    )
    config['injection']["LeptonInjector"]['paths']["lic file"] = (
        f"{args.timing_outdir}/lic_files/{prefix}_LI_config.lic"
    )
    config["lepton propagator"]['name'] = "new proposal"
    config['photon propagator']['name'] = "PPC_CUDA"
    config['run']['outfile'] = (
        f"{args.timing_outdir}/photons/{prefix}_photons.parquet"
    )
    config['photon propagator']["PPC_CUDA"]["paths"]['ppc_tmpfile'] = args.ppc_tmpfile.replace(".ppc", f"{seed}.ppc")
    config['photon propagator']["PPC_CUDA"]["paths"]['f2k_tmpfile'] = args.f2k_tmpfile.replace(".f2k", f"{seed}.f2k")
    config['photon propagator']["PPC_CUDA"]["paths"]['location'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/"
    config['photon propagator']["PPC_CUDA"]["paths"]['ppctables'] = "../../resources/PPC_tables/south_pole/"
    # Uncomment this line to not use IceCube's angular acceptance
    config['photon propagator']["PPC_CUDA"]["paths"]['ppc_exe'] = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/common_software/source/PPC_CUDA_new/ppc"
    config["photon propagator"]["PPC_CUDA"]["paths"]["ppc_tmpdir"] = f"./ppc_tmpdir_{np.random.randint(0, 2**32 - 1)}"
    #config["photon propagator"]["PPC_CUDA"]["simulation"]["supress_output"] = False
    prometheus = Prometheus(userconfig=config)
    prometheus.sim()

if __name__ == "__main__":
    args = initialize_args()
    if args.timing_outdir:
        timing_outdir = args.timing_outdir
    else:
        raise ValueError()
    filename = f"{timing_outdir}/timing_{args.energy}.dat"
    from cProfile import run
    run("main(args)", filename=filename)
    import pstats
    from pstats import SortKey
    p = pstats.Stats(filename)
    filters = [
        "ppc_photon_propagator.py:87", # photon propagation + proposal
        "old_proposal_lepton_propagator.py:283", # energy losses
        "lepton_propagator.py:15", # making propagator
        "prometheus.py:158", # making a new injection
        "prometheus.py:252", # constructing the output
        "prometheus.py:243" # full runtime
    ]
    with open(f"{timing_outdir}/{args.energy}_summary_stats.txt", "w") as stream:
        p = pstats.Stats(filename, stream=stream)
        for filt in filters:
            p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(filt)
    #filters = [
    #    "ppc_photon_propagator.py:87", # photon propagation + proposal
    #    "old_proposal_lepton_propagator.py:283", # energy losses
    #    "lepton_propagator.py:15", # making propagator
    #    "prometheus.py:158", # making a new injection
    #    "prometheus.py:252", # constructing the output
    #    "prometheus.py:243" # full runtime
    #]
    ##filters = ["import", "photon", "inject", "awkward", "proposal", "prometheus"]
    #ns = []
    #with open(f"{timing_outdir}/{args.energy}_summary_stats.txt", "w") as stream:
    #    p = pstats.Stats(filename, stream=stream)
    #    for filt in filters:
    #        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(filt, n)
