import json
import pyarrow.parquet as pq
import numpy as np
import awkward as ak

from prometheus.weighting import ParquetWeighter

def initialize_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--parquet_file",
        required=True,
    )
    parser.add_argument(
        "--nbins",
        default=10,
        type=int
    )

    args = parser.parse_args()
    return args

def main(parquet_file=None, nbins=10) -> None:

    args = initialize_args()
    parquet_file = args.parquet_file

    if parquet_file is None:
        parquet_file = args.parquet_file
    weighter = ParquetWeighter(parquet_file)

    # Load up the configuraion
    config = json.loads(
        pq.read_metadata(parquet_file).metadata[b"config_prometheus"]
    )
    inj_conf = config["injection"]["LeptonInjector"]["simulation"]

    # Compute the solid angle
    delta_omega = (
        (np.radians(inj_conf["max azimuth"]) - np.radians(inj_conf["min azimuth"])) *
        (np.cos(np.radians(inj_conf["min zenith"])) - np.cos(np.radians(inj_conf["max zenith"])))
    )

    # Make the energy bins
    edges = np.logspace(
        np.log10(inj_conf["minimal energy"]),
        np.log10(inj_conf["maximal energy"]),
        args.nbins+1
    )
    widths = np.diff(edges)
    cents = (edges[1:] + edges[:-1]) / 2

    # Load up events
    a = ak.from_parquet(parquet_file)
    # Only consider the effective area for events that produced light
    mask = np.array([len(x["t"]) > 0 for x in a["photons"]])
    weights = weighter.weight_events()

    h, _ = np.histogram(
        a["mc_truth", "initial_state_energy", mask].to_numpy(),
        bins=edges,
        weights=weights[mask]
    )

    effa = h / delta_omega / widths / len(a)

    print(f"Energies: {cents} GeV")
    print(f"Effective area: {effa} m")

if __name__=="__main__":
    main()
