"""02_train_antares_nflow.py
Train KM3NeT/ANTARES normalizing-flow photon models.

Loads shape_data.npz and counts_data.npz produced by
``examples/water_model/01_generate_photon_mc.py``, trains the shape (timing)
and counts (survival probability) models, and saves the results as pickle
files ready for use in the Prometheus photon propagator.

Usage
-----
    .prometheus_env/bin/python examples/water_model/02_train_antares_nflow.py

Output
------
    resources/olympus_resources/antares_nflow_params.pickle
    resources/olympus_resources/antares_counts_params.pickle

Notes on model sizing (learned empirically)
-------------------------------------------
The shape dataset (~1.9 M weighted photon hits) supports the large MLP
default (mlp_hidden_size=400, mlp_num_layers=3) with no overfitting. The
counts dataset is much smaller (~1,400 rows, fixed by the MC grid, not by
n_photons), so it needs a small MLP (--small-counts, the default) trained
full-batch; a large MLP overfits it severely.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from hyperion.data import DataLoader, SimpleDataset, create_random_split
from hyperion.models.photon_arrival_time_nflow.net import (
    make_counts_net_fn,
    train_counts_model,
    train_shape_model,
)
from hyperion.models.photon_arrival_time_nflow.validation import check_counts_physics

# ---------------------------------------------------------------------------
# Defaults (match the existing P-ONE model hyper-parameters)
# ---------------------------------------------------------------------------
_SHAPE_CONFIG = dict(
    flow_num_layers=2,
    flow_num_bins=10,
    flow_rmin=0,
    flow_rmax=500,
    mlp_hidden_size=400,
    mlp_num_layers=3,
    lr=1e-3,
    steps=4000,
)

# Large counts MLP — kept for reference only.  Overfits badly on the small
# (~1,400-row) counts dataset; do not use unless the dataset is much larger.
_COUNTS_CONFIG = dict(
    mlp_hidden_size=500,
    mlp_num_layers=3,
    lr=5e-3,
    steps=25000,
)

# Preferred counts model.  64-unit, 2-layer MLP has ~10 k parameters, which
# is appropriate for ~1,200 training rows.  Train/test gap stays below 10 %
# throughout training and the test loss plateaus cleanly.
_COUNTS_CONFIG_SMALL = dict(
    mlp_hidden_size=64,
    mlp_num_layers=2,
    lr=5e-3,
    steps=25000,
)

_SHAPE_BATCH = 10_000
_COUNTS_BATCH = 300
_TRAIN_FRAC = 0.9
_SEED = 42

_HERE = Path(__file__).parent
_DATA_DIR = _HERE.parent / "output" / "water_model"
_RESOURCE_DIR = _HERE.parent.parent / "resources" / "olympus_resources"


def _make_loaders(dataset, batch_size, rng, train_frac=_TRAIN_FRAC):
    n_train = int(train_frac * len(dataset))
    train_ds, test_ds = create_random_split(dataset, n_train, rng)
    train_loader = DataLoader(train_ds, batch_size=batch_size, rng=rng, shuffle=True, infinite=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, rng=rng, shuffle=False, infinite=False)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DATA_DIR,
        help="Directory containing shape_data.npz and counts_data.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_RESOURCE_DIR,
        help="Directory to write trained pickle files.",
    )
    parser.add_argument(
        "--shape-steps",
        type=int,
        default=_SHAPE_CONFIG["steps"],
        help="Training steps for the shape model.",
    )
    parser.add_argument(
        "--counts-steps",
        type=int,
        default=_COUNTS_CONFIG["steps"],
        help="Training steps for the counts model.",
    )
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Skip shape model training; only retrain the counts model.",
    )
    parser.add_argument(
        "--large-counts",
        action="store_true",
        help="Use the large counts MLP (500 units, 3 layers). Not recommended: "
        "overfits severely on the ~1,400-row counts dataset.",
    )
    parser.add_argument(
        "--counts-out",
        type=str,
        default="antares_counts_params_low_E.pickle",
        help="Output filename for the counts model pickle (relative to --out-dir).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # Shape model
    # -----------------------------------------------------------------------
    if not args.counts_only:
        shape_path = args.data_dir / "shape_data.npz"
        print(f"Loading shape data from {shape_path}")
        s = np.load(shape_path)
        shape_ds = SimpleDataset(s["log10_dist"], s["angle"], s["t_residual"], s["weight"])
        print(f"  {len(shape_ds):,} samples")

        shape_config = dict(_SHAPE_CONFIG)
        shape_config["steps"] = args.shape_steps

        train_loader, test_loader = _make_loaders(shape_ds, _SHAPE_BATCH, rng)
        print(f"\nTraining shape model ({shape_config['steps']} steps) …")
        shape_params = train_shape_model(shape_config, train_loader, test_loader)

        shape_out = args.out_dir / "antares_nflow_params_low_E.pickle"
        with open(shape_out, "wb") as f:
            pickle.dump((shape_config, shape_params), f)
        print(f"Shape model saved → {shape_out}")

    # -----------------------------------------------------------------------
    # Counts model
    # -----------------------------------------------------------------------
    counts_path = args.data_dir / "counts_data.npz"
    print(f"\nLoading counts data from {counts_path}")
    c = np.load(counts_path)
    counts_ds = SimpleDataset(c["log10_dist"], c["angle"], c["log10_survival"], c["n_detected"])
    print(f"  {len(counts_ds):,} samples")

    base_config = _COUNTS_CONFIG if args.large_counts else _COUNTS_CONFIG_SMALL
    counts_config = dict(base_config)
    counts_config["steps"] = args.counts_steps

    # Full-batch training: with only ~1,400 counts samples, using the whole
    # training set as one batch avoids the noisy mini-batch gradient estimates
    # that cause overfitting at large step counts.
    counts_batch = max(len(counts_ds), 1)
    train_lc, test_lc = _make_loaders(counts_ds, counts_batch, rng)
    label = "large (not recommended)" if args.large_counts else "small"
    print(f"\nTraining counts model [{label}] ({counts_config['steps']} steps) …")
    counts_params = train_counts_model(counts_config, train_lc, test_lc)

    counts_net = make_counts_net_fn(counts_config)
    check_counts_physics(counts_net, counts_params)

    counts_out = args.out_dir / args.counts_out
    with open(counts_out, "wb") as f:
        pickle.dump((counts_config, counts_params), f)
    print(f"Counts model saved → {counts_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
