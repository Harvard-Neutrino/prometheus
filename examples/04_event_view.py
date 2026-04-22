#!/usr/bin/env python3
"""04_event_view.py
Visualise a single Prometheus event in 3-D.

By default the brightest event (most photon hits) in the parquet file is
shown.  Unhit OMs appear as small black dots.  Hit OMs are drawn as colored
spheres whose:

  - **color** encodes the mean photon arrival time at that OM
    (black = earliest, purple = intermediate, orange = latest),
  - **size** is proportional to the number of photons that arrived.

The background plane color is water-blue (dark) or ice-blue (light) and is
chosen automatically from the geo file medium when ``--geo`` is supplied.

Usage
-----
python examples/04_event_view.py examples/output/1_photons.parquet

python examples/04_event_view.py examples/output/1_photons.parquet \\
    --geo resources/geofiles/demo_water.geo \\
    --event 2 \\
    --out event.png \\
    --show
"""
import argparse
import sys
import logging

logger = logging.getLogger(__name__)

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection


# ── Custom colormap: black → purple → orange ─────────────────────────────────
_CMAP_COLORS = [
    (0.00, (0.00, 0.00, 0.00)),   # t_min  → black
    (0.50, (0.44, 0.18, 0.63)),   # t_mid  → purple
    (1.00, (1.00, 0.60, 0.10)),   # t_max  → orange
]
TIME_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "time_cmap",
    [(v, c) for v, c in _CMAP_COLORS],
)

# Background fill colors
_BG_WATER = "#06305a"   # deep water blue
_BG_ICE   = "#cde4f5"   # pale ice blue


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-D event-view for a Prometheus parquet output file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "parquet",
        help="Path to the Prometheus photon parquet file.",
    )
    p.add_argument(
        "--geo",
        default=None,
        metavar="GEO",
        help=(
            "Geo file used for the simulation.  When supplied, unhit OMs are "
            "shown as small black dots and the background color reflects the "
            "medium (water = dark blue, ice = light blue)."
        ),
    )
    p.add_argument(
        "--event",
        type=int,
        default=None,
        metavar="IDX",
        help=(
            "0-based row index of the event to visualise.  "
            "Defaults to the brightest event (most photon hits)."
        ),
    )
    p.add_argument(
        "--out",
        default="event_view.png",
        metavar="FILE",
        help="Output image file.  Set to '' to skip saving.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Open an interactive matplotlib window after saving.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        metavar="N",
        help="DPI of the saved figure.",
    )
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.exception("Cannot read parquet file %s", path)
        sys.exit(1)


def brightest_event(df: pd.DataFrame) -> int:
    """Return the row index with the most photon hits."""
    counts = [len(row["photons"]["t"]) for _, row in df.iterrows()]
    return int(np.argmax(counts))


def aggregate_hits(photons: dict) -> pd.DataFrame:
    """Group individual photon hits by OM and compute mean-time and hit-count."""
    x = np.array(photons["sensor_pos_x"], dtype=float)
    y = np.array(photons["sensor_pos_y"], dtype=float)
    z = np.array(photons["sensor_pos_z"], dtype=float)
    t = np.array(photons["t"], dtype=float)

    df_hits = pd.DataFrame({"x": x, "y": y, "z": z, "t": t})
    agg = (
        df_hits.groupby(["x", "y", "z"])
        .agg(n_photons=("t", "count"), mean_t=("t", "mean"))
        .reset_index()
    )
    return agg


# ── Detector geometry (optional) ──────────────────────────────────────────────
def load_detector(geo_path: str):
    """Return (module_coords np.ndarray, medium_str) or (None, None)."""
    try:
        from prometheus.detector import detector_from_geo
        det = detector_from_geo(geo_path)
        medium_str = str(det.medium).lower()   # e.g. 'medium.water'
        return det.module_coords, medium_str
    except Exception as e:
        logger.warning("Could not load geo file %s: %s. Unhit OMs hidden.", geo_path, e)
        return None, None


# ── Plotting ──────────────────────────────────────────────────────────────────
def draw_event(
    hit_oms: pd.DataFrame,
    all_om_coords,         # np.ndarray (N, 3) or None
    medium_str: str,
    mc_truth: dict,
    event_idx: int,
    args: argparse.Namespace,
) -> plt.Figure:

    # Background / panel color
    is_water = medium_str is not None and "water" in medium_str
    bg_color = _BG_WATER if is_water else _BG_ICE
    fg_label = "water" if is_water else "ice"

    # ── Normalize arrival time for colormap ──────────────────────────────────
    t_min, t_max = hit_oms["mean_t"].min(), hit_oms["mean_t"].max()
    if t_max > t_min:
        t_norm = (hit_oms["mean_t"] - t_min) / (t_max - t_min)
    else:
        t_norm = np.zeros(len(hit_oms))

    colors = TIME_CMAP(t_norm)

    # ── Marker sizes: sqrt-scale so very bright OMs don't swamp the rest ─────
    n = hit_oms["n_photons"].values.astype(float)
    size_min, size_max = 20.0, 400.0
    n_norm = np.sqrt(n) / np.sqrt(n.max()) if n.max() > 0 else np.ones_like(n)
    sizes = size_min + n_norm * (size_max - size_min)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
    ax: Axes3D = fig.add_subplot(111, projection="3d", facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Suppress axis grid / planes, keep axis labels
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)

    text_color = "white" if is_water else "black"

    ax.tick_params(colors=text_color, labelsize=7)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.zaxis.label.set_color(text_color)
    ax.set_xlabel("x  [m]", labelpad=4)
    ax.set_ylabel("y  [m]", labelpad=4)
    ax.set_zlabel("z  [m]", labelpad=4)

    # ── Unhit OMs (small black dots) ─────────────────────────────────────────
    if all_om_coords is not None:
        ax.scatter(
            all_om_coords[:, 0],
            all_om_coords[:, 1],
            all_om_coords[:, 2],
            s=2,
            c="black",
            alpha=0.3,
            depthshade=False,
            zorder=1,
            label="Unhit OM",
        )

    # ── Hit OMs (colored + sized) ─────────────────────────────────────────────
    sc = ax.scatter(
        hit_oms["x"].values,
        hit_oms["y"].values,
        hit_oms["z"].values,
        s=sizes,
        c=colors,
        depthshade=False,
        zorder=2,
        edgecolors="none",
    )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=TIME_CMAP, norm=mcolors.Normalize(vmin=t_min, vmax=t_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.07, shrink=0.55, aspect=18)
    cbar.set_label("Mean photon arrival time  [ns]", color=text_color, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=text_color)

    # ── Event info title ──────────────────────────────────────────────────────
    energy_gev = mc_truth.get("initial_state_energy", float("nan"))
    ptype = int(mc_truth.get("initial_state_type", 0))
    pname = {12: r"$\nu_e$", -12: r"$\bar\nu_e$",
             14: r"$\nu_\mu$", -14: r"$\bar\nu_\mu$",
             16: r"$\nu_\tau$", -16: r"$\bar\nu_\tau$"}.get(ptype, f"PDG {ptype}")
    total_hits = int(hit_oms["n_photons"].sum())
    n_oms_hit = len(hit_oms)

    title = (
        f"Event {event_idx}  |  {pname}  CC  "
        f"E = {energy_gev/1e3:.1f} TeV  |  "
        f"{total_hits} photons on {n_oms_hit} OMs  |  {fg_label}"
    )
    ax.set_title(title, color=text_color, fontsize=9, pad=8)

    # ── Size legend ───────────────────────────────────────────────────────────
    legend_counts = [1, 10, 100]
    legend_handles = []
    for nc in legend_counts:
        if nc <= n.max():
            frac = np.sqrt(nc) / np.sqrt(n.max())
            ms = size_min + frac * (size_max - size_min)
            h = ax.scatter([], [], [], s=ms, c="gray", label=f"{nc} hits")
            legend_handles.append(h)
    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            title="Photons / OM",
            loc="upper left",
            framealpha=0.3,
            fontsize=7,
            title_fontsize=7,
            labelcolor=text_color,
            facecolor=bg_color,
        )
        legend.get_title().set_color(text_color)

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not args.show:
        matplotlib.use("Agg")

    df = load_parquet(args.parquet)
    if df.empty:
        logger.error("Parquet file %s contains no events.", args.parquet)
        sys.exit(1)

    event_idx = args.event if args.event is not None else brightest_event(df)
    if event_idx < 0 or event_idx >= len(df):
        logger.error("--event %s out of range (0–%s).", event_idx, len(df)-1)
        sys.exit(1)

    logger.info("Visualising event %s (%s events in file)", event_idx, len(df))

    row = df.iloc[event_idx]
    hit_oms = aggregate_hits(row["photons"])
    mc_truth = row["mc_truth"]

    all_om_coords, medium_str = (None, "water")
    if args.geo:
        all_om_coords, medium_str = load_detector(args.geo)

    print(f"  Hit OMs:       {len(hit_oms)}")
    print(f"  Total photons: {hit_oms['n_photons'].sum()}")
    if all_om_coords is not None:
        print(f"  Total OMs:     {len(all_om_coords)}")

    fig = draw_event(hit_oms, all_om_coords, medium_str, mc_truth, event_idx, args)

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        logger.info("Saved: %s", args.out)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


# Run parse_args at module level so matplotlib.use() can be called before any
# display-dependent import.
args = parse_args()

# Resolve geo path relative to the repo root when a relative path isn't found
from pathlib import Path
if args.geo:
    _g = Path(args.geo)
    if not _g.is_absolute() and not _g.exists():
        REPO_ROOT = Path(__file__).resolve().parent.parent
        args.geo = str(REPO_ROOT / args.geo)

if __name__ == "__main__":
    main()
