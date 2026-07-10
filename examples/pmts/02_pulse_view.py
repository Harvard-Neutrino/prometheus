#!/usr/bin/env python3
"""02_pulse_view.py
Visualise digitised module responses for one event.

Produces two figures:

  - **3-D detector view.**  Each hit module is a sphere whose colour encodes
    its first FADC pulse time (black = earliest, purple = mid, orange =
    latest) and whose size is proportional to the number of signal
    photoelectrons (PE).

  - **FADC charge curves.**  Filled step-plot of charge vs time for the 3
    most-hit modules, one figure with all three curves overlaid.

Reads ``output/11_pulses.parquet`` (from
``examples/pmts/01_photon_to_pulses.py``) and ``output/10_photons.parquet``
(from an upstream Prometheus run, used only for sensor positions).

Usage
-----
Run from the repository root::

    .prometheus_env/bin/python examples/pmts/02_pulse_view.py

    # Specific event index
    .prometheus_env/bin/python examples/pmts/02_pulse_view.py --event 3
"""

import argparse
import logging
import sys
from pathlib import Path

import awkward as ak
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PULSES = str(REPO_ROOT / "examples" / "output" / "pmts" / "11_pulses.parquet")
_DEFAULT_PHOTONS = str(REPO_ROOT / "examples" / "output" / "pmts" / "10_photons.parquet")

# Shared colormap (same as 04_event_view)
TIME_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "time_cmap",
    [
        (0.00, (0.00, 0.00, 0.00)),
        (0.50, (0.44, 0.18, 0.63)),
        (1.00, (1.00, 0.60, 0.10)),
    ],
)
_BG = "#06305a"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-D + FADC waveform view of digitised module responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pulses", default=_DEFAULT_PULSES, metavar="FILE",
                   help="Path to 11_pulses.parquet.")
    p.add_argument("--photons", default=_DEFAULT_PHOTONS, metavar="FILE",
                   help="Path to 10_photons.parquet (sensor positions).")
    p.add_argument("--event", type=int, default=None, metavar="IDX",
                   help="0-based event index.  Defaults to brightest by signal PE.")
    p.add_argument("--out", default=str(REPO_ROOT / "examples" / "output" / "pmts" / "pulse_view.png"),
                   metavar="FILE", help="Output image.  Set to '' to skip saving.")
    p.add_argument("--show", action="store_true", default=False,
                   help="Open an interactive matplotlib window.")
    p.add_argument("--geo", default=None, metavar="FILE",
                   help="Geo file to show unhit modules as gray dots.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_detector(geo_path: str) -> np.ndarray | None:
    """Return (N, 3) array of all module positions from a geo file, or None on failure."""
    try:
        from prometheus.detector import detector_from_geo
        return detector_from_geo(geo_path).module_coords
    except Exception as exc:
        logger.warning("Could not load geo file %s: %s", geo_path, exc)
        return None

def load_event(pulses_path: str, photons_path: str, event_idx: int | None):
    pulses = ak.from_parquet(pulses_path)
    photons = ak.from_parquet(photons_path)

    if event_idx is None:
        total_pe = [int(ak.sum(pulses[i]["n_pe"])) for i in range(len(pulses))]
        event_idx = int(np.argmax(total_pe))
        logger.info("Auto-selected event %d (total signal PE = %d)",
                    event_idx, total_pe[event_idx])

    return event_idx, pulses[event_idx], photons[event_idx]


def build_module_df(photons_row, pulses_row) -> pd.DataFrame:
    """Build one row per module with positions and pulse summary.

    Parameters
    ----------
    photons_row :
        One event from 10_photons.parquet (awkward record).
    pulses_row :
        One event from 11_pulses.parquet (awkward record).

    Returns
    -------
    pandas.DataFrame with columns: string_id, sensor_id, x, y, z,
    n_photons, n_pe, total_q, first_t, fadc_t (ndarray), fadc_q (ndarray).
    """
    ph = ak.to_list(photons_row["photons"])

    # Deduplicate sensor positions and count raw photon hits per module
    pos_map: dict[tuple[int, int], tuple[float, float, float]] = {}
    photon_counts: dict[tuple[int, int], int] = {}
    for sid, mid, x, y, z in zip(
        ph["string_id"], ph["sensor_id"],
        ph["sensor_pos_x"], ph["sensor_pos_y"], ph["sensor_pos_z"],
    ):
        key = (int(sid), int(mid))
        photon_counts[key] = photon_counts.get(key, 0) + 1
        if key not in pos_map:
            pos_map[key] = (float(x), float(y), float(z))

    string_ids = ak.to_list(pulses_row["string_id"])
    sensor_ids = ak.to_list(pulses_row["sensor_id"])
    n_pes = ak.to_list(pulses_row["n_pe"])
    fadc_ts = ak.to_list(pulses_row["fadc_t"])
    fadc_qs = ak.to_list(pulses_row["fadc_q"])

    rows = []
    for sid, mid, n_pe, ft, fq in zip(string_ids, sensor_ids, n_pes, fadc_ts, fadc_qs):
        ft_arr = np.asarray(ft, dtype=float)
        fq_arr = np.asarray(fq, dtype=float)
        x, y, z = pos_map.get((int(sid), int(mid)), (np.nan, np.nan, np.nan))
        key = (int(sid), int(mid))
        rows.append({
            "string_id":  int(sid),
            "sensor_id":  int(mid),
            "x": x, "y": y, "z": z,
            "n_photons":  photon_counts.get(key, 0),
            "n_pe":       int(n_pe),
            "total_q":    float(fq_arr.sum()),
            "first_t":    float(ft_arr.min()) if len(ft_arr) > 0 else np.nan,
            "fadc_t":     ft_arr,
            "fadc_q":     fq_arr,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_3d(ax: "Axes3D", df: pd.DataFrame, event_idx: int,
            n_photons_total: int, n_pe_total: int,
            vertex: tuple[float, float, float] | None = None,
            min_dist: float | None = None,
            all_module_coords: np.ndarray | None = None) -> None:
    """3-D scatter: colour = first pulse time, size ∝ signal PE."""
    active = df[df["total_q"] > 0].copy()
    if active.empty:
        return

    t_min = active["first_t"].min()
    t_max = active["first_t"].max()
    t_span = t_max - t_min if t_max > t_min else 1.0
    t_norm = ((active["first_t"] - t_min) / t_span).fillna(0.5).values
    colors = TIME_CMAP(t_norm)

    n = active["n_pe"].values.astype(float)
    n_max = n.max() if n.max() > 0 else 1.0
    sizes = 15.0 + (np.sqrt(n) / np.sqrt(n_max)) * 380.0

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.tick_params(colors="white", labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
    ax.set_xlabel("x  [m]", labelpad=4)
    ax.set_ylabel("y  [m]", labelpad=4)
    ax.set_zlabel("z  [m]", labelpad=4)

    # Draw order: unhit modules → hit modules → vertex (each layer on top of previous)
    if all_module_coords is not None:
        ax.scatter(all_module_coords[:, 0], all_module_coords[:, 1], all_module_coords[:, 2],
                   s=2, c="black", alpha=0.3, depthshade=False, zorder=1)

    ax.scatter(active["x"], active["y"], active["z"],
               s=sizes, c=colors, depthshade=False, zorder=2, edgecolors="none")

    if vertex is not None:
        ax.scatter(*vertex, s=120, c="#FFD700", marker="*",
                   edgecolors="white", linewidths=0.5,
                   depthshade=False, zorder=3, label="vertex")

    eff = n_pe_total / n_photons_total if n_photons_total > 0 else 0.0
    dist_str = f"  |  closest module {min_dist:.1f} m" if min_dist is not None else ""
    ax.set_title(
        f"Event {event_idx}  |  {len(active)} modules  |  "
        f"{n_photons_total:,} photons  →  {n_pe_total:,} PE  (QE {eff:.0%})"
        f"{dist_str}",
        color="white", fontsize=9, pad=6,
    )

    sm = plt.cm.ScalarMappable(
        cmap=TIME_CMAP, norm=mcolors.Normalize(vmin=t_min, vmax=t_max)
    )
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, pad=0.07, shrink=0.55, aspect=18)
    cbar.set_label("First pulse time  [ns]", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")


_CURVE_COLORS = ["#4fc3f7", "#ff8a65", "#a5d6a7"]  # light-blue, orange, green


def _expand_sparse_bins(t: np.ndarray, q: np.ndarray,
                        bin_width: float = 3.3) -> tuple[np.ndarray, np.ndarray]:
    """Insert explicit zero samples at gap boundaries for a clean step display."""
    if len(t) == 0:
        return np.array([]), np.array([])
    t_out: list[float] = []
    q_out: list[float] = []
    for i in range(len(t)):
        if i > 0 and t[i] - t[i - 1] > bin_width * 1.5:
            t_out.append(t[i - 1] + bin_width)
            q_out.append(0.0)
            t_out.append(t[i] - bin_width)
            q_out.append(0.0)
        t_out.append(float(t[i]))
        q_out.append(float(q[i]))
    return np.array(t_out), np.array(q_out)


def draw_pe_curves(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Filled step-plot of FADC charge vs time for the 3 most-hit modules.

    All three curves are drawn on the same axes.  Modules with zero detected
    PE appear as flat baselines.

    Parameters
    ----------
    ax :
        Single Axes to draw on.
    df :
        Module DataFrame from ``build_module_df``.
    """
    top3 = df.nlargest(3, "n_photons").reset_index(drop=True)

    all_t = np.concatenate([r for r in top3["fadc_t"] if len(r) > 0])
    t_lo = all_t.min() - 20.0 if len(all_t) else 0.0
    t_hi = all_t.max() + 20.0 if len(all_t) else 100.0

    sorted_q = top3["total_q"].sort_values(ascending=False).values
    second_q = sorted_q[1] if len(sorted_q) > 1 else 0.0
    use_log = sorted_q[0] > 0 and (second_q == 0 or sorted_q[0] / second_q > 10)

    for i, (_, row) in enumerate(top3.iterrows()):
        color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
        ft, fq = _expand_sparse_bins(row["fadc_t"], row["fadc_q"])
        label = (
            f"str {int(row.string_id)} / dom {int(row.sensor_id)}"
            f"  ({row.n_photons:,} ph → {row.n_pe:,} PE)"
        )
        if len(ft) > 0:
            ax.step(ft, fq, where="mid", color=color, lw=1.5, label=label)
            ax.fill_between(ft, fq, step="mid", color=color, alpha=0.25)
        else:
            ax.axhline(0, color=color, lw=1.0, linestyle="--", label=label)

    if use_log:
        ax.set_yscale("log", nonpositive="clip")
        ax.set_ylim(bottom=0.05)

    ax.set_facecolor(_BG)
    ax.set_xlim(t_lo, t_hi)
    ax.set_xlabel("Time  [ns]", color="white", fontsize=9)
    ax.set_ylabel("Charge  [PE]", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(alpha=0.15, linestyle="--", color="white")

    leg = ax.legend(fontsize=8, framealpha=0.3, facecolor=_BG,
                    labelcolor="white", loc="upper right")
    leg.get_frame().set_edgecolor("white")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not args.show:
        matplotlib.use("Agg")

    for path in (args.pulses, args.photons):
        if not Path(path).exists():
            logger.error("File not found: %s", path)
            logger.info("Run an upstream Prometheus injection/propagation "
                        "script (e.g. examples/genie/01_genie_injection.py) "
                        "followed by examples/pmts/01_photon_to_pulses.py first.")
            sys.exit(1)

    event_idx, pulses_row, photons_row = load_event(
        args.pulses, args.photons, args.event
    )
    df = build_module_df(photons_row, pulses_row)

    mc = photons_row["mc_truth"]
    vertex = (
        float(mc["initial_state_x"]),
        float(mc["initial_state_y"]),
        float(mc["initial_state_z"]),
    )
    df["dist_to_vertex"] = np.sqrt(
        (df["x"] - vertex[0]) ** 2
        + (df["y"] - vertex[1]) ** 2
        + (df["z"] - vertex[2]) ** 2
    )
    min_dist = float(df["dist_to_vertex"].min())

    n_photons_total = int(df["n_photons"].sum())
    n_pe_total = int(df["n_pe"].sum())
    print(
        f"Event {event_idx}:  {len(df)} modules  |  "
        f"{n_photons_total:,} photons  →  {n_pe_total:,} signal PEs  |  "
        f"closest module {min_dist:.1f} m"
    )

    fig = plt.figure(figsize=(8, 6), facecolor=_BG)
    ax3d = fig.add_subplot(projection="3d", facecolor=_BG)
    ax3d.set_facecolor(_BG)

    all_module_coords = load_detector(args.geo) if args.geo else None

    draw_3d(ax3d, df, event_idx, n_photons_total, n_pe_total,
            vertex=vertex, min_dist=min_dist,
            all_module_coords=all_module_coords)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight",
                    facecolor=_BG)
        print(f"Saved: {out_path}")

    # Second figure: PE vs time curves for the 3 most-hit modules
    fig2, ax_curves = plt.subplots(figsize=(10, 4), facecolor=_BG)
    fig2.suptitle(
        f"Event {event_idx}  —  FADC charge vs time (top 3 modules by photon count)"
        f"  |  closest module {min_dist:.1f} m from vertex",
        color="white", fontsize=10,
    )
    draw_pe_curves(ax_curves, df)

    if args.out:
        curves_path = out_path.with_stem(out_path.stem + "_curves")
        fig2.savefig(str(curves_path), dpi=args.dpi, bbox_inches="tight",
                     facecolor=_BG)
        print(f"Saved: {curves_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)


# Parse at module level so matplotlib.use() fires before any display import.
args = parse_args()

if __name__ == "__main__":
    main()
