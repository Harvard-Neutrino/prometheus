#!/usr/bin/env python3
"""mdom_pmt_view.py
Per-PMT hit-pattern visualisation for mDOM signal events.

Three panels:

  - **Left:** 3-D detector map.  Each hit module is a sphere whose colour
    encodes its first FADC pulse time and whose size is proportional to the
    total detected PE across all 24 PMTs.

  - **Middle:** Angular hit map on the 24-PMT Fibonacci sphere for the
    selected module shown as an azimuth × elevation sky map.  Circle area is
    proportional to detected PE; unhit PMTs are shown as small gray dots.  A
    gold ★ marks the photon source direction (vertex → module, reversed).
    The charge-weighted angular discriminant α and the dipole score D are
    printed in the panel title.

  - **Right:** Per-PMT FADC bubble waveform for the selected module (rows =
    individual PMTs, sorted earliest → latest, same bubble style as
    08_pulse_view.py).

Input files
-----------
``--pulses``  : mDOM pulse parquet from ``07_photon_to_pulses.py``
                (e.g. ``output/signal_genie_example_layer0_r10_pulses_mdom.parquet``)
``--photons`` : raw photon parquet from prometheus
                (e.g. ``output/signal_genie_example_layer0_r10.parquet``)

Usage
-----
Run from the repository root::

    .prometheus_env/bin/python examples/mdom_pmt_view.py \\
        --pulses  output/signal_genie_example_layer0_r10_pulses_mdom.parquet \\
        --photons output/signal_genie_example_layer0_r10.parquet

    # Choose a specific event and module
    .prometheus_env/bin/python examples/mdom_pmt_view.py \\
        --pulses  output/signal_genie_example_layer0_r10_pulses_mdom.parquet \\
        --photons output/signal_genie_example_layer0_r10.parquet \\
        --event 5 --module 3:12
"""

import argparse
import logging
import sys
from pathlib import Path

import awkward as ak
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PULSES  = str(REPO_ROOT / "output" / "signal_genie_example_layer0_r10_pulses_mdom.parquet")
_DEFAULT_PHOTONS = str(REPO_ROOT / "output" / "signal_genie_example_layer0_r10.parquet")

_N_PMTS = 24
_BG = "#06305a"
TIME_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "time_cmap",
    [(0.00, (0.00, 0.00, 0.00)),
     (0.50, (0.44, 0.18, 0.63)),
     (1.00, (1.00, 0.60, 0.10))],
)


# ---------------------------------------------------------------------------
# PMT geometry (must match 07_photon_to_pulses.py)
# ---------------------------------------------------------------------------

def _fibonacci_sphere(n: int) -> np.ndarray:
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n, dtype=float)
    theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    phi   = 2.0 * np.pi * i / golden
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


_PMT_DIRS: np.ndarray = _fibonacci_sphere(_N_PMTS)  # (24, 3)


# ---------------------------------------------------------------------------
# Angular discriminant observables
# ---------------------------------------------------------------------------

def angular_discriminant(q: np.ndarray, dirs: np.ndarray) -> float:
    """Charge-weighted mean pairwise opening angle α [deg].

    Only PMTs with q > 0 are included.
    """
    mask = q > 0
    q, dirs = q[mask], dirs[mask]
    n = len(q)
    if n < 2:
        return float("nan")
    numerator = denominator = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cos_ij = np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0)
            w = q[i] * q[j]
            numerator   += w * np.degrees(np.arccos(cos_ij))
            denominator += w
    return numerator / denominator if denominator > 0 else float("nan")


def dipole_score(q: np.ndarray, dirs: np.ndarray) -> float:
    """Charge-weighted dipole D = |Σ q_i r̂_i| / Σ q_i."""
    mask = q > 0
    Q = q[mask].sum()
    if Q == 0:
        return 0.0
    vec = (q[mask, None] * dirs[mask]).sum(axis=0)
    return float(np.linalg.norm(vec) / Q)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mDOM per-PMT hit-pattern view (3-panel figure)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pulses",  default=_DEFAULT_PULSES,  metavar="FILE",
                   help="mDOM pulse parquet from 07_photon_to_pulses.py")
    p.add_argument("--photons", default=_DEFAULT_PHOTONS, metavar="FILE",
                   help="Raw photon parquet from prometheus")
    p.add_argument("--event",  type=int, default=None, metavar="IDX",
                   help="0-based event index; defaults to event with most PE")
    p.add_argument("--module", type=str, default=None, metavar="STR:DOM",
                   help="string_id:sensor_id of module to show; defaults to "
                        "module with most PE in the selected event")
    p.add_argument("--out",  default="mdom_pmt_view.png", metavar="FILE",
                   help="Output image path ('' to skip saving)")
    p.add_argument("--show", action="store_true", default=False,
                   help="Open interactive matplotlib window")
    p.add_argument("--dpi",  type=int, default=150)
    return p.parse_args()


def load_event(pulses_path: str, photons_path: str,
               event_idx: int | None) -> tuple[int, object, object]:
    pulses  = ak.from_parquet(pulses_path)
    photons = ak.from_parquet(photons_path)
    if event_idx is None:
        total_pe = [int(ak.sum(pulses[i]["n_pe"])) for i in range(len(pulses))]
        event_idx = int(np.argmax(total_pe))
        logger.info("Auto-selected event %d  (%d total PE)",
                    event_idx, total_pe[event_idx])
    return event_idx, pulses[event_idx], photons[event_idx]


def get_sensor_positions(photons_row) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Return {(string_id, sensor_id): (x, y, z)} from one photon event row."""
    ph = ak.to_list(photons_row["photons"])
    pos: dict[tuple[int, int], tuple[float, float, float]] = {}
    for sid, mid, x, y, z in zip(
        ph["string_id"], ph["sensor_id"],
        ph["sensor_pos_x"], ph["sensor_pos_y"], ph["sensor_pos_z"],
    ):
        pos[(int(sid), int(mid))] = (float(x), float(y), float(z))
    return pos


def build_module_df(pulses_row,
                    pos_map: dict[tuple[int, int], tuple]) -> pd.DataFrame:
    """Per-module aggregate: sum PE / charge across all PMTs in each module."""
    string_ids = ak.to_list(pulses_row["string_id"])
    sensor_ids = ak.to_list(pulses_row["sensor_id"])
    n_pes      = ak.to_list(pulses_row["n_pe"])
    fadc_ts    = ak.to_list(pulses_row["fadc_t"])
    fadc_qs    = ak.to_list(pulses_row["fadc_q"])

    agg: dict[tuple[int, int], dict] = {}
    for sid, mid, n_pe, ft, fq in zip(string_ids, sensor_ids, n_pes, fadc_ts, fadc_qs):
        key = (int(sid), int(mid))
        fq_arr = np.asarray(fq, dtype=float)
        ft_arr = np.asarray(ft, dtype=float)
        if key not in agg:
            agg[key] = {"n_pe": 0, "total_q": 0.0, "first_t": np.inf}
        agg[key]["n_pe"]    += int(n_pe)
        agg[key]["total_q"] += float(fq_arr.sum())
        if len(ft_arr) > 0:
            agg[key]["first_t"] = min(agg[key]["first_t"], float(ft_arr.min()))

    rows = []
    for (sid, mid), v in agg.items():
        x, y, z = pos_map.get((sid, mid), (np.nan, np.nan, np.nan))
        rows.append({
            "string_id": sid, "sensor_id": mid,
            "x": x, "y": y, "z": z,
            "n_pe":    v["n_pe"],
            "total_q": v["total_q"],
            "first_t": v["first_t"] if v["first_t"] < np.inf else np.nan,
        })
    return pd.DataFrame(rows)


def build_pmt_df(pulses_row, string_id: int, sensor_id: int) -> pd.DataFrame:
    """Per-PMT records for a single module."""
    string_ids = np.asarray(ak.to_list(pulses_row["string_id"]))
    sensor_ids = np.asarray(ak.to_list(pulses_row["sensor_id"]))
    pmt_ids    = np.asarray(ak.to_list(pulses_row["pmt_id"]))
    n_pes      = np.asarray(ak.to_list(pulses_row["n_pe"]))
    fadc_ts    = ak.to_list(pulses_row["fadc_t"])
    fadc_qs    = ak.to_list(pulses_row["fadc_q"])

    mask = (string_ids == string_id) & (sensor_ids == sensor_id)
    rows = []
    for idx in np.where(mask)[0]:
        ft_arr = np.asarray(fadc_ts[idx], dtype=float)
        fq_arr = np.asarray(fadc_qs[idx], dtype=float)
        rows.append({
            "pmt_id":  int(pmt_ids[idx]),
            "n_pe":    int(n_pes[idx]),
            "total_q": float(fq_arr.sum()),
            "first_t": float(ft_arr.min()) if len(ft_arr) > 0 else np.nan,
            "fadc_t":  ft_arr,
            "fadc_q":  fq_arr,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("pmt_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detector_3d(ax, module_df: pd.DataFrame, event_idx: int,
                     vertex: tuple, sel_key: tuple[int, int]) -> None:
    """3-D scatter of hit modules; selected module highlighted."""
    active = module_df[module_df["total_q"] > 0].dropna(subset=["x"]).copy()
    if active.empty:
        return

    t_min = active["first_t"].min()
    t_max = active["first_t"].max()
    t_span = t_max - t_min if t_max > t_min else 1.0
    t_norm = ((active["first_t"] - t_min) / t_span).fillna(0.5).values
    colors = TIME_CMAP(t_norm)

    n = active["n_pe"].values.astype(float)
    n_max = n.max() if n.max() > 0 else 1.0
    sizes = 15.0 + (np.sqrt(n) / np.sqrt(n_max)) * 280.0

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

    ax.scatter(active["x"], active["y"], active["z"],
               s=sizes, c=colors, depthshade=False, zorder=2,
               edgecolors="none", alpha=0.85)

    # Highlight the selected module
    sel = module_df[(module_df["string_id"] == sel_key[0]) &
                    (module_df["sensor_id"] == sel_key[1])]
    if not sel.empty and not np.isnan(sel.iloc[0]["x"]):
        ax.scatter(sel["x"], sel["y"], sel["z"],
                   s=160, marker="D", c="#00FFAA",
                   edgecolors="white", linewidths=0.8,
                   depthshade=False, zorder=4, label="selected mDOM")

    # Vertex
    ax.scatter(*vertex, s=140, c="#FFD700", marker="*",
               edgecolors="white", linewidths=0.5,
               depthshade=False, zorder=5, label="vertex")

    n_pe_total = int(active["n_pe"].sum())
    ax.set_title(
        f"Event {event_idx}  |  {len(active)} modules  |  {n_pe_total:,} PE",
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


def draw_pmt_sphere(ax: plt.Axes,
                    pmt_df: pd.DataFrame,
                    source_dir: np.ndarray,
                    string_id: int, sensor_id: int,
                    alpha_val: float, dipole_val: float) -> None:
    """Azimuth × elevation sky map of the 24 PMTs for one module.

    Circle area ∝ detected PE.  Unhit PMTs are drawn as small gray rings.
    The photon source direction (pointing from module toward vertex) is
    marked with a gold ★.
    """
    ax.set_facecolor(_BG)

    # ---- Build 24-element PE and first_t arrays ----
    n_pe_arr  = np.zeros(_N_PMTS)
    first_t_arr = np.full(_N_PMTS, np.nan)
    if not pmt_df.empty:
        for _, row in pmt_df.iterrows():
            pid = int(row["pmt_id"])
            n_pe_arr[pid]   = row["n_pe"]
            first_t_arr[pid] = row["first_t"]

    # ---- Convert PMT directions to azimuth [deg] and elevation [deg] ----
    phi_deg  = np.degrees(np.arctan2(_PMT_DIRS[:, 1], _PMT_DIRS[:, 0]))   # [-180, 180]
    elev_deg = np.degrees(np.arcsin(np.clip(_PMT_DIRS[:, 2], -1.0, 1.0))) # [-90,  90]

    # ---- Source direction marker ----
    phi_src  = np.degrees(np.arctan2(source_dir[1], source_dir[0]))
    elev_src = np.degrees(np.arcsin(np.clip(source_dir[2], -1.0, 1.0)))

    # ---- Colour: first-hit time (only for hit PMTs) ----
    ft_valid = first_t_arr[~np.isnan(first_t_arr)]
    if len(ft_valid) > 1:
        t_min, t_max_val = ft_valid.min(), ft_valid.max()
    elif len(ft_valid) == 1:
        t_min, t_max_val = ft_valid[0] - 1.0, ft_valid[0] + 1.0
    else:
        t_min, t_max_val = 0.0, 1.0
    t_span = t_max_val - t_min if t_max_val > t_min else 1.0

    has_hit = n_pe_arr > 0

    # ---- Unhit PMTs ----
    if (~has_hit).any():
        ax.scatter(phi_deg[~has_hit], elev_deg[~has_hit],
                   s=25, c="none", edgecolors="gray",
                   linewidths=0.8, alpha=0.5, zorder=2)

    # ---- Hit PMTs ----
    if has_hit.any():
        n_max = n_pe_arr[has_hit].max()
        sizes = 60.0 + (n_pe_arr[has_hit] / n_max) * 350.0
        t_norm = (first_t_arr[has_hit] - t_min) / t_span
        t_norm = np.nan_to_num(t_norm, nan=0.5)
        colors = TIME_CMAP(np.clip(t_norm, 0.0, 1.0))

        ax.scatter(phi_deg[has_hit], elev_deg[has_hit],
                   s=sizes, c=colors,
                   edgecolors="white", linewidths=0.5,
                   alpha=0.9, zorder=3)

        # PE labels inside each hit PMT circle
        for pidx in np.where(has_hit)[0]:
            ax.text(phi_deg[pidx], elev_deg[pidx],
                    str(int(n_pe_arr[pidx])),
                    color="white", fontsize=6, ha="center", va="center",
                    fontweight="bold", zorder=4)

    # ---- Source direction ----
    ax.scatter([phi_src], [elev_src], s=250, marker="*",
               c="#FFD700", edgecolors="white", linewidths=0.5,
               zorder=5, label="source dir")

    # ---- Formatting ----
    ax.set_xlim(-185, 185)
    ax.set_ylim(-95, 95)
    ax.set_xticks(np.arange(-180, 181, 90))
    ax.set_yticks(np.arange(-90, 91, 45))
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_xlabel("Azimuth φ [°]", color="white", fontsize=8)
    ax.set_ylabel("Elevation ε [°]", color="white", fontsize=8)
    ax.grid(alpha=0.2, color="white", linestyle="--")

    n_hit  = int(has_hit.sum())
    Q_tot  = int(n_pe_arr.sum())
    a_str  = f"α = {alpha_val:.1f}°" if not np.isnan(alpha_val) else "α = —"
    d_str  = f"D = {dipole_val:.3f}" if not np.isnan(dipole_val) else "D = —"
    ax.set_title(
        f"str {string_id} / dom {sensor_id}  |  "
        f"{n_hit} / {_N_PMTS} PMTs  |  {Q_tot} PE\n"
        f"{a_str}     {d_str}",
        color="white", fontsize=8, pad=5,
    )

    leg = ax.legend(fontsize=7, framealpha=0.3, facecolor=_BG,
                    labelcolor="white", loc="lower right")
    leg.get_frame().set_edgecolor("white")

    # Size legend (bottom-left)
    for frac, label in ((0.2, "20% PE_max"), (0.6, "60%"), (1.0, "100%")):
        ax.scatter([], [], s=60 + frac * 350, c="gray", alpha=0.7,
                   edgecolors="white", linewidths=0.4, label=label)
    leg2 = ax.legend(fontsize=6, framealpha=0.3, facecolor=_BG,
                     labelcolor="white", loc="lower left",
                     title="PE / PE_max", title_fontsize=6)
    leg2.get_frame().set_edgecolor("white")
    leg2.get_title().set_color("white")


def draw_pmt_waveforms(ax: plt.Axes, pmt_df: pd.DataFrame) -> None:
    """Bubble waveform per hit PMT, sorted by first-hit time.

    Each row is one PMT; each bubble is one FADC bin (3.3 ns), area ∝ charge.
    Mirrors ``draw_waveforms`` in 08_pulse_view.py.
    """
    active = pmt_df[pmt_df["total_q"] > 0].sort_values("first_t").reset_index(drop=True)
    if active.empty:
        ax.text(0.5, 0.5, "No hits on this module",
                transform=ax.transAxes, ha="center", color="white", fontsize=9)
        return

    ax.set_facecolor(_BG)

    all_t = np.concatenate([r for r in active["fadc_t"] if len(r) > 0])
    all_q = np.concatenate([r for r in active["fadc_q"] if len(r) > 0])
    t_min_val, t_max_val = all_t.min(), all_t.max()
    q_max = all_q.max() if all_q.max() > 0 else 1.0
    norm  = mcolors.Normalize(vmin=t_min_val, vmax=t_max_val)

    for rank, row in active.iterrows():
        ft = np.asarray(row["fadc_t"])
        fq = np.asarray(row["fadc_q"])
        if len(ft) == 0:
            continue
        sizes = np.clip(fq / q_max * 280.0, 4.0, 400.0)
        ax.scatter(ft, np.full(len(ft), rank),
                   s=sizes, c=TIME_CMAP(norm(ft)),
                   linewidths=0, zorder=3, alpha=0.9)
        ax.axhline(rank, color="white", lw=0.3, alpha=0.2, zorder=1)

    n = len(active)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"PMT {int(r.pmt_id):02d}  ({int(r.n_pe)} PE)" for _, r in active.iterrows()],
        fontsize=7,
    )
    ax.set_xlim(t_min_val - 20.0, t_max_val + 20.0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Time  [ns]", fontsize=9, color="white")
    ax.set_title("Per-PMT FADC pulses  (earliest → latest)",
                 fontsize=9, color="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(axis="x", alpha=0.15, linestyle="--", color="white")

    # Charge bubble size legend
    for frac, label in ((0.1, "10 %"), (0.5, "50 %"), (1.0, "100 %")):
        ax.scatter([], [], s=frac * 280.0, c="gray", alpha=0.8, label=f"{label} q_max")
    leg = ax.legend(title="Charge", fontsize=7, title_fontsize=7,
                    loc="lower right", framealpha=0.3,
                    facecolor=_BG, labelcolor="white")
    leg.get_title().set_color("white")


# ---------------------------------------------------------------------------
# FADC step-curves (companion to pulse_view_curves)
# ---------------------------------------------------------------------------

# 12-colour palette cycling across up to 24 PMTs
_PMT_COLORS = [
    "#4fc3f7", "#ff8a65", "#a5d6a7", "#ce93d8", "#fff176", "#80cbc4",
    "#ffcc80", "#ef9a9a", "#b0bec5", "#80deea", "#f48fb1", "#c5e1a5",
]


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


def draw_pmt_curves(ax: plt.Axes, pmt_df: pd.DataFrame,
                    string_id: int, sensor_id: int,
                    n_pmts: int = 8) -> None:
    """Filled FADC step-curves for the N most-hit PMTs on one module.

    Each PMT gets its own colour; curves share the same axes so the relative
    timing and charge of different PMTs can be compared directly.
    """
    active = (
        pmt_df[pmt_df["total_q"] > 0]
        .nlargest(n_pmts, "n_pe")
        .sort_values("first_t")
        .reset_index(drop=True)
    )
    if active.empty:
        ax.text(0.5, 0.5, "No pulses", transform=ax.transAxes,
                ha="center", color="white", fontsize=9)
        return

    ax.set_facecolor(_BG)

    all_t = np.concatenate([r for r in active["fadc_t"] if len(r) > 0])
    t_lo = all_t.min() - 20.0 if len(all_t) else 0.0
    t_hi = all_t.max() + 20.0 if len(all_t) else 100.0

    sorted_q = active["total_q"].sort_values(ascending=False).values
    second_q = sorted_q[1] if len(sorted_q) > 1 else 0.0
    use_log = sorted_q[0] > 0 and (second_q == 0 or sorted_q[0] / second_q > 10)

    for i, (_, row) in enumerate(active.iterrows()):
        color = _PMT_COLORS[int(row["pmt_id"]) % len(_PMT_COLORS)]
        ft, fq = _expand_sparse_bins(
            np.asarray(row["fadc_t"]), np.asarray(row["fadc_q"])
        )
        label = f"PMT {int(row.pmt_id):02d}  ({int(row.n_pe)} PE)"
        if len(ft) > 0:
            ax.step(ft, fq, where="mid", color=color, lw=1.5, label=label)
            ax.fill_between(ft, fq, step="mid", color=color, alpha=0.20)
        else:
            ax.axhline(0, color=color, lw=1.0, linestyle="--", label=label)

    if use_log:
        ax.set_yscale("log", nonpositive="clip")
        ax.set_ylim(bottom=0.05)

    ax.set_xlim(t_lo, t_hi)
    ax.set_xlabel("Time  [ns]", color="white", fontsize=9)
    ax.set_ylabel("Charge  [PE]", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(alpha=0.15, linestyle="--", color="white")

    leg = ax.legend(fontsize=8, framealpha=0.3, facecolor=_BG,
                    labelcolor="white", loc="upper right",
                    ncols=2 if len(active) > 6 else 1)
    leg.get_frame().set_edgecolor("white")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    for path in (args.pulses, args.photons):
        if not Path(path).exists():
            logger.error("File not found: %s", path)
            sys.exit(1)

    # ---- Load data ----
    event_idx, pulses_row, photons_row = load_event(
        args.pulses, args.photons, args.event
    )

    mc = ak.to_list(photons_row["mc_truth"])
    vertex = (
        float(mc["initial_state_x"]),
        float(mc["initial_state_y"]),
        float(mc["initial_state_z"]),
    )

    pos_map   = get_sensor_positions(photons_row)
    module_df = build_module_df(pulses_row, pos_map)

    # ---- Select module ----
    if args.module is not None:
        parts = args.module.split(":")
        sel_key = (int(parts[0]), int(parts[1]))
    else:
        # Brightest module by total PE
        best = module_df.loc[module_df["n_pe"].idxmax()]
        sel_key = (int(best["string_id"]), int(best["sensor_id"]))
        logger.info("Auto-selected module str %d / dom %d  (%d PE)",
                    sel_key[0], sel_key[1], int(best["n_pe"]))

    pmt_df = build_pmt_df(pulses_row, sel_key[0], sel_key[1])

    # ---- Source direction for the selected module ----
    mod_pos = np.array(pos_map.get(sel_key, (0.0, 0.0, 0.0)))
    v_pos   = np.array(vertex)
    diff = v_pos - mod_pos
    norm = np.linalg.norm(diff)
    source_dir = diff / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    # ---- Compute discriminants ----
    q_arr   = np.zeros(_N_PMTS)
    dir_arr = _PMT_DIRS.copy()
    if not pmt_df.empty:
        for _, row in pmt_df.iterrows():
            q_arr[int(row["pmt_id"])] = row["n_pe"]
    alpha_val  = angular_discriminant(q_arr, dir_arr)
    dipole_val = dipole_score(q_arr, dir_arr)

    # ---- Summary ----
    n_hit_pmts = int((q_arr > 0).sum())
    Q_module   = int(q_arr.sum())
    mod_dist   = float(np.linalg.norm(mod_pos - v_pos))
    print(
        f"Event {event_idx}  |  str {sel_key[0]} / dom {sel_key[1]}"
        f"  |  {n_hit_pmts}/{_N_PMTS} PMTs  |  {Q_module} PE"
        f"  |  dist to vertex {mod_dist:.1f} m"
    )
    print(f"  α = {alpha_val:.2f}°   D = {dipole_val:.3f}")

    # ---- Layout ----
    fig = plt.figure(figsize=(16, 6), facecolor=_BG)
    gs  = gridspec.GridSpec(
        1, 3,
        width_ratios=[3, 2.2, 2.5],
        figure=fig,
        wspace=0.40,
        left=0.04, right=0.97,
    )

    ax3d = fig.add_subplot(gs[0], projection="3d", facecolor=_BG)
    ax3d.set_facecolor(_BG)
    ax_sphere = fig.add_subplot(gs[1])
    ax_wf     = fig.add_subplot(gs[2])

    draw_detector_3d(ax3d, module_df, event_idx, vertex, sel_key)
    draw_pmt_sphere(ax_sphere, pmt_df, source_dir,
                    sel_key[0], sel_key[1], alpha_val, dipole_val)
    draw_pmt_waveforms(ax_wf, pmt_df)

    # ---- Save / show main figure ----
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight",
                    facecolor=_BG)
        print(f"Saved: {out_path}")

    # ---- Curves figure (companion to pulse_view_curves.png) ----
    n_show = min(len(pmt_df[pmt_df['total_q'] > 0]), 8)
    fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor=_BG)
    fig2.suptitle(
        f"Event {event_idx}  —  per-PMT FADC charge vs time"
        f"  |  str {sel_key[0]} / dom {sel_key[1]}"
        f"  |  {n_hit_pmts}/{_N_PMTS} PMTs fired"
        f"  |  dist to vertex {mod_dist:.1f} m"
        f"  |  α = {alpha_val:.1f}°   D = {dipole_val:.3f}",
        color="white", fontsize=9,
    )
    draw_pmt_curves(ax2, pmt_df, sel_key[0], sel_key[1], n_pmts=n_show)

    if args.out:
        curves_path = out_path.with_stem(out_path.stem + "_curves")
        fig2.savefig(str(curves_path), dpi=args.dpi, bbox_inches="tight",
                     facecolor=_BG)
        print(f"Saved: {curves_path}")

    if args.show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)


if __name__ == "__main__":
    main()
