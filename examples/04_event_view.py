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

Multi-PMT modules
-----------------
With ``--om-conf`` and ``--om-type`` the hit OMs are drawn as enlarged module
shapes, and a second row shows one module from the side the light comes from
and from the opposite side.  Each PMT is a disc colored by its number of
photons; the arrow is the mean photon travel direction.  This needs ``pmt_id``
and photon directions in the parquet file (``output_mode`` "photon" or
"full", or ``05_lom_ppc.py --out``).

The module is drawn either as the pressure vessel, approximated by a capsule
with the ``Rr`` and ``Rz`` of ``om.conf``, or as the surface PPC uses (a
spheroid, or a cylinder if ``Rz`` is negative).  PMTs are placed where their
``om.conf`` direction crosses that surface.  For the LOM-16 those directions
are effective values fitted to the module response, not the mechanical tilts.

Usage
-----
python examples/04_event_view.py examples/output/1_photons.parquet

python examples/04_event_view.py examples/output/lom16.parquet \\
    --om-conf resources/optical_modules/lom16/om.conf --om-type 171

python examples/04_event_view.py examples/output/1_photons.parquet \\
    --geo resources/geofiles/demo_water.geo \\
    --event 2 \\
    --out event.png \\
    --show
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

# ── Custom colormap: black → purple → orange ─────────────────────────────────
_CMAP_COLORS = [
    (0.00, (0.00, 0.00, 0.00)),  # t_min  → black
    (0.50, (0.44, 0.18, 0.63)),  # t_mid  → purple
    (1.00, (1.00, 0.60, 0.10)),  # t_max  → orange
]
TIME_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "time_cmap",
    [(v, c) for v, c in _CMAP_COLORS],
)

# Background fill colors
_BG_WATER = "#06305a"  # deep water blue
_BG_ICE = "#cde4f5"  # pale ice blue


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
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
        default=str(REPO_ROOT / "examples" / "output" / "event_view.png"),
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
    p.add_argument("--om-conf", default=None, metavar="FILE", help="PPC om.conf file.")
    p.add_argument(
        "--om-type", type=int, default=None, metavar="ID", help="Module type ID in --om-conf."
    )
    p.add_argument(
        "--module",
        default=None,
        metavar="STR:OM",
        help="Module to show in detail.  Defaults to the one with most photons.",
    )
    p.add_argument(
        "--shape",
        choices=("vessel", "ppc"),
        default="vessel",
        help="Draw the pressure vessel (capsule) or the surface PPC simulates.",
    )
    p.add_argument(
        "--pmt-radius",
        type=float,
        default=0.051,
        metavar="M",
        help="Radius of the drawn PMT discs [m].  Display only.",
    )
    args = p.parse_args(argv)
    if (args.om_conf is None) != (args.om_type is None):
        p.error("--om-conf and --om-type must be given together")
    return args


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
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
        medium_str = str(det.medium).lower()  # e.g. 'medium.water'
        return det.module_coords, medium_str
    except Exception as e:
        logger.warning("Could not load geo file %s: %s. Unhit OMs hidden.", geo_path, e)
        return None, None


# ── Multi-PMT modules ─────────────────────────────────────────────────────────
def unit_vector(zenith, azimuth) -> np.ndarray:
    """Return unit vectors for angles in radians."""
    return np.stack(
        [np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)],
        axis=-1,
    )


def load_om(om_conf: str, om_type: int) -> dict:
    """Return the om.conf entry for one module type."""
    from prometheus.detector import read_om_conf

    types = read_om_conf(om_conf)
    if om_type not in types:
        raise SystemExit(f"Type {om_type} is not in {om_conf}; available: {sorted(types)}")
    return types[om_type]


def profile(om: dict, shape: str, n: int = 40) -> tuple:
    """Return (radius, height) along the module outline, from bottom to top [m]."""
    rr, rz = om["Rr"], abs(om["Rz"])
    if om["Rz"] < 0:
        return np.array([0.0, rr, rr, 0.0]), np.array([-rz, -rz, rz, rz])
    if shape == "vessel" and rz > rr:
        angle = np.linspace(0, np.pi / 2, n)
        radius = rr * np.sin(angle)
        height = rz - rr + rr * np.cos(angle)
        return np.r_[radius, radius[::-1]], np.r_[-height, height[::-1]]
    angle = np.linspace(np.pi, 0, 2 * n)
    return rr * np.sin(angle), rz * np.cos(angle)


def surface_distance(om: dict, shape: str, directions: np.ndarray) -> np.ndarray:
    """Return the distance from the module center to its surface [m]."""
    rr, rz = om["Rr"], abs(om["Rz"])
    rho = np.hypot(directions[:, 0], directions[:, 1])
    height = np.abs(directions[:, 2])
    with np.errstate(divide="ignore"):
        side = np.where(rho > 0, rr / rho, np.inf)
        top = np.where(height > 0, rz / height, np.inf)
    if om["Rz"] < 0:
        return np.minimum(side, top)
    if shape == "vessel" and rz > rr:
        half = rz - rr
        cap = height * half + np.sqrt(rr**2 - (rho * half) ** 2)
        return np.where(side * height <= half, side, cap)
    return 1 / np.sqrt((rho / rr) ** 2 + (height / rz) ** 2)


def draw_shell(ax, om, shape, center, scale, color, alpha, n_azimuth=24) -> None:
    """Draw the module surface."""
    radius, height = profile(om, shape)
    azimuth = np.linspace(0, 2 * np.pi, n_azimuth)
    ax.plot_surface(
        center[0] + scale * np.outer(radius, np.cos(azimuth)),
        center[1] + scale * np.outer(radius, np.sin(azimuth)),
        center[2] + scale * np.outer(height, np.ones_like(azimuth)),
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=True,
        rcount=len(radius),
        ccount=n_azimuth,
    )


def disc(center: np.ndarray, axis: np.ndarray, radius: float, n: int = 40) -> np.ndarray:
    """Return the outline of a disc with the given axis."""
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + radius * (np.outer(np.cos(angle), u) + np.outer(np.sin(angle), v))


def select_module(photons: dict, requested) -> tuple:
    """Return the (string, OM) to show and the mask of its photons."""
    strings = np.asarray(photons["string_id"], dtype=int)
    oms = np.asarray(photons["sensor_id"], dtype=int)
    if requested is None:
        keys, counts = np.unique(np.column_stack([strings, oms]), axis=0, return_counts=True)
        key = tuple(int(i) for i in keys[np.argmax(counts)])
    else:
        try:
            string, om = requested.split(":")
            key = (int(string), int(om))
        except ValueError:
            raise SystemExit(f"--module must be STR:OM, got {requested!r}") from None
    mask = (strings == key[0]) & (oms == key[1])
    if not mask.any():
        raise SystemExit(f"Module {key[0]}:{key[1]} has no photons in this event")
    return key, mask


def pmt_counts(photons: dict, mask: np.ndarray, n_pmts: int) -> np.ndarray:
    """Return the number of photons on each PMT of the masked module."""
    if "pmt_id" not in photons or photons["pmt_id"] is None:
        raise SystemExit("The parquet file has no pmt_id; were the modules multi-PMT?")
    ids = np.asarray(photons["pmt_id"], dtype=float)[mask]
    if np.isnan(ids).any():
        raise SystemExit("Some photons have no PMT; was PPC run in multi-PMT mode?")
    ids = ids.astype(int)
    if ids.max() >= n_pmts:
        raise SystemExit(f"pmt_id {ids.max()} does not fit a module with {n_pmts} PMTs")
    return np.bincount(ids, minlength=n_pmts)


def mean_photon_direction(photons: dict, mask: np.ndarray) -> np.ndarray:
    """Return the unit mean travel direction of the masked photons."""
    if photons.get("photon_zenith") is None:
        raise SystemExit("The parquet file has no photon directions; use output_mode 'photon'")
    zenith = np.asarray(photons["photon_zenith"], dtype=float)[mask]
    azimuth = np.asarray(photons["photon_azimuth"], dtype=float)[mask]
    mean = unit_vector(zenith, azimuth).mean(axis=0)
    norm = np.linalg.norm(mean)
    if not norm > 0:
        raise SystemExit("The photons have no mean direction")
    return mean / norm


def draw_module(ax, om, counts, travel, view, shape, pmt_radius, norm, cmap, text_color) -> None:
    """Draw one module seen from direction ``view``."""
    zenith, azimuth = np.radians(np.array(om["pmt_dirs"], dtype=float)).T
    axes = unit_vector(zenith, azimuth)
    # Sink each disc so that its rim, not its center, meets the rounded surface.
    depth = pmt_radius**2 / (2 * om["Rr"])
    centers = axes * (surface_distance(om, shape, axes) - depth)[:, None]

    ax.computed_zorder = False
    draw_shell(ax, om, shape, np.zeros(3), 1.0, "lightsteelblue", 0.35, n_azimuth=48)
    # matplotlib does not hide surfaces behind others, so only PMTs facing the
    # viewer are drawn, from back to front.
    for i in np.argsort(axes @ view):
        facing = axes[i] @ view
        if facing <= 0.05:
            continue
        outline = disc(centers[i], axes[i], pmt_radius)
        ax.add_collection3d(
            Poly3DCollection(
                [outline],
                facecolor=cmap(norm(counts[i])),
                edgecolor="black",
                linewidth=0.8,
                zorder=3 + facing,
            )
        )
        if facing > 0.3:
            rgb = np.array(cmap(norm(counts[i]))[:3])
            label_color = "black" if rgb @ [0.299, 0.587, 0.114] > 0.5 else "white"
            ax.text(
                *(centers[i] + 0.01 * view),
                f"{i}\n{counts[i]}",
                ha="center",
                va="center",
                fontsize=7,
                color=label_color,
                zorder=10,
            )

    reach = 1.3 * max(om["Rr"], abs(om["Rz"]))
    if travel @ view < 0:
        # Draw the arrow on the side the light comes from, ending at the surface.
        tip = -travel * surface_distance(om, shape, -travel[None])[0] * 1.1
        tail = tip - travel * 0.22
        ax.quiver(*tail, *(tip - tail), color="crimson", linewidth=2.5, zorder=11)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(-reach, reach)
    ax.set_box_aspect((1, 1, 1), zoom=1.5)
    ax.set_axis_off()
    ax.view_init(elev=np.degrees(np.arcsin(view[2])), azim=np.degrees(np.arctan2(view[1], view[0])))
    ax.set_proj_type("ortho")


def side_views(travel: np.ndarray) -> tuple:
    """Return horizontal view directions toward and away from the light.

    Vertical light has no horizontal direction, so +x is used instead.
    """
    horizontal = -np.array([travel[0], travel[1], 0.0])
    length = np.linalg.norm(horizontal)
    horizontal = horizontal / length if length > 0.1 else np.array([1.0, 0.0, 0.0])
    # Turn the view away from the light so that the arrow is not seen end-on.
    turn, tilt = np.radians(35), np.radians(12)
    sideways = np.array([-horizontal[1], horizontal[0], 0.0])
    turned = np.cos(turn) * horizontal + np.sin(turn) * sideways
    up = np.array([0.0, 0.0, np.sin(tilt)])
    return np.cos(tilt) * turned + up, -np.cos(tilt) * turned + up


def draw_om_shapes(ax, hit_oms, all_om_coords, om, shape, colors, size_fraction) -> None:
    """Draw hit OMs as enlarged modules on equally scaled axes."""
    hit = hit_oms[["x", "y", "z"]].values
    coords = hit if all_om_coords is None else np.vstack([hit, all_om_coords])
    span = np.ptp(coords, axis=0)
    size = max(span.max(), 1.0)
    largest = 0.03 * size / max(om["Rr"], abs(om["Rz"]))
    for center, color, fraction in zip(hit, colors, size_fraction):
        draw_shell(ax, om, shape, center, largest * (0.25 + 0.75 * fraction), color, 1.0)
    middle = (coords.min(axis=0) + coords.max(axis=0)) / 2
    # Keep some width for a single string.
    half = np.maximum(span, 0.35 * size) / 2 + 0.04 * size
    ax.locator_params(axis="x", nbins=3)
    ax.locator_params(axis="y", nbins=3)
    ax.set_xlim(middle[0] - half[0], middle[0] + half[0])
    ax.set_ylim(middle[1] - half[1], middle[1] + half[1])
    ax.set_zlim(middle[2] - half[2], middle[2] + half[2])
    ax.set_box_aspect(half, zoom=1.1)


def add_module_panels(fig, grid, photons, om, args, text_color) -> tuple:
    """Draw the two views of the selected module; return its key and counts."""
    key, mask = select_module(photons, args.module)
    counts = pmt_counts(photons, mask, om["n_pmts"])
    travel = mean_photon_direction(photons, mask)
    norm = mcolors.Normalize(vmin=0, vmax=max(int(counts.max()), 1))
    cmap = plt.get_cmap("viridis")
    views = side_views(travel)
    axes = []
    for column, (view, title) in enumerate(zip(views, ("side facing the light", "far side"))):
        ax = fig.add_subplot(grid[1, column], projection="3d", facecolor="none")
        draw_module(
            ax, om, counts, travel, view, args.shape, args.pmt_radius, norm, cmap, text_color
        )
        total = int(counts.sum())
        ax.set_title(
            f"String {key[0]}, OM {key[1]}: {title}\n{total} photons, {om['n_pmts']} PMTs",
            color=text_color,
            fontsize=9,
        )
        axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.7, aspect=18, pad=0.02
    )
    cbar.set_label("Photons per PMT (disc label: PMT, photons)", color=text_color, fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=text_color)
    return key, counts


# ── Plotting ──────────────────────────────────────────────────────────────────
def draw_event(
    hit_oms: pd.DataFrame,
    all_om_coords,  # np.ndarray (N, 3) or None
    medium_str: str,
    mc_truth: dict,
    event_idx: int,
    args: argparse.Namespace,
    photons: dict = None,
    om: dict = None,
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
    if om is None:
        fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
        ax: Axes3D = fig.add_subplot(111, projection="3d", facecolor=bg_color)
    else:
        fig = plt.figure(figsize=(11, 13), facecolor=bg_color)
        grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1), hspace=0.2)
        ax = fig.add_subplot(grid[0, :], projection="3d", facecolor=bg_color)
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
    if om is None:
        ax.scatter(
            hit_oms["x"].values,
            hit_oms["y"].values,
            hit_oms["z"].values,
            s=sizes,
            c=colors,
            depthshade=False,
            zorder=2,
            edgecolors="none",
        )
    else:
        draw_om_shapes(ax, hit_oms, all_om_coords, om, args.shape, colors, n_norm)

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
    pname = {
        12: r"$\nu_e$",
        -12: r"$\bar\nu_e$",
        14: r"$\nu_\mu$",
        -14: r"$\bar\nu_\mu$",
        16: r"$\nu_\tau$",
        -16: r"$\bar\nu_\tau$",
    }.get(ptype, f"PDG {ptype}")
    # The event type is only known to be a neutrino interaction for neutrinos.
    label = f"{pname}  CC" if abs(ptype) in (12, 14, 16) else pname
    total_hits = int(hit_oms["n_photons"].sum())
    n_oms_hit = len(hit_oms)

    title = (
        f"Event {event_idx}  |  {label}  |  "
        f"E = {energy_gev:.1f} GeV  |  "
        f"{total_hits} photons on {n_oms_hit} OMs  |  {fg_label}"
    )
    ax.set_title(title, color=text_color, fontsize=9, pad=8)

    # ── Size legend ───────────────────────────────────────────────────────────
    legend_counts = [1, 10, 100]
    legend_handles = []
    for nc in legend_counts if om is None else []:
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

    if om is None:
        fig.tight_layout()
    else:
        key, counts = add_module_panels(fig, grid, photons, om, args, text_color)
        print(f"  Module {key[0]}:{key[1]}: {counts.sum()} photons, per PMT {counts.tolist()}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    # Resolve geo path relative to the repo root when a relative path isn't found
    if args.geo:
        geo = Path(args.geo)
        if not geo.is_absolute() and not geo.exists():
            args.geo = str(REPO_ROOT / args.geo)
    if not args.show:
        matplotlib.use("Agg")

    df = load_parquet(args.parquet)
    if df.empty:
        logger.error("Parquet file %s contains no events.", args.parquet)
        sys.exit(1)

    event_idx = args.event if args.event is not None else brightest_event(df)
    if event_idx < 0 or event_idx >= len(df):
        logger.error("--event %s out of range (0–%s).", event_idx, len(df) - 1)
        sys.exit(1)

    logger.info("Visualising event %s (%s events in file)", event_idx, len(df))

    row = df.iloc[event_idx]
    hit_oms = aggregate_hits(row["photons"])
    mc_truth = row["mc_truth"]

    all_om_coords, medium_str = (None, "ice" if args.om_conf else "water")
    if args.geo:
        all_om_coords, medium_str = load_detector(args.geo)

    print(f"  Hit OMs:       {len(hit_oms)}")
    print(f"  Total photons: {hit_oms['n_photons'].sum()}")
    if all_om_coords is not None:
        print(f"  Total OMs:     {len(all_om_coords)}")

    om = load_om(args.om_conf, args.om_type) if args.om_conf else None
    fig = draw_event(
        hit_oms, all_om_coords, medium_str, mc_truth, event_idx, args, row["photons"], om
    )

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        logger.info("Saved: %s", args.out)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
