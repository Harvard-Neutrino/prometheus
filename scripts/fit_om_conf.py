#!/usr/bin/env python
"""Fit PPC om.conf shape parameters to a simulated effective area.

The input is a text file with two columns, cos(zenith) and effective area, for
light arriving from that zenith, averaged over azimuth. The fit varies ``beta``
and the zenith angle of each PMT ring. Rings mirrored about the equator share
one angle, PMTs at the poles stay there, and azimuths are kept. The overall
scale is profiled out, because PPC takes it from ``om.wv``.

Example:
    python scripts/fit_om_conf.py --om-conf om.conf --type 171 --curve lom16.txt
"""

import argparse
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from prometheus.detector import read_om_conf

N_AZIMUTH = 180
MIRROR_TOLERANCE_DEG = 0.01
MAX_RING_DEG = 89.9
MIN_BETA = 0.05
MAX_SHIFT_DEG = 25.0
BOUND_TOLERANCE = 1e-2


class FitResult(NamedTuple):
    """Best-fit shape and its residuals."""

    beta: float
    pmt_dirs: List[Tuple[float, float]]
    scale: float
    rms: float
    success: bool
    ring_angles: List[float]
    at_bound: List[str]


def segment_area(u: np.ndarray) -> np.ndarray:
    """Return the area of the unit-disc segment beyond distance ``u``."""
    angle = np.arccos(np.clip(u, -1.0, 1.0))
    return angle - np.sin(2 * angle) / 2


def pmt_response(x: np.ndarray, beta: float) -> np.ndarray:
    """Return PPC's response of one PMT (``itype::f`` in ini.cxx).

    Parameters
    ----------
    x : np.ndarray
        Cosine of the angle between the PMT axis and the direction to the light.
    beta : float
        Cap shape, from 0 (hemisphere) to 1 (flat disc).
    """
    x = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    response = np.maximum(x, 0.0)
    if beta >= 1:
        return response
    sin_theta = np.sqrt(1 - x * x)
    side = sin_theta > beta
    eta = beta / sin_theta[side]
    abs_x = np.abs(x[side])
    c = 1 - beta * beta
    cap = segment_area(eta) / c - segment_area(eta * abs_x / np.sqrt(c)) * abs_x
    response[side] += cap / np.pi
    return response


def zenith_profile(
    cos_zenith: np.ndarray, pmt_dirs: Sequence[Tuple[float, float]], beta: float
) -> np.ndarray:
    """Return the azimuth-averaged module response, summed over PMTs."""
    cos_zenith = np.asarray(cos_zenith, dtype=float)
    sin_zenith = np.sqrt(np.clip(1 - cos_zenith**2, 0, None))
    # An azimuth average does not depend on the PMT azimuth.
    azimuth = (np.arange(N_AZIMUTH) + 0.5) * 2 * np.pi / N_AZIMUTH
    total = np.zeros_like(cos_zenith)
    for zenith_deg, _ in pmt_dirs:
        zenith = np.radians(zenith_deg)
        x = np.cos(zenith) * cos_zenith[:, None]
        x = x + np.sin(zenith) * sin_zenith[:, None] * np.cos(azimuth)[None, :]
        total += pmt_response(x, beta).mean(axis=1)
    return total


def group_rings(pmt_dirs: Sequence[Tuple[float, float]]) -> Tuple[List[float], List[Optional[int]]]:
    """Group PMTs into mirrored rings.

    Returns
    -------
    tuple
        Free ring angles measured from the nearest pole, and for each PMT the
        index of its ring, or None if the PMT is fixed at a pole or the equator.
    """
    angles: List[float] = []
    index: List[Optional[int]] = []
    for zenith, _ in pmt_dirs:
        folded = min(zenith, 180 - zenith)
        if folded < MIRROR_TOLERANCE_DEG or abs(folded - 90) < MIRROR_TOLERANCE_DEG:
            index.append(None)
            continue
        for i, angle in enumerate(angles):
            if abs(angle - folded) < MIRROR_TOLERANCE_DEG:
                index.append(i)
                break
        else:
            index.append(len(angles))
            angles.append(folded)
    return angles, index


def apply_rings(
    pmt_dirs: Sequence[Tuple[float, float]],
    angles: Sequence[float],
    index: Sequence[Optional[int]],
) -> List[Tuple[float, float]]:
    """Return ``pmt_dirs`` with the ring angles replaced."""
    moved = []
    for (zenith, azimuth), ring in zip(pmt_dirs, index):
        if ring is not None:
            zenith = float(angles[ring]) if zenith < 90 else 180 - float(angles[ring])
        moved.append((zenith, azimuth))
    return moved


def keep_ring_order(
    start: Sequence[float], fitted: Sequence[float], index: Sequence[Optional[int]]
) -> List[float]:
    """Order fitted angles of equally populated rings as in ``start``.

    Swapping two such rings leaves the azimuth average unchanged, so the fit
    cannot tell them apart. Their azimuths differ, so the assignment matters.
    """
    ordered = [float(a) for a in fitted]
    counts = [index.count(i) for i in range(len(start))]
    for count in set(counts):
        rings = [i for i in range(len(start)) if counts[i] == count]
        by_start = sorted(rings, key=lambda i: start[i])
        values = sorted(ordered[i] for i in rings)
        for ring, value in zip(by_start, values):
            ordered[ring] = value
    return ordered


def best_scale(model: np.ndarray, area: np.ndarray) -> float:
    """Return the least-squares scale from ``model`` to ``area``."""
    return float(model @ area / (model @ model))


def shape_rms(
    cos_zenith: np.ndarray,
    area: np.ndarray,
    pmt_dirs: Sequence[Tuple[float, float]],
    beta: float,
) -> float:
    """Return the RMS residual over the mean area, after the best scale."""
    model = zenith_profile(cos_zenith, pmt_dirs, beta)
    residual = best_scale(model, area) * model - area
    return float(np.sqrt(np.mean(residual**2)) / area.mean())


def fit(
    cos_zenith: np.ndarray,
    area: np.ndarray,
    module: Dict,
    n_starts: int = 12,
    seed: int = 1,
    fixed_beta: Optional[float] = None,
    max_shift: float = MAX_SHIFT_DEG,
) -> FitResult:
    """Fit ``beta`` and the ring angles of ``module`` to ``area``.

    Several starting points are used because the residual has local minima. Ring
    angles stay within ``max_shift`` degrees of the angles in ``module``.
    """
    cos_zenith = np.asarray(cos_zenith, dtype=float)
    area = np.asarray(area, dtype=float)
    pmt_dirs = module["pmt_dirs"]
    angles, index = group_rings(pmt_dirs)
    n_free = len(angles) + (fixed_beta is None)
    if n_free == 0:
        raise ValueError("Nothing to fit: beta is fixed and no PMT is in a movable ring")
    if len(cos_zenith) <= n_free:
        raise ValueError(f"{len(cos_zenith)} points cannot constrain {n_free} parameters")

    def unpack(params: np.ndarray) -> Tuple[float, List[Tuple[float, float]]]:
        beta = params[-1] if fixed_beta is None else fixed_beta
        return beta, apply_rings(pmt_dirs, params[: len(angles)], index)

    def residual(params: np.ndarray) -> np.ndarray:
        beta, dirs = unpack(params)
        model = zenith_profile(cos_zenith, dirs, beta)
        return (best_scale(model, area) * model - area) / area.mean()

    lower = [max(0.0, a - max_shift) for a in angles]
    upper = [min(MAX_RING_DEG, a + max_shift) for a in angles]
    start = list(angles)
    if fixed_beta is None:
        lower.append(MIN_BETA)
        upper.append(1.0)
        start.append(module["beta"])
    lower_bound = np.array(lower)
    upper_bound = np.array(upper)
    rng = np.random.default_rng(seed)
    starts = [np.clip(start, lower_bound + 1e-3, upper_bound - 1e-3)]
    for _ in range(n_starts - 1):
        starts.append(rng.uniform(lower_bound, upper_bound))
    scale = np.where(upper_bound > 1.5, 10.0, 0.1)
    solutions = []
    for x0 in starts:
        solutions.append(
            least_squares(
                residual, x0, bounds=(lower_bound, upper_bound), x_scale=scale, diff_step=1e-4
            )
        )
    best = min(solutions, key=lambda s: s.cost)
    params = best.x.copy()
    names = [f"ring {a:g} deg" for a in angles] + ["beta"]
    at_bound = []
    for name, value, low, high in zip(names, params, lower_bound, upper_bound):
        tolerance = BOUND_TOLERANCE * (high - low)
        if value - low < tolerance or high - value < tolerance:
            at_bound.append(name)
    params[: len(angles)] = keep_ring_order(angles, params[: len(angles)], index)
    beta, dirs = unpack(params)
    amplitude = best_scale(zenith_profile(cos_zenith, dirs, beta), area)
    return FitResult(
        beta=float(beta),
        pmt_dirs=dirs,
        scale=amplitude,
        rms=shape_rms(cos_zenith, area, dirs, beta),
        success=bool(best.success),
        ring_angles=[float(a) for a in params[: len(angles)]],
        at_bound=at_bound,
    )


def format_module(name: str, module: Dict, result: FitResult) -> str:
    """Return an ``om.conf`` block for the fitted module."""
    head = [
        name,
        str(module["module_type"]),
        f"{module['area']:g}",
        f"{result.beta:.3f}",
        f"{module['Rr']:g}",
        f"{module['Rz']:g}",
        str(module["n_pmts"]),
    ]
    rows = [f"{zenith:.1f} {azimuth:g}" for zenith, azimuth in result.pmt_dirs]
    first = "\t".join(head + [rows[0]])
    if module["cable_azimuth"] is not None:
        first += f"\t{module['cable_azimuth']:g}"
    return "\n".join([first] + ["\t" * 7 + row for row in rows[1:]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--om-conf", required=True, help="om.conf with the starting module")
    parser.add_argument("--type", type=int, required=True, help="type ID in om.conf")
    parser.add_argument("--curve", required=True, help="cos(zenith), effective area")
    parser.add_argument("--name", default="OM", help="name in the printed om.conf block")
    parser.add_argument("--beta", type=float, help="fix beta instead of fitting it")
    parser.add_argument("--starts", type=int, default=12, help="number of starting points")
    parser.add_argument(
        "--max-shift", type=float, default=MAX_SHIFT_DEG, help="limit on ring angle changes [deg]"
    )
    args = parser.parse_args()

    types = read_om_conf(args.om_conf)
    if args.type not in types:
        parser.error(f"type {args.type} is not in {args.om_conf}")
    module = types[args.type]
    if module["beta"] < -1:
        parser.error("cylindrical modules (beta < -1) have no ring angles or beta to fit")
    cos_zenith, area = np.loadtxt(args.curve, usecols=(0, 1), unpack=True)

    before = shape_rms(cos_zenith, area, module["pmt_dirs"], module["beta"])
    result = fit(
        cos_zenith,
        area,
        module,
        args.starts,
        fixed_beta=args.beta,
        max_shift=args.max_shift,
    )
    print(f"# RMS residual / mean: {before:.4f} as given, {result.rms:.4f} fitted")
    rings = ", ".join(f"{angle:.2f}" for angle in result.ring_angles)
    print(f"# beta = {result.beta:.4f}, rings = {rings} deg")
    if not result.success:
        print("# WARNING: the fit did not converge")
    if result.at_bound:
        print(f"# WARNING: at a limit, so not determined by the fit: {', '.join(result.at_bound)}")
    print(format_module(args.name, module, result))


if __name__ == "__main__":
    main()
