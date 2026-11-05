"""Photon Monte Carlo training-data generation for the shape/counts models.

Runs photon MC in ANTARES/KM3NeT Mediterranean water with absorption. DOM
placed at origin (radius 0.30 m); point sources at logarithmically spaced
distances from d_min to d_max.

Angle convention
----------------
The propagator (``olympus/event_generation/photon_propagation/utils.py``)
evaluates both models at the VIEWING angle, the angle between the source
direction and the source-to-module vector, and multiplies the returned counts
fraction by every photon the source emits (``norm_flow_photons.py``). The
labels here are built in that convention: the counts label is the fraction of
ALL photons a source emits that reach the DOM, and each row is keyed by a
viewing angle.

Emission
--------
``emission="cherenkov"`` (default): the source emits with the angular profile
of an electromagnetic cascade around its axis (Radel & Wiebusch, arXiv:1210.5140,
:func:`hyperion.utils.cherenkov_ang_dist`), so the light peaks near the
Cherenkov angle. Every (distance, viewing angle) pair is a separate MC run with
the source axis tilted by the viewing angle away from the source-to-DOM line,
and ``n_photons`` photons are emitted per pair.

``emission="isotropic"``: the source has no preferred direction, so the
fraction does not depend on the viewing angle. One MC run per distance is
written once per point of the viewing-angle grid with the same label, and the
network learns a flat angle dependence. This conserves the total light but not
the per-module pattern of a real cascade.

An earlier version binned photons by their own emission angle and divided
by the photons emitted into that bin. Read at the viewing angle and scaled
by all photons, that overstates the light on modules near the source axis
by the inverse solid angle of the bin, hundreds of times at 50 m.

Each photon accumulates an absorption survival weight
``w = prod_steps exp(-step_size / abs_len(λ))``. This weight-tracking
approach avoids killing photons early (improving statistics at large
distances) and is equivalent to the absorption-corrected expected photon
count.

Importance sampling
-------------------
The DOM subtends 2.5e-7 of the sphere at 300 m, so isotropic emission alone
leaves every far distance empty (25 hits per 2M photons at 50 m).
Directions are therefore drawn from a mixture: with probability
``aim_fraction`` uniformly inside a narrow cone around the source-to-DOM
axis, otherwise from the emission profile itself. Every photon carries the
weight ``p_emit / q_mix`` of its direction. The expectation of every label is
unchanged for any mixture (the proposal covers the whole sphere); the cone
supplies the direct photons, the emission component the scattered ones.

Produces two datasets, compatible with ``SimpleDataset`` / ``DataLoader``
and ``train_shape_model`` / ``train_counts_model`` in
:mod:`hyperion.models.photon_arrival_time_nflow.net`:

Shape data: one row per detected photon with its weight (absorption times
direction weight), normalised to mean 1 within its distance so that every
distance enters the weighted timing fit in proportion to its detected
photons. The row's angle is its viewing angle (Cherenkov emission) or drawn
uniformly over 0-180 deg (isotropic emission, where timing does not depend on it).

Counts data: one row per (distance x viewing-angle grid point), with
``log10_survival = log10(sum_w_detected / n_emitted)``, the raw detected
count ``n_detected`` and the Kish effective count ``n_effective``.
"""

import logging
from time import time as wall_time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.lax import cond, while_loop

from hyperion.constants import Constants
from hyperion.medium import (
    antares_ref_index_func,
    mixed_hg_rayleigh_antares,
    sca_len_func_antares,
)
from hyperion.propagate import (
    calc_new_direction,
    make_cherenkov_spectral_sampling_func,
    make_photon_sphere_intersection_func,
)
from hyperion.utils import cherenkov_ang_dist

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_C_VAC_M_NS = Constants.BaseConstants.c_vac * 1e-9  # 0.2998 m/ns
DOM_RADIUS = 0.30  # m — from orca.geo / arca.geo metadata
# Recorded with every model so the propagator and the checks know what was trained.
ANGLE_CONVENTIONS = {
    "cherenkov": "viewing-angle-cherenkov",
    "isotropic": "viewing-angle-isotropic",
}
# Refractive index for the position of the Cherenkov peak in the emission profile
_N_CHERENKOV = 1.35
_THETA_CHERENKOV = np.arccos(1.0 / _N_CHERENKOV)
# The aim cone is the DOM disc with this margin, so photons that scatter
# slightly on the way still start inside the well-sampled component.
_AIM_RADIUS_FACTOR = 2.0

# ---------------------------------------------------------------------------
# KM3NeT absorption length model
# Polynomial fit (degree 4) to KM3NeT TDR deep-sea measurements in log-space.
# ---------------------------------------------------------------------------

_ABS_WL_NM = np.array([350.0, 380.0, 400.0, 420.0, 450.0, 470.0, 500.0, 520.0, 550.0, 600.0])
_ABS_LEN_M = np.array([8.0, 14.0, 22.0, 34.0, 52.0, 62.0, 65.0, 60.0, 45.0, 20.0])
_ABS_POLY = np.polyfit(_ABS_WL_NM, np.log(_ABS_LEN_M), 4)


def _make_km3net_abs_len():
    """Build a JAX function giving KM3NeT absorption length [m] at wavelength [nm].

    Returns
    -------
    callable
        Function ``wl -> abs_len`` operating on JAX arrays.
    """
    poly = jnp.array(_ABS_POLY)

    def km3net_abs_len(wl):
        return jnp.exp(jnp.polyval(poly, wl))

    return km3net_abs_len


# ---------------------------------------------------------------------------
# Modified step function — weight tracking for absorption
# ---------------------------------------------------------------------------


def make_step_with_absorption(
    intersection_f,
    scattering_function,
    scattering_length_function,
    ref_index_func,
    absorption_length_function,
    dtype=jnp.float64,
):
    """Build a photon step function extended with absorption weight tracking.

    Identical to ``hyperion.propagate.make_step_function`` except that the
    photon state carries an additional ``weight`` field which is multiplied by
    ``exp(-actual_step / abs_len(λ))`` at every step. Photons are never
    killed; the weight encodes the cumulative survival probability.

    Parameters
    ----------
    intersection_f : callable
        DOM intersection function (from
        ``hyperion.propagate.make_photon_sphere_intersection_func``).
    scattering_function : callable
        Scattering angle sampler.
    scattering_length_function : callable
        Scattering length as a function of wavelength [nm] -> [m].
    ref_index_func : callable
        Refractive index as a function of wavelength [nm].
    absorption_length_function : callable
        Absorption length as a function of wavelength [nm] -> [m].
    dtype : jnp.dtype, optional
        Floating-point precision (default ``jnp.float64``).

    Returns
    -------
    callable
        Step function ``(photon_state, rng_key) -> (new_state, new_key)``.
        Photon state keys: ``pos``, ``dir``, ``time``, ``isec``, ``stepcnt``,
        ``wavelength``, ``weight``.
    """

    def step(photon_state, rng_key):
        pos = photon_state["pos"]
        pdir = photon_state["dir"]
        time = photon_state["time"]
        stepcnt = photon_state["stepcnt"]
        wavelength = photon_state["wavelength"]
        weight = photon_state["weight"]

        k1, k2, k3, k4 = random.split(rng_key, 4)

        sca_coeff = dtype(1.0) / scattering_length_function(wavelength)
        abs_coeff = dtype(1.0) / absorption_length_function(wavelength)
        c_med = dtype(_C_VAC_M_NS / ref_index_func(wavelength))

        step_size = -jnp.log(random.uniform(k1, dtype=dtype)) / sca_coeff

        new_pos = jnp.asarray(pos + step_size * pdir, dtype=dtype)
        new_time = time + step_size / c_med

        isec_hit, isec_pos = intersection_f(photon_x=pos, photon_p=pdir, step_size=step_size)

        # Distance actually travelled (straight-line to DOM or full step)
        dist_to_isec = jnp.linalg.norm(pos - isec_pos)
        actual_step = jnp.where(isec_hit, dist_to_isec, step_size)
        new_weight = weight * jnp.exp(-actual_step * abs_coeff)

        isec_time = time + dist_to_isec / c_med

        new_pos = cond(isec_hit, lambda a: a[0], lambda a: a[1], (isec_pos, new_pos))
        new_time = cond(isec_hit, lambda a: a[0], lambda a: a[1], (isec_time, new_time))
        new_dir = cond(
            isec_hit,
            lambda a: a[1],
            lambda a: calc_new_direction(a[0], a[1], scattering_function),
            ([k2, k3], pdir),
        )
        stepcnt = cond(isec_hit, lambda s: s, lambda s: s + jnp.int32(1), stepcnt)

        return {
            "pos": new_pos,
            "dir": new_dir,
            "time": new_time,
            "isec": isec_hit,
            "stepcnt": stepcnt,
            "wavelength": wavelength,
            "weight": new_weight,
        }, k4

    return step


# ---------------------------------------------------------------------------
# Absorption upper-bound helpers, for empty-bin labels in the counts data
# ---------------------------------------------------------------------------


def _absorption_bound_tables(wl_min, wl_max, n_grid=400):
    """Precompute the Cherenkov-spectrum-weighted absorption-length grid.

    Parameters
    ----------
    wl_min, wl_max : float
        Cherenkov wavelength range [nm].
    n_grid : int, optional
        Number of wavelength grid points (default 400).

    Returns
    -------
    abs_len_grid : np.ndarray
        Absorption length [m] at each grid wavelength.
    cher_w : np.ndarray
        Normalized Cherenkov-yield weight (proportional to 1/lambda^2) at
        each grid wavelength.
    """
    wl_grid = np.linspace(wl_min, wl_max, n_grid)
    abs_len_grid = np.exp(np.polyval(_ABS_POLY, wl_grid))
    cher_w = 1.0 / wl_grid**2
    cher_w /= cher_w.sum()
    return abs_len_grid, cher_w


def _absorption_upper_bound(d, abs_len_grid, cher_w):
    """Return the Cherenkov-spectrum-averaged absorption survival for a straight-line path.

    Parameters
    ----------
    d : float
        Straight-line path length [m].
    abs_len_grid : np.ndarray
        Absorption length [m] per wavelength grid point, from
        :func:`_absorption_bound_tables`.
    cher_w : np.ndarray
        Normalized Cherenkov-yield weight per wavelength grid point, from
        :func:`_absorption_bound_tables`.

    Returns
    -------
    float
        Upper bound on the true per-photon survival fraction. Any
        scatter-assisted path is longer than the straight-line path, so this
        is a genuine upper bound.
    """
    return float(np.sum(cher_w * np.exp(-d / abs_len_grid)))


def _absorption_upper_incremental(d_from, d_to, abs_len_grid, cher_w):
    """Return an upper bound on the incremental absorption factor between two distances.

    Used for the monotonicity constraint: if a bin at distance ``d_from`` has
    measured ``log10_survival = s``, then at ``d_to > d_from`` the survival is
    at most ``s + log10(<exp(-(d_to - d_from) / abs_len(λ))>_spectrum)``.

    Parameters
    ----------
    d_from, d_to : float
        Straight-line path lengths [m], with ``d_to > d_from``.
    abs_len_grid : np.ndarray
        Absorption length [m] per wavelength grid point, from
        :func:`_absorption_bound_tables`.
    cher_w : np.ndarray
        Normalized Cherenkov-yield weight per wavelength grid point, from
        :func:`_absorption_bound_tables`.

    Returns
    -------
    float
        Upper bound on the extra absorption factor over the incremental path.
    """
    return float(np.sum(cher_w * np.exp(-(d_to - d_from) / abs_len_grid)))


# ---------------------------------------------------------------------------
# Training-data generation
# ---------------------------------------------------------------------------


def _dom_solid_angle(d, radius=DOM_RADIUS):
    """Return the solid angle [sr] a sphere of ``radius`` subtends at distance ``d``.

    Parameters
    ----------
    d : float
        Distance from the source to the sphere centre [m].
    radius : float, optional
        Sphere radius [m].

    Returns
    -------
    float
        ``2 pi (1 - sqrt(1 - (R/d)^2))``, the exact solid angle.
    """
    ratio = min(1.0, radius / d)
    return 2.0 * np.pi * (1.0 - np.sqrt(1.0 - ratio**2))


def _cherenkov_tables(n_ph=_N_CHERENKOV, n_grid=20001):
    """Tabulate the cascade emission profile in ``cos(psi)`` around the source axis.

    Parameters
    ----------
    n_ph : float, optional
        Refractive index that places the Cherenkov peak.
    n_grid : int, optional
        Grid points over ``cos(psi)`` in [-1, 1].

    Returns
    -------
    cos_grid : np.ndarray
        ``cos(psi)`` grid.
    pdf : np.ndarray
        Probability density in ``cos(psi)``, normalised to 1 over [-1, 1].
    cdf : np.ndarray
        Its cumulative distribution, from 0 to 1.
    """
    cos_grid = np.linspace(-1.0, 1.0, n_grid)
    f = np.maximum(cherenkov_ang_dist(cos_grid, n_ph), 0.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(cos_grid))])
    pdf = f / cdf[-1]
    return cos_grid, pdf, cdf / cdf[-1]


def viewing_angle_grid(n_angle_bins, refine_deg=0.0, refine_step_deg=0.5):
    """Return the viewing angles [rad] at which the counts labels are written.

    Parameters
    ----------
    n_angle_bins : int
        Equal bins over 0-180 deg; their centres form the base grid.
    refine_deg : float, optional
        Half-width [deg] of a band around the Cherenkov angle filled with
        extra points (default 0: the base grid only). The cascade profile has
        a cusp about 0.4 deg wide at the Cherenkov angle, which a coarse grid
        cannot resolve, so the network would not conserve the total light.
    refine_step_deg : float, optional
        Spacing [deg] of the extra points (default 0.5).

    Returns
    -------
    np.ndarray
        Sorted unique viewing angles [rad].
    """
    edges = np.linspace(0.0, np.pi, n_angle_bins + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    if refine_deg > 0.0:
        half = np.radians(refine_deg)
        step = np.radians(refine_step_deg)
        extra = np.arange(_THETA_CHERENKOV - half, _THETA_CHERENKOV + half + 0.5 * step, step)
        grid = np.concatenate([grid, extra[(extra > 0.0) & (extra < np.pi)]])
    return np.unique(np.round(grid, 6))


def _source_frame(theta_v):
    """Return the source axis and two unit vectors normal to it for a viewing angle.

    The DOM sits at the origin and the source at ``(0, 0, d)``, so the
    source-to-DOM direction is ``(0, 0, -1)``. The axis is tilted by
    ``theta_v`` away from it in the x-z plane.

    Parameters
    ----------
    theta_v : float
        Viewing angle [rad].

    Returns
    -------
    np.ndarray
        Array of shape (3, 3): the axis, then the two normal unit vectors.
    """
    axis = np.array([np.sin(theta_v), 0.0, -np.cos(theta_v)])
    e1 = np.array([np.cos(theta_v), 0.0, np.sin(theta_v)])
    e2 = np.cross(axis, e1)
    return np.stack([axis, e1, e2])


def generate_training_data(
    n_photons: int,
    n_distances: int,
    n_angle_bins: int,
    d_min: float,
    d_max: float,
    wl_min: float,
    wl_max: float,
    max_time: float,
    seed: int,
    aim_fraction: float = 0.5,
    chunk_size: int = 500_000,
    emission: str = "cherenkov",
    refine_deg: float = 8.0,
):
    """Run the photon MC and build the shape and counts training datasets.

    Parameters
    ----------
    n_photons : int
        Photons emitted per source distance (isotropic emission) or per
        (distance, viewing angle) pair (Cherenkov emission).
    n_distances : int
        Number of source distances (log-uniform from ``d_min`` to ``d_max``).
    n_angle_bins : int
        Viewing-angle grid points (bin centres of equal bins over 0-180 deg)
        at which each distance's label is written.
    d_min, d_max : float
        Source distance range [m]. ``d_min`` must be greater than
        :data:`DOM_RADIUS` to avoid the source sitting exactly on the sphere
        surface, which causes d=0 intersection tests to fail the strict d>0
        check and suppresses hits.
    wl_min, wl_max : float
        Cherenkov wavelength range [nm].
    max_time : float
        Max photon propagation time [ns] (cutoff for un-intersected photons).
    seed : int
        Random seed.
    aim_fraction : float, optional
        Fraction of photons drawn inside the aim cone around the
        source-to-DOM axis (default 0.5). Any value in (0, 1) gives the
        same expectation; it only changes the variance.
    chunk_size : int, optional
        Photons propagated per JAX batch (default 500 000). Throughput
        drops for larger batches; memory grows with them.
    emission : {"cherenkov", "isotropic"}, optional
        Angular emission profile of the source (default ``"cherenkov"``).
        Cherenkov emission runs one MC per viewing angle, so many times more
        than isotropic.
    refine_deg : float, optional
        Half-width [deg] of the band of extra viewing angles (0.5 deg apart)
        around the Cherenkov angle (default 8; Cherenkov emission only). See
        :func:`viewing_angle_grid`.

    Returns
    -------
    shape_data : dict or None
        Arrays ``log10_dist``, ``angle``, ``t_residual``, ``weight`` (one row
        per detected photon), or ``None`` if no photons were detected.
    counts_data : dict or None
        Arrays ``log10_dist``, ``angle``, ``log10_survival``, ``n_detected``,
        ``n_effective`` (one row per distance x viewing-angle grid point),
        or ``None`` if nothing was generated.
    meta : dict
        Training domain and settings to record with the trained models:
        ``angle_convention``, ``emission``, ``train_d_min``, ``train_d_max``
        [m], ``n_distances``, ``n_angle_bins``, ``n_photons_per_distance``,
        ``n_photons_per_row``, ``dom_radius_m``, ``aim_fraction``,
        ``max_time_ns``, ``seed``.
    """
    if not 0.0 < aim_fraction < 1.0:
        raise ValueError("aim_fraction must lie strictly between 0 and 1")
    if emission not in ANGLE_CONVENTIONS:
        raise ValueError(f"emission must be one of {sorted(ANGLE_CONVENTIONS)}, got {emission!r}")
    convention = ANGLE_CONVENTIONS[emission]
    cherenkov = emission == "cherenkov"

    # Reference speed of light (400 nm) used for t_residual = t_hit - d/c_ref
    c_ref = float(_C_VAC_M_NS / antares_ref_index_func(400.0))

    distances = np.logspace(np.log10(d_min), np.log10(d_max), n_distances)
    angle_centers = viewing_angle_grid(n_angle_bins)
    run_angles = viewing_angle_grid(n_angle_bins, refine_deg) if cherenkov else angle_centers[:1]

    n_runs_per_distance = len(run_angles)
    meta = {
        "angle_convention": convention,
        "emission": emission,
        "train_d_min": float(distances[0]),
        "train_d_max": float(distances[-1]),
        "n_distances": int(n_distances),
        "n_angle_bins": int(n_angle_bins),
        "n_viewing_angles": int(len(run_angles) if cherenkov else len(angle_centers)),
        "refine_deg": float(refine_deg if cherenkov else 0.0),
        "n_photons_per_distance": int(n_photons) * n_runs_per_distance,
        "n_photons_per_row": int(n_photons),
        "dom_radius_m": float(DOM_RADIUS),
        "aim_fraction": float(aim_fraction),
        "max_time_ns": float(max_time),
        "seed": int(seed),
    }

    print(f"DOM radius : {DOM_RADIUS:.2f} m")
    print(f"Convention : {convention}")
    print(f"c_ref (400nm): {c_ref:.4f} m/ns")
    print(f"Distances  : {distances[0]:.2f} – {distances[-1]:.2f} m  ({n_distances} points)")
    print(f"Angle grid : {len(run_angles) if cherenkov else n_angle_bins} viewing angles")
    per = "(distance, angle)" if cherenkov else "distance"
    print(f"Photons    : {n_photons:,} per {per}  (aim fraction {aim_fraction:.2f})")
    print(f"Chunk size : {chunk_size:,}")
    print(f"Max time   : {max_time:.0f} ns")
    print()

    # --- Build physics functions ---
    km3net_abs_len = _make_km3net_abs_len()
    isec_f = make_photon_sphere_intersection_func(jnp.zeros(3), DOM_RADIUS)
    wl_sampler = make_cherenkov_spectral_sampling_func((wl_min, wl_max), antares_ref_index_func)
    step_fn = make_step_with_absorption(
        isec_f,
        mixed_hg_rayleigh_antares,
        sca_len_func_antares,
        antares_ref_index_func,
        km3net_abs_len,
    )

    max_time_j = jnp.float64(max_time)
    abs_len_grid, cher_w = _absorption_bound_tables(wl_min, wl_max)
    aim_frac_j = jnp.float64(aim_fraction)
    p_iso = jnp.float64(1.0 / (4.0 * np.pi))
    two_pi = 2.0 * np.pi
    cos_grid_np, pdf_np, cdf_np = _cherenkov_tables()
    cos_grid_j, pdf_j, cdf_j = jnp.array(cos_grid_np), jnp.array(pdf_np), jnp.array(cdf_np)

    def p_emit(direction, axis):
        """Emission density per steradian of ``direction`` for a source along ``axis``."""
        if not cherenkov:
            return p_iso
        return jnp.interp(jnp.dot(direction, axis), cos_grid_j, pdf_j) / two_pi

    # --- JIT-compile batch propagator (compiled once, reused for all distances) ---
    @jax.jit
    def run_batch(keys, source_pos, cos_alpha, frame):
        """Propagate N photons from ``source_pos``; return ``(dir_weights, final_states)``.

        Directions come from the emission-profile / aim-cone mixture; the
        photon's initial weight is the importance weight of its direction.
        ``frame`` holds the source axis and two normal unit vectors (only the
        Cherenkov profile uses them).
        """
        inv_cone = jnp.float64(1.0) / (two_pi * (jnp.float64(1.0) - cos_alpha))
        axis, e1, e2 = frame[0], frame[1], frame[2]

        def run_one(key):
            k_mix, k_theta, k_phi, k_wl, k_prop = random.split(key, 5)
            aimed = random.uniform(k_mix, dtype=jnp.float64) < aim_frac_j
            u = random.uniform(k_theta, dtype=jnp.float64)
            phi = random.uniform(k_phi, minval=0.0, maxval=two_pi, dtype=jnp.float64)
            cphi, sphi = jnp.cos(phi), jnp.sin(phi)
            # Aimed: uniform inside the cone around the source-to-DOM axis (0, 0, -1).
            cos_aim = -(jnp.float64(1.0) - u * (jnp.float64(1.0) - cos_alpha))
            sin_aim = jnp.sqrt(jnp.maximum(jnp.float64(1.0) - cos_aim**2, jnp.float64(0.0)))
            dir_aim = jnp.array([sin_aim * cphi, sin_aim * sphi, cos_aim], dtype=jnp.float64)
            # Emission profile: isotropic, or cos(psi) from the Cherenkov
            # inverse CDF around the source axis.
            if cherenkov:
                cos_em = jnp.interp(u, cdf_j, cos_grid_j)
                sin_em = jnp.sqrt(jnp.maximum(jnp.float64(1.0) - cos_em**2, jnp.float64(0.0)))
                dir_em = cos_em * axis + sin_em * (cphi * e1 + sphi * e2)
            else:
                cos_em = jnp.float64(2.0) * u - jnp.float64(1.0)
                sin_em = jnp.sqrt(jnp.maximum(jnp.float64(1.0) - cos_em**2, jnp.float64(0.0)))
                dir_em = jnp.array([sin_em * cphi, sin_em * sphi, cos_em], dtype=jnp.float64)
            init_dir = jnp.where(aimed, dir_aim, dir_em)
            in_cone = -init_dir[2] >= cos_alpha
            p_dir = p_emit(init_dir, axis)
            q_mix = (
                aim_frac_j * jnp.where(in_cone, inv_cone, jnp.float64(0.0))
                + (jnp.float64(1.0) - aim_frac_j) * p_dir
            )
            dir_weight = p_dir / q_mix

            wl = wl_sampler(k_wl)

            init_state = {
                "pos": jnp.asarray(source_pos, dtype=jnp.float64),
                "dir": init_dir,
                "time": jnp.float64(0.0),
                "isec": jnp.bool_(False),
                "stepcnt": jnp.int32(0),
                "wavelength": jnp.float64(wl),
                "weight": dir_weight,
            }

            final_state, _ = while_loop(
                lambda args: ~args[0]["isec"] & (args[0]["time"] < max_time_j),
                lambda args: step_fn(args[0], args[1]),
                (init_state, k_prop),
            )
            return dir_weight, final_state

        return jax.vmap(run_one)(keys)

    # --- Storage for training data ---
    shape_log10d: list = []
    shape_angle: list = []
    shape_tres: list = []
    shape_weight: list = []

    counts_log10d: list = []
    counts_angle: list = []
    counts_log10surv: list = []
    counts_ndet: list = []
    counts_neff: list = []

    # Most recent populated (d, log10_survival) per viewing angle: survival can
    # only fall with distance, bounded by the incremental absorption factor.
    last_populated = {}

    rng_key = random.PRNGKey(seed)
    angle_rng = np.random.default_rng(seed)

    print("Compiling JAX batch propagator (first call)...")
    t_compile = wall_time()

    first_call = True
    for i_d, d in enumerate(distances):
        if d <= DOM_RADIUS:
            print(f"  d={d:7.2f} m  SKIPPED (source on or inside DOM surface)")
            continue

        source_pos = jnp.array([0.0, 0.0, float(d)], dtype=jnp.float64)
        sin_alpha = min(1.0, _AIM_RADIUS_FACTOR * DOM_RADIUS / float(d))
        cos_alpha = jnp.float64(np.sqrt(1.0 - sin_alpha**2))
        log10d = np.float32(np.log10(d))
        t0 = wall_time()
        n_hits_d = 0

        for i_a, theta_run in enumerate(run_angles):
            frame = jnp.array(_source_frame(theta_run), dtype=jnp.float64)
            isec_parts, time_parts, weight_parts = [], [], []
            n_left = int(n_photons)
            while n_left > 0:
                n_chunk = min(int(chunk_size), n_left)
                rng_key, subkey = random.split(rng_key)
                keys = random.split(subkey, n_chunk)
                _, final_states = run_batch(keys, source_pos, cos_alpha, frame)
                jax.block_until_ready(final_states["time"])
                if first_call:
                    logger.info("First-call (compile + run): %.1f s", wall_time() - t_compile)
                    first_call = False
                isec_parts.append(np.asarray(final_states["isec"]))
                time_parts.append(np.asarray(final_states["time"]))
                weight_parts.append(np.asarray(final_states["weight"]))
                n_left -= n_chunk

            isec_np = np.concatenate(isec_parts)  # (N,) bool
            time_np = np.concatenate(time_parts)  # (N,) float64
            weight_np = np.concatenate(weight_parts)  # (N,) direction x absorption weight

            n_detected = int(isec_np.sum())
            n_hits_d += n_detected
            w_det = weight_np[isec_np]
            w_sum = float(w_det.sum())

            if n_detected == 0 or w_sum <= 0.0:
                # No more than the DOM's solid angle times the largest emission
                # density toward it, attenuated over the straight path, can
                # arrive; and no more than the last populated distance allowed.
                alpha_dom = np.arcsin(min(1.0, DOM_RADIUS / float(d)))
                if cherenkov:
                    psi = np.linspace(max(theta_run - alpha_dom, 0.0), theta_run + alpha_dom, 64)
                    p_max = np.interp(np.cos(psi), cos_grid_np, pdf_np).max() / two_pi
                else:
                    p_max = 1.0 / (4.0 * np.pi)
                bounds = [
                    np.log10(
                        _dom_solid_angle(float(d))
                        * p_max
                        * _absorption_upper_bound(d, abs_len_grid, cher_w)
                    )
                ]
                if i_a in last_populated:
                    d_prev, s_prev = last_populated[i_a]
                    bounds.append(
                        s_prev
                        + np.log10(_absorption_upper_incremental(d_prev, d, abs_len_grid, cher_w))
                    )
                label = np.float32(min(bounds))
                n_det_row, n_eff_row = np.float32(0), np.float32(0)
            else:
                survival = w_sum / n_photons
                label = np.float32(np.log10(survival))
                n_eff = w_sum**2 / float(np.sum(w_det**2))
                n_det_row, n_eff_row = np.float32(n_detected), np.float32(n_eff)
                if n_eff >= 2.0:
                    last_populated[i_a] = (d, float(label))

                # --- Shape data: detected photons, weights relative within the run ---
                # Weights enter the weighted NLL loss during training:
                #   L = -sum(w_i * log p(t_i)) / sum(w_i)
                # The aimed photons carry direction weights many orders below the
                # emission-profile ones, so weights are normalised to mean 1 per run.
                t_res_det = time_np[isec_np] - d / c_ref
                shape_log10d.append(np.full(n_detected, log10d, dtype=np.float32))
                if cherenkov:
                    shape_angle.append(np.full(n_detected, theta_run, dtype=np.float32))
                else:
                    # Isotropic timing does not depend on the angle.
                    shape_angle.append(angle_rng.uniform(0.0, np.pi, n_detected).astype(np.float32))
                shape_tres.append(t_res_det.astype(np.float32))
                shape_weight.append((w_det / w_det.mean()).astype(np.float32))

            # --- Counts data: this run's row, or the same label at every grid
            # point for isotropic emission ---
            for theta_v in [theta_run] if cherenkov else angle_centers:
                counts_log10d.append(log10d)
                counts_angle.append(np.float32(theta_v))
                counts_log10surv.append(label)
                counts_ndet.append(n_det_row)
                counts_neff.append(n_eff_row)

        print(
            f"  d={d:7.2f} m  hits={n_hits_d:9d} in {len(run_angles)} run(s) of {n_photons:,}  "
            f"[{wall_time() - t0:.1f} s]"
        )

    # --- Build shape data ---
    if shape_log10d:
        shape_data = {
            "log10_dist": np.concatenate(shape_log10d),
            "angle": np.concatenate(shape_angle),
            "t_residual": np.concatenate(shape_tres),
            "weight": np.concatenate(shape_weight),
        }
    else:
        shape_data = None

    # --- Build counts data ---
    if counts_log10d:
        counts_data = {
            "log10_dist": np.array(counts_log10d, dtype=np.float32),
            "angle": np.array(counts_angle, dtype=np.float32),
            "log10_survival": np.array(counts_log10surv, dtype=np.float32),
            "n_detected": np.array(counts_ndet, dtype=np.float32),
            "n_effective": np.array(counts_neff, dtype=np.float32),
        }
        n_bound = int((counts_data["n_detected"] == 0).sum())
        print(f"\nCounts rows : {len(counts_log10d)} ({n_bound} labelled with an upper bound)")
    else:
        counts_data = None

    return shape_data, counts_data, meta


# Half-widths of the label pooling, fixed by distance only: upper distance [m],
# distance rows, degrees inside the Cherenkov zone, degrees outside it.
_POOL_SCHEDULE = (
    (8.0, 0, 0.0, 0.0),
    (15.0, 1, 0.5, 0.0),
    (30.0, 2, 1.0, 10.0),
    (60.0, 3, 2.0, 20.0),
    (120.0, 4, 3.0, 30.0),
    (np.inf, 6, 4.0, 45.0),
)


def pool_counts_rows(counts: dict, refine_deg: float = 8.0, cone_deg: float = 42.2) -> dict:
    """Average the counts labels over windows fixed by distance.

    Far from the source a row's label rests on a few scattered photons of large
    weight. A row that catches one has a high label and a low effective count,
    and a row that misses them has a low label and a high effective count. A
    fit of the logarithm weighted by that count therefore follows the second
    kind and loses light. The labels are averaged here in linear space, with
    solid-angle weights, which keeps their expectation.

    The window depends only on the distance (``_POOL_SCHEDULE``). A window that
    grew until its effective count looked sufficient would stop early on the
    rows that missed the heavy photons and repeat the same selection. Angles
    are pooled only within their zone (below, inside or above the Cherenkov
    cone), so that the cone stays as sharp as the schedule allows. Rows are
    carried across a distance window with a smooth fit of the light against
    distance.

    Parameters
    ----------
    counts : dict
        ``log10_dist``, ``angle``, ``log10_survival``, ``n_detected`` and
        ``n_effective``, on a rectangular grid of distance and angle.
    refine_deg : float, optional
        Half-width of the Cherenkov zone [deg].
    cone_deg : float, optional
        Centre of the Cherenkov zone [deg].

    Returns
    -------
    dict
        The same arrays with pooled labels. ``n_detected`` is the number of raw
        hits in the window and ``n_effective`` the effective count of the
        window.

    Raises
    ------
    ValueError
        If the rows do not form a rectangular grid.
    """
    log10_dist, dist_index = np.unique(counts["log10_dist"], return_inverse=True)
    angle, angle_index = np.unique(counts["angle"], return_inverse=True)
    n_dist, n_angle = log10_dist.size, angle.size
    if counts["log10_dist"].size != n_dist * n_angle:
        raise ValueError("pool_counts_rows needs a rectangular distance x angle grid")

    def on_grid(key):
        out = np.zeros((n_dist, n_angle))
        out[dist_index, angle_index] = counts[key]
        return out

    label = 10.0 ** on_grid("log10_survival")
    hits = on_grid("n_detected")
    sum_w2 = label**2 / np.maximum(on_grid("n_effective"), 1e-9)

    dist = 10.0**log10_dist
    weight = np.gradient(angle) * np.sin(angle)
    weight = weight / weight.sum()
    light = (label * weight).sum(axis=1)
    fit = dist >= 5.0
    coeff = np.polyfit(dist[fit], np.log(dist[fit] ** 2 * light[fit]), 2)
    scale = np.exp(np.polyval(coeff, dist)) / dist**2

    degrees = np.degrees(angle)
    zone = np.digitize(degrees, [cone_deg - refine_deg, cone_deg + refine_deg])
    pooled, pooled_hits, pooled_w2 = (
        np.zeros_like(label),
        np.zeros_like(label),
        np.zeros_like(label),
    )
    for i, d in enumerate(dist):
        n_rows, half_cone, half_off = next(s[1:] for s in _POOL_SCHEDULE if d < s[0])
        rows = np.arange(max(i - n_rows, 0), min(i + n_rows + 1, n_dist))
        carry = (scale[i] / scale[rows])[:, None]
        for j in range(n_angle):
            half = half_cone if zone[j] == 1 else half_off
            cols = np.flatnonzero((zone == zone[j]) & (np.abs(degrees - degrees[j]) <= half + 1e-6))
            w = weight[cols] / weight[cols].sum()
            pooled[i, j] = (label[np.ix_(rows, cols)] * carry * w).sum() / rows.size
            pooled_hits[i, j] = hits[np.ix_(rows, cols)].sum()
            pooled_w2[i, j] = (sum_w2[np.ix_(rows, cols)] * carry**2 * w**2).sum() / rows.size**2

    return {
        **counts,
        "log10_survival": np.log10(pooled[dist_index, angle_index]).astype(np.float32),
        "n_detected": pooled_hits[dist_index, angle_index].astype(np.float32),
        "n_effective": (pooled**2 / np.maximum(pooled_w2, 1e-300))[dist_index, angle_index].astype(
            np.float32
        ),
    }
