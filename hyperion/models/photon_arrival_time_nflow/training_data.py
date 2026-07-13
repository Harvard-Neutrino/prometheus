"""Photon Monte Carlo training-data generation for the shape/counts models.

Runs photon MC in ANTARES/KM3NeT Mediterranean water with absorption. DOM
placed at origin (radius 0.30 m); isotropic Cherenkov photon sources at
logarithmically spaced distances from d_min to d_max.

Each photon accumulates an absorption survival weight
``w = prod_steps exp(-step_size / abs_len(λ))``. This weight-tracking
approach avoids killing photons early (improving statistics at large
distances) and is equivalent to the absorption-corrected expected photon
count.

Produces two datasets, compatible with ``SimpleDataset`` / ``DataLoader``
and ``train_shape_model`` / ``train_counts_model`` in
:mod:`hyperion.models.photon_arrival_time_nflow.net`:

Shape data: one row per detected photon after importance-sampling thinning
(i.e., each detected photon is kept with probability proportional to its
absorption weight, producing unweighted samples from the correct
distribution).

Counts data: one row per (distance x angle-bin) combination, with
``log10_survival = log10(sum_weights_detected / n_emitted_in_bin)``.
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

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_C_VAC_M_NS = Constants.BaseConstants.c_vac * 1e-9  # 0.2998 m/ns
DOM_RADIUS = 0.30  # m — from orca.geo / arca.geo metadata

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
):
    """Run the photon MC and build the shape and counts training datasets.

    Parameters
    ----------
    n_photons : int
        Photons emitted per source distance.
    n_distances : int
        Number of source distances (log-uniform from ``d_min`` to ``d_max``).
    n_angle_bins : int
        Emission-angle bins for the counts data.
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

    Returns
    -------
    shape_data : dict or None
        Arrays ``log10_dist``, ``angle``, ``t_residual``, ``weight`` (one row
        per detected photon), or ``None`` if no photons were detected.
    counts_data : dict or None
        Arrays ``log10_dist``, ``angle``, ``log10_survival``, ``n_detected``
        (one row per populated distance x angle-bin combination), or ``None``
        if no bins were populated.
    """
    # Reference speed of light (400 nm) used for t_residual = t_hit - d/c_ref
    c_ref = float(_C_VAC_M_NS / antares_ref_index_func(400.0))

    distances = np.logspace(np.log10(d_min), np.log10(d_max), n_distances)
    angle_edges = np.linspace(0.0, np.pi, n_angle_bins + 1)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])

    print(f"DOM radius : {DOM_RADIUS:.2f} m")
    print(f"c_ref (400nm): {c_ref:.4f} m/ns")
    print(f"Distances  : {distances[0]:.2f} – {distances[-1]:.2f} m  ({n_distances} points)")
    print(f"Photons/d  : {n_photons:,}")
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

    # --- JIT-compile batch propagator (compiled once, reused for all distances) ---
    @jax.jit
    def run_batch(keys, source_pos):
        """Propagate N photons from ``source_pos``; return ``(init_dirs, final_states)``."""

        def run_one(key):
            k_dir, k_wl, k_prop = random.split(key, 3)
            k_theta, k_phi = random.split(k_dir, 2)
            cos_theta = random.uniform(k_theta, minval=-1.0, maxval=1.0, dtype=jnp.float64)
            phi = random.uniform(k_phi, minval=0.0, maxval=2.0 * np.pi, dtype=jnp.float64)
            sin_theta = jnp.sqrt(jnp.maximum(jnp.float64(1.0) - cos_theta**2, jnp.float64(0.0)))
            init_dir = jnp.array(
                [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta],
                dtype=jnp.float64,
            )
            wl = wl_sampler(k_wl)

            init_state = {
                "pos": jnp.asarray(source_pos, dtype=jnp.float64),
                "dir": init_dir,
                "time": jnp.float64(0.0),
                "isec": jnp.bool_(False),
                "stepcnt": jnp.int32(0),
                "wavelength": jnp.float64(wl),
                "weight": jnp.float64(1.0),
            }

            final_state, _ = while_loop(
                lambda args: ~args[0]["isec"] & (args[0]["time"] < max_time_j),
                lambda args: step_fn(args[0], args[1]),
                (init_state, k_prop),
            )
            return init_dir, final_state

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

    # Per-angle-bin record of the most recent *populated* (d, log10_survival) entry.
    # Used to propagate the monotonicity upper bound: survival can only decrease
    # with distance, bounded by the incremental absorption factor.
    _last_surv_by_bin: dict = {}  # j -> (d_last, log10surv_last)

    rng_key = random.PRNGKey(seed)

    print("Compiling JAX batch propagator (first call)...")
    t_compile = wall_time()

    for i_d, d in enumerate(distances):
        if d <= DOM_RADIUS:
            print(f"  d={d:7.2f} m  SKIPPED (source on or inside DOM surface)")
            continue

        source_pos = jnp.array([0.0, 0.0, float(d)], dtype=jnp.float64)
        rng_key, subkey = random.split(rng_key)
        keys = random.split(subkey, n_photons)

        t0 = wall_time()
        init_dirs, final_states = run_batch(keys, source_pos)
        jax.block_until_ready(final_states["time"])

        if i_d == 0:
            logger.info("First-call (compile + run): %.1f s", wall_time() - t_compile)

        init_dirs_np = np.asarray(init_dirs)  # (N, 3)
        isec_np = np.asarray(final_states["isec"])  # (N,) bool
        time_np = np.asarray(final_states["time"])  # (N,) float64
        weight_np = np.asarray(final_states["weight"])  # (N,) float64

        n_detected = int(isec_np.sum())
        elapsed = wall_time() - t0

        # Emission angle: angle between init_dir and source→DOM axis (0,0,-1)
        # cos(em_angle) = dot(init_dir, (0,0,-1)) = -init_dir_z
        em_angle_all = np.arccos(np.clip(-init_dirs_np[:, 2], -1.0, 1.0))  # (N,)

        print(
            f"  d={d:7.2f} m  hits={n_detected:7d}/{n_photons}  "
            f"w_sum={weight_np[isec_np].sum():.3e}  [{elapsed:.1f} s]"
        )

        if n_detected == 0:
            continue

        em_angle_det = em_angle_all[isec_np]  # angles of detected photons
        weight_det = weight_np[isec_np]  # absorption weights of detected photons
        t_res_det = time_np[isec_np] - d / c_ref  # time residuals [ns]

        # --- Shape data: keep all detected photons with absorption weights ---
        # Weights are used directly in the weighted NLL loss during training:
        #   L = -sum(w_i * log p(t_i)) / sum(w_i)
        # This avoids the acceptance-rejection thinning that discards >90% of
        # photons at large distances, where weights span many orders of magnitude.
        shape_log10d.append(np.full(n_detected, np.log10(d), dtype=np.float32))
        shape_angle.append(em_angle_det.astype(np.float32))
        shape_tres.append(t_res_det.astype(np.float32))
        shape_weight.append(weight_det.astype(np.float32))

        # --- Counts data: bin detected photons by emission angle ---
        for j in range(n_angle_bins):
            a_lo = angle_edges[j]
            a_hi = angle_edges[j + 1]

            in_bin_all = (em_angle_all >= a_lo) & (em_angle_all < a_hi)
            in_bin_det = in_bin_all & isec_np

            n_emit_bin = int(in_bin_all.sum())
            if n_emit_bin == 0:
                continue

            w_bin = float(weight_np[in_bin_det].sum())
            if w_bin <= 0.0:
                # No photons reached the DOM from this bin.  Compute the tightest
                # of three independent upper bounds on the true survival fraction:
                #
                # (1) Absorption physics: every photon travels the straight-line
                #     path of length d; scatter-assisted paths are longer so this
                #     is a genuine upper bound.
                bound_abs = np.log10(_absorption_upper_bound(d, abs_len_grid, cher_w))
                #
                # (2) MC statistics: threw n_emit_bin photons and observed 0 hits;
                #     Poisson 95 % upper limit on the rate is 3 / n_emit_bin.
                bound_mc = np.log10(3.0 / n_emit_bin)
                #
                # (3) Monotonicity: survival decreases with distance.  If the
                #     closest populated bin at the same angle had log10_survival
                #     s_prev at d_prev, the upper bound at d is
                #     s_prev + log10(<exp(-(d-d_prev)/abs_len(λ))>_spectrum).
                bounds = [bound_abs, bound_mc]
                if j in _last_surv_by_bin:
                    d_prev, s_prev = _last_surv_by_bin[j]
                    mono_bound = s_prev + np.log10(
                        _absorption_upper_incremental(d_prev, d, abs_len_grid, cher_w)
                    )
                    bounds.append(mono_bound)
                #
                log10_surv_upper = min(bounds)
                counts_log10d.append(np.float32(np.log10(d)))
                counts_angle.append(np.float32(angle_centers[j]))
                counts_log10surv.append(np.float32(log10_surv_upper))
                counts_ndet.append(np.float32(0))  # training weight = sqrt(0+1) = 1
                continue

            n_det_bin = int(in_bin_det.sum())
            survival = w_bin / n_emit_bin
            log10s = float(np.log10(survival))
            # Only seed the monotonicity tracker from reliable multi-photon bins.
            # Single-photon entries can have extreme absorption weights from rare
            # long scatter paths, producing spuriously large negative values that
            # would propagate forward and over-constrain all subsequent distances.
            if n_det_bin >= 2:
                _last_surv_by_bin[j] = (d, log10s)
            counts_log10d.append(np.float32(np.log10(d)))
            counts_angle.append(np.float32(angle_centers[j]))
            counts_log10surv.append(np.float32(log10s))
            counts_ndet.append(np.float32(n_det_bin))

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

    # --- Angular monotonicity pass for counts data ---
    # Survival is physically non-increasing with angle (forward emission always
    # dominates backward).  Empty bins near 180° are assigned a loose MC Poisson
    # upper bound log10(3/n_emit) that can be orders of magnitude above the true
    # value.  By sweeping from small to large angles within each distance slice
    # we propagate the tightest available bound: any empty bin at angle θ has
    # survival ≤ survival(θ_prev) where θ_prev < θ is the most recent bin.
    # This only tightens labels that were already upper bounds (n_det = 0); real
    # measurements (n_det > 0) are never modified.
    if counts_log10d:
        _c_log10d = np.array(counts_log10d)
        _c_angle = np.array(counts_angle)
        _c_log10s = np.array(counts_log10surv)
        _c_ndet = np.array(counts_ndet)

        unique_log10d = np.unique(_c_log10d)
        n_tightened = 0
        for ld in unique_log10d:
            mask = _c_log10d == ld
            idx = np.where(mask)[0]
            # Sort by angle within this distance slice
            order = np.argsort(_c_angle[idx])
            idx_sorted = idx[order]

            running_bound = 0.0  # start at 0 (log10(1) = no constraint yet)
            for k in idx_sorted:
                if _c_ndet[k] >= 2:
                    # Reliable multi-photon measurement: update running bound
                    # but do not modify the label.  Single-photon entries are
                    # excluded here because an extreme absorption weight on one
                    # photon produces spuriously large negative log10_survival
                    # values that would over-constrain all larger-angle bins.
                    running_bound = _c_log10s[k]
                elif _c_ndet[k] == 0:
                    # Empty bin: tighten if the angular neighbour bound is stricter
                    if _c_log10s[k] > running_bound:
                        n_tightened += 1
                        _c_log10s[k] = running_bound
                    # Propagate only if this tightened value is itself tighter
                    running_bound = min(running_bound, _c_log10s[k])
                # n_det == 1: single photon — do not update running_bound,
                # but do tighten the label if the running bound is stricter.
                elif _c_log10s[k] > running_bound:
                    n_tightened += 1
                    _c_log10s[k] = running_bound

        if n_tightened:
            print(
                f"  Angular monotonicity pass tightened {n_tightened} empty-bin "
                f"labels (of {int((_c_ndet == 0).sum())} total empty bins)."
            )

        counts_data = {
            "log10_dist": _c_log10d,
            "angle": _c_angle,
            "log10_survival": _c_log10s,
            "n_detected": _c_ndet,
        }
    else:
        counts_data = None

    return shape_data, counts_data
