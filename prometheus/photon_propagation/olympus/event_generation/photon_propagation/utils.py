import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from ..photon_source import PhotonSource, PhotonSourceType


def next_bucket(n: int, base: int = 2, minimum: int = 1) -> int:
    """Return the smallest power of ``base`` that is >= both ``n`` and ``minimum``.

    Array shapes padded to bucket sizes let jitted functions compile once per
    bucket instead of once per unique shape, which otherwise dominates the
    runtime of the photon propagation loop.

    Parameters
    ----------
    n : int
        Number of elements that must fit in the bucket.
    base : int
        Bucket growth factor.
    minimum : int
        Smallest bucket size to return.

    Returns
    -------
    int
        Bucket size.
    """
    bucket = minimum
    while bucket < n:
        bucket *= base
    return bucket


def source_to_model_input_per_module(module_coords, source_pos, source_dir, source_t0, c_medium):
    """Convert photon source and module coordinates into neural net input.

    Calculates the distance and viewing angle between the source and the module.
    The viewing angle is the angle of the vector between module and source and the direction
    vector of the source.

    Parameters
    ----------
    module_coords : jnp.ndarray
        Coordinates of the module.
    source_pos : jnp.ndarray
        Position of the photon source.
    source_dir : jnp.ndarray
        Direction vector of the photon source.
    source_t0 : float
        Emission time of the photon source.
    c_medium : float
        Speed of light in the medium.

    Returns
    -------
    inp_pars : jnp.ndarray
        Array of ``[log10(distance), viewing_angle]``.
    time_geo : float
        Geometric time (expected arrival time for a direct photon).

    """

    source_targ_vec = module_coords - source_pos

    dist = jnp.linalg.norm(source_targ_vec)
    # angles = jnp.arccos(jnp.einsum("ak, k -> a", source_targ_vec, source_dir) / dist)

    angle = jnp.arccos(jnp.sum(source_targ_vec * source_dir) / dist)

    time_geo = dist / c_medium + source_t0

    inp_pars = jnp.asarray([jnp.log10(dist), angle])

    return inp_pars, time_geo


# Vectorize across modules
source_to_model_input = vmap(source_to_model_input_per_module, in_axes=(0, None, None, None, None))

# Vectorize across sources and jit
sources_to_model_input = jit(vmap(source_to_model_input, in_axes=(None, 0, 0, 0, None)))

# Vectorize across sources and jit
sources_to_model_input_per_module = vmap(
    source_to_model_input_per_module, in_axes=(None, 0, 0, 0, None)
)


def sources_to_model_input_chunked(
    module_coords, source_pos, source_dir, source_time, c_medium, module_chunk
):
    """Evaluate the model-input kernel in fixed-size module chunks.

    ``sources_to_model_input`` is jitted and vmapped over both the module and
    the source axis, so XLA compiles one executable per distinct
    ``(n_sources, n_modules)`` shape pair. Padding both axes to power-of-two
    buckets makes the number of cached executables the *product* of the two
    ladders -- roughly nine module buckets times eight source buckets, so up
    to ~70 resident variants of this one kernel. That cache is never
    reclaimed, and it is the bulk of the several GB each process accumulates.

    Holding the module axis at a fixed ``module_chunk`` and looping takes it
    out of the cross-product, leaving only the source bucket to vary. Values
    are unchanged: each ``(module, source)`` pair is evaluated independently
    of every other, so splitting the module axis is exact. The fixed chunk
    also wastes less padding than rounding a few thousand modules up to the
    next power of two.

    Parameters
    ----------
    module_coords : np.ndarray
        Module coordinates, shape ``(n_modules, 3)``.
    source_pos : np.ndarray
        Source positions, shape ``(n_sources, 3)``.
    source_dir : np.ndarray
        Source direction vectors, shape ``(n_sources, 3)``.
    source_time : np.ndarray
        Source emission times, shape ``(n_sources, 1)``.
    c_medium : float
        Speed of light in the medium.
    module_chunk : int
        Number of modules per jitted call.

    Returns
    -------
    inp_pars : np.ndarray
        ``[log10(distance), viewing_angle]`` per pair, shape
        ``(n_sources, n_modules, 2)``.
    time_geo : np.ndarray
        Geometric arrival time per pair, shape ``(n_sources, n_modules, 1)``.
    """
    module_coords = np.asarray(module_coords)
    n_mod = module_coords.shape[0]
    n_src = np.shape(source_pos)[0]
    chunk = max(1, int(module_chunk))

    if n_mod == 0:
        return np.empty((n_src, 0, 2)), np.empty((n_src, 0, 1))

    inp_chunks = []
    time_chunks = []
    for start in range(0, n_mod, chunk):
        block = module_coords[start : start + chunk]
        pad = chunk - block.shape[0]
        if pad:
            # Only the final chunk is ever padded, and its extra columns are
            # trimmed below. 1e6 merely keeps the distance finite and nonzero
            # so that log10 and arccos stay well defined.
            block = np.pad(block, ((0, pad), (0, 0)), constant_values=1e6)
        inp_pars, time_geo = sources_to_model_input(
            block, source_pos, source_dir, source_time, c_medium
        )
        inp_chunks.append(np.asarray(inp_pars))
        time_chunks.append(np.asarray(time_geo))

    inp_pars = np.concatenate(inp_chunks, axis=1)[:, :n_mod]
    time_geo = np.concatenate(time_chunks, axis=1)[:, :n_mod]
    return inp_pars, time_geo


def sources_to_array(sources):
    source_pos = np.empty((len(sources), 3))
    source_dir = np.empty((len(sources), 3))
    source_time = np.empty((len(sources), 1))
    source_photons = np.empty((len(sources), 1))

    for i, source in enumerate(sources):
        if source.type != PhotonSourceType.STANDARD_CHERENKOV:
            raise ValueError(f"Only Cherenkov-like sources are supported. Got {source.type}.")
        source_pos[i] = source.position
        source_dir[i] = source.direction
        source_time[i] = source.time
        source_photons[i] = source.n_photons
    return source_pos, source_dir, source_time, source_photons


def source_array_to_sources(source_pos, source_dir, source_time, source_nphotons):
    sources = []
    for i in range(source_pos.shape[0]):
        source = PhotonSource(
            np.asarray(source_pos[i]),
            np.asarray(source_nphotons[i]),
            np.asarray(source_time[i]),
            np.asarray(source_dir[i]),
        )
        sources.append(source)
    return sources
