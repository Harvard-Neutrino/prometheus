import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

from ..photon_source import PhotonSource, PhotonSourceType


def source_to_model_input_per_module(
    module_coords, source_pos, source_dir, source_t0, c_medium
):
    """
    Convert photon source and module coordinates into neural net input.

    Calculates the distance and viewing angle between the source and the module.
    The viewing angle is the angle of the vector between module and source and the direction
    vector of the source.
    Also calculates the geometric time (expected arrival time for a direct photon).

    Returns the viewing angle and log10(distance) the geometric time.

    """

    source_targ_vec = module_coords - source_pos

    dist = jnp.linalg.norm(source_targ_vec)
    # angles = jnp.arccos(jnp.einsum("ak, k -> a", source_targ_vec, source_dir) / dist)

    angle = jnp.arccos(jnp.sum(source_targ_vec * source_dir) / dist)

    time_geo = dist / c_medium + source_t0

    inp_pars = jnp.asarray([jnp.log10(dist), angle])

    return inp_pars, time_geo


# Vectorize across modules
source_to_model_input = vmap(
    source_to_model_input_per_module, in_axes=(0, None, None, None, None)
)

# Vectorize across sources and jit
sources_to_model_input = jit(vmap(source_to_model_input, in_axes=(None, 0, 0, 0, None)))

# Vectorize across sources and jit
sources_to_model_input_per_module = vmap(
    source_to_model_input_per_module, in_axes=(None, 0, 0, 0, None)
)


def sources_to_array(sources):
    source_pos = np.empty((len(sources), 3))
    source_dir = np.empty((len(sources), 3))
    source_time = np.empty((len(sources), 1))
    source_photons = np.empty((len(sources), 1))

    for i, source in enumerate(sources):
        if source.type != PhotonSourceType.STANDARD_CHERENKOV:
            raise ValueError(
                f"Only Cherenkov-like sources are supported. Got {source.type}."
            )
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
