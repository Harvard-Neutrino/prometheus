"""Event generation helpers."""

import functools
import logging

import awkward as ak
import numpy as np
from jax import random
from tqdm.auto import trange

from .constants import Constants
from .detector import (
    generate_noise,
    sample_cylinder_surface,
    sample_cylinder_volume,
    sample_direction,
)
from .lightyield import make_pointlike_cascade_source, make_realistic_cascade_source
from .mc_record import MCRecord
from .photon_propagation.utils import next_bucket, source_array_to_sources
from .utils import track_isects_cyl

logger = logging.getLogger(__name__)


def _propagate_in_range_modules(
    det,
    module_mask,
    source_pos,
    source_dir,
    source_time,
    source_nphotons,
    pprop_func,
    key,
    splitter,
):
    """Propagate photons to the masked modules and scatter back to full length.

    Only modules selected by ``module_mask`` are passed to ``pprop_func``; the
    ragged result is re-expanded to one entry per detector module, with empty
    lists for modules that were out of range.

    Parameters
    ----------
    det : Detector
        Instance of Detector class.
    module_mask : np.ndarray
        Boolean mask of shape (n_modules,) selecting modules within reach of
        at least one source.
    source_pos, source_dir, source_time, source_nphotons : np.ndarray
        Source description arrays passed through to ``pprop_func``.
    pprop_func : callable
        Function that computes the photon propagation signal.
    key : jax.random.PRNGKey
        Random key for photon sampling.
    splitter : int
        Number of modules per subset for memory-efficient propagation.

    Returns
    -------
    ak.Array
        Ragged array of photon arrival times with one entry per detector module.
    """
    n_modules = det.module_coords.shape[0]

    if source_pos.shape[0] == 0 or not np.any(module_mask):
        return ak.Array([[] for _ in range(n_modules)])

    sub_coords = det.module_coords[module_mask]
    sub_eff = det.module_efficiencies[module_mask]
    n_sub = sub_coords.shape[0]

    # Pad modules and sources to power-of-two buckets so the jitted
    # model-input kernel compiles once per bucket pair instead of once per
    # unique (n_sources, n_modules) shape.  Padded entries sit far outside
    # max_distance and carry zero photons / zero efficiency, so the distance
    # mask in the photon propagator drops every padded pair.
    mod_pad = next_bucket(n_sub, minimum=16) - n_sub
    src_pad = next_bucket(source_pos.shape[0], minimum=16) - source_pos.shape[0]
    sub_coords = np.pad(sub_coords, ((0, mod_pad), (0, 0)), constant_values=1e6)
    sub_eff = np.pad(np.asarray(sub_eff), (0, mod_pad))
    source_pos = np.pad(np.asarray(source_pos), ((0, src_pad), (0, 0)), constant_values=-1e6)
    source_dir = np.pad(np.asarray(source_dir), ((0, src_pad), (0, 0)))
    source_time = np.pad(
        np.asarray(source_time), [(0, src_pad)] + [(0, 0)] * (np.ndim(source_time) - 1)
    )
    source_nphotons = np.pad(
        np.asarray(source_nphotons), [(0, src_pad)] + [(0, 0)] * (np.ndim(source_nphotons) - 1)
    )

    if sub_coords.shape[0] > splitter:
        det_subsets_coords = np.array_split(sub_coords, sub_coords.shape[0] % splitter)
        det_subsets_eff = np.array_split(sub_eff, sub_coords.shape[0] % splitter)
        sub_result = [
            pprop_func(
                det_subsets_coords[id_set],
                det_subsets_eff[id_set],
                source_pos,
                source_dir,
                source_time,
                source_nphotons,
                seed=key,
            )
            for id_set, _ in enumerate(det_subsets_coords)
        ]
        sub_result = ak.concatenate(sub_result)
    else:
        sub_result = pprop_func(
            sub_coords,
            sub_eff,
            source_pos,
            source_dir,
            source_time,
            source_nphotons,
            seed=key,
        )

    # Scatter the masked result back into a full-length ragged array so the
    # positional module -> (string, om) mapping downstream stays intact.
    # Padded modules cannot have received photons, so dropping their (empty)
    # trailing entries keeps the flattened times aligned with the counts.
    counts = np.zeros(n_modules, dtype=np.int64)
    counts[module_mask] = np.asarray(ak.num(sub_result))[:n_sub]
    return ak.unflatten(ak.flatten(sub_result), counts)


def simulate_noise(det, event):

    if ak.count(event) == 0:
        time_range = [-1000, 4000]
        noise = generate_noise(det, time_range)
        event = ak.sort(noise, axis=1)

    else:
        time_range = [
            ak.min(ak.flatten(event)) - 1000,
            ak.max(ak.flatten(event)) + 4000,
        ]
        noise = generate_noise(det, time_range)
        event = ak.sort(ak.concatenate([event, noise], axis=1))

    return event, noise


def generate_cascade(
    det,
    event_data,
    seed,
    pprop_func,
    converter_func,
    splitter=100000,
    max_distance=300.0,
):
    """Generate a single cascade and return detected photon times.

    Parameters
    ----------
    det : Detector
        Instance of Detector class.
    event_data : dict
        Container with event parameters (position, energy, direction, time, etc.).
    seed : int
        Random seed.
    pprop_func : callable
        Function that computes the photon propagation signal.
    converter_func : callable
        Callable that converts event energy and metadata to source positions,
        directions, times and photon counts.
    splitter : int, optional
        Number of modules per subset for memory-efficient propagation.
    max_distance : float, optional
        Maximum source-to-module distance in metres.  Source positions farther
        than this from every module are dropped before photon propagation.

    Returns
    -------
    tuple
        Tuple ``(propagation_result, record)`` where ``propagation_result`` is an
        ``awkward.Array`` of detected photon times and ``record`` is an
        ``MCRecord`` instance.
    """

    k1, k2 = random.split(seed)

    source_pos, source_dir, source_time, source_nphotons = converter_func(
        event_data["pos"],
        event_data["time"],
        event_data["dir"],
        event_data["energy"],
        event_data["particle_id"],
        key=k1,
    )

    # Drop sources whose photon budget is negligible — they cannot produce
    # detected hits (detection efficiency << 1) and dominate the flat
    # (n_sources × n_modules) matrix built in the photon propagator.
    photon_mask = np.asarray(source_nphotons).squeeze() >= 1.0
    source_pos = source_pos[photon_mask]
    source_dir = source_dir[photon_mask]
    source_time = source_time[photon_mask]
    source_nphotons = source_nphotons[photon_mask]

    # Drop sources farther than max_distance from every module, and modules
    # farther than max_distance from every surviving source — neither can
    # contribute hits, and both inflate the dense (n_sources x n_modules)
    # matrix built in the photon propagator.
    module_mask = np.zeros(det.module_coords.shape[0], dtype=bool)
    if source_pos.shape[0] > 0:
        dist_matrix = np.linalg.norm(
            np.asarray(source_pos)[:, np.newaxis, :] - det.module_coords[np.newaxis, :, :],
            axis=-1,
        )
        distance_mask = np.any(dist_matrix < max_distance, axis=1)
        module_mask = np.any(dist_matrix[distance_mask] < max_distance, axis=0)
        source_pos = source_pos[distance_mask]
        source_dir = source_dir[distance_mask]
        source_time = source_time[distance_mask]
        source_nphotons = source_nphotons[distance_mask]

    record = MCRecord(
        "cascade",
        source_array_to_sources(source_pos, source_dir, source_time, source_nphotons),
        event_data,
    )

    propagation_result = _propagate_in_range_modules(
        det,
        module_mask,
        source_pos,
        source_dir,
        source_time,
        source_nphotons,
        pprop_func,
        k2,
        splitter,
    )

    return propagation_result, record


def generate_cascades(
    det,
    cylinder_height,
    cylinder_radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    particle_id,
    pprop_func,
    converter_func,
    noise_function=simulate_noise,
):
    """Generate a sample of cascades sampled uniformly in a cylinder.

    Parameters
    ----------
    det : Detector
        Detector instance used for propagation.
    cylinder_height : float
        Cylinder height for source sampling.
    cylinder_radius : float
        Cylinder radius for source sampling.
    nsamples : int
        Number of cascades to generate.
    seed : int
        RNG seed.
    log_emin : float
        Log10 of minimum energy.
    log_emax : float
        Log10 of maximum energy.
    particle_id : int
        PDG particle id used by the converter.
    pprop_func : callable
        Photon propagation function.
    converter_func : callable
        Function converting event parameters to source descriptions.
    noise_function : callable, optional
        Function used to simulate detector noise (default is ``simulate_noise``).

    Returns
    -------
    tuple
        Tuple ``(events, records)`` where ``events`` is a list of per-event
        ``awkward.Array`` objects and ``records`` is a list of corresponding
        ``MCRecord`` objects.
    """
    rng = np.random.RandomState(seed)
    key = random.PRNGKey(seed)

    events = []
    records = []

    for i in trange(nsamples):
        pos = sample_cylinder_volume(cylinder_height, cylinder_radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(log_emin, log_emax))
        dir = sample_direction(1, rng).squeeze()

        event_data = {
            "pos": pos,
            "dir": dir,
            "energy": energy,
            "time": 0,
            "particle_id": particle_id,
        }

        key, subkey = random.split(key)
        event, record = generate_cascade(
            det,
            event_data,
            subkey,
            pprop_func,
            converter_func,
        )
        if noise_function is not None:
            event, _ = noise_function(det, event)

        events.append(event)
        records.append(record)

    return events, records


# @profile
def generate_muon_energy_losses(
    propagator,
    energy,
    track_len,
    position,
    direction,
    time,
    key,
    loss_resolution=0.2,
    cont_resolution=1,
):
    try:
        import proposal as pp
    except ImportError as e:
        logger.critical("Could not import proposal!")
        raise e

    init_state = pp.particle.ParticleState()
    init_state.energy = energy * 1e3  # initial energy in MeV
    init_state.position = pp.Cartesian3D(position[0] * 100, position[1] * 100, position[2] * 100)
    init_state.direction = pp.Cartesian3D(direction[0], direction[1], direction[2])
    track = propagator.propagate(init_state, track_len * 100)  # cm

    aspos = []
    asdir = []
    astime = []
    asph = []

    loss_map = {
        "brems": 11,
        "epair": 11,
        "hadrons": 211,
        "ioniz": 11,
        "photonuclear": 211,
    }

    # all_losses = np.array([
    #     loss.energy / 1e3 for loss in track.stochastic_losses()
    # ])
    # with open("proposal_losses.txt", "ab") as f:
    #     np.savetxt(f, all_losses)
    # with open("proposal_losses.txt", "a") as f:
    #     f.write("\n New Event \n")
    # harvest losses
    for loss in track.stochastic_losses():
        # dist = loss.position.z / 100
        e_loss = loss.energy / 1e3

        # dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])

        # p = position + dist * direction
        # t = dist / Constants.c_vac + time

        p = np.asarray([loss.position.x, loss.position.y, loss.position.z]) / 100
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        t = np.linalg.norm(p - position) / Constants.c_vac + time

        loss_type_name = pp.particle.Interaction_Type(loss.type).name
        ptype = loss_map[loss_type_name]

        if e_loss < 1e3:
            spos, sdir, stime, sph = make_pointlike_cascade_source(p, t, dir, e_loss, ptype)
        else:
            key, subkey = random.split(key)
            spos, sdir, stime, sph = make_realistic_cascade_source(
                p,
                t,
                dir,
                e_loss,
                ptype,
                subkey,
                resolution=loss_resolution,
                moliere_rand=True,
            )

        aspos.append(spos)
        asdir.append(sdir)
        astime.append(stime)
        asph.append(sph)

    # distribute continuous losses uniformly along track
    # TODO: check if thats a good approximation
    # TODO: track segments

    cont_loss_sum = sum([loss.energy for loss in track.continuous_losses()]) / 1e3
    total_dist = track.track_propagated_distances()[-1] / 100
    loss_dists = np.arange(0, total_dist, cont_resolution)
    # TODO: Remove this really ugly fix
    if len(loss_dists) == 0:
        cont_loss_sum = 1.0
        total_dist = 1.1
        loss_dists = np.array([0.0, 1.0])
    e_loss = cont_loss_sum / len(loss_dists)

    for ld in loss_dists:
        p = ld * direction + position
        t = np.linalg.norm(p - position) / Constants.c_vac + time

        spos, sdir, stime, sph = make_pointlike_cascade_source(p, t, direction, e_loss, 11)

        aspos.append(spos)
        asdir.append(sdir)
        astime.append(stime)
        asph.append(sph)

    if not aspos:
        return None, None, None, None, total_dist

    return (
        np.concatenate(aspos),
        np.concatenate(asdir),
        np.concatenate(astime),
        np.concatenate(asph),
        total_dist,
    )


# @profile
def generate_realistic_track(
    det, event_data, key, pprop_func, proposal_prop, splitter=100000, max_distance=300.0
):
    """Generate a realistic muon track using energy losses from PROPOSAL.

    Parameters
    ----------
    det : Detector
        Detector instance used for propagation.
    event_data : dict
        Event parameters including position, direction, energy and length.
    key : jax.random.PRNGKey
        Random key for stochastic sampling.
    pprop_func : callable
        Photon propagation function.
    proposal_prop : callable
        PROPOSAL propagator instance.
    splitter : int, optional
        Split size for detector modules to reduce memory usage.
    max_distance : float, optional
        Maximum source-to-module distance in metres.  Energy-loss sources
        farther than this from every module are dropped before propagation.

    Returns
    -------
    tuple
        Tuple ``(propagation_result, record)`` where ``propagation_result`` is an
        ``awkward.Array`` of detected photon times and ``record`` is an
        ``MCRecord`` instance, or ``(None, None)`` when no sources remain.
    """

    if proposal_prop is None:
        raise RuntimeError()
    key, k1, k2 = random.split(key, 3)
    (
        source_pos,
        source_dir,
        source_time,
        source_photons,
        prop_dist,
    ) = generate_muon_energy_losses(
        proposal_prop,
        event_data["energy"],
        event_data["length"],
        event_data["pos"],
        event_data["dir"],
        event_data["time"],
        k1,
    )
    event_data["length"] = prop_dist

    if source_pos is None:
        return None, None

    # Early mask sources that are out of reach of every module, and modules
    # that are out of reach of every surviving source.
    dist_matrix = np.linalg.norm(
        source_pos[:, np.newaxis, ...] - det.module_coords[np.newaxis, ...], axis=-1
    )

    mask = np.any(dist_matrix < max_distance, axis=1)
    module_mask = np.any(dist_matrix[mask] < max_distance, axis=0)
    source_pos = source_pos[mask]
    source_dir = source_dir[mask]
    source_time = source_time[mask]
    source_photons = source_photons[mask]

    record = MCRecord(
        "realistic_track",
        source_array_to_sources(source_pos, source_dir, source_time, source_photons),
        event_data,
    )

    propagation_result = _propagate_in_range_modules(
        det,
        module_mask,
        source_pos,
        source_dir,
        source_time,
        source_photons,
        pprop_func,
        k2,
        splitter,
    )
    return propagation_result, record


def generate_realistic_tracks(
    det,
    cylinder_height,
    cylinder_radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    pprop_func,
    proposal_prop=None,
):
    """Generate realistic muon tracks sampled from the cylinder surface.

    Parameters
    ----------
    det : Detector
        Detector instance used for propagation.
    cylinder_height : float
        Cylinder height for source sampling.
    cylinder_radius : float
        Cylinder radius for source sampling.
    nsamples : int
        Number of tracks to generate.
    seed : int
        RNG seed.
    log_emin : float
        Log10 of minimum energy.
    log_emax : float
        Log10 of maximum energy.
    pprop_func : callable
        Photon propagation function.
    proposal_prop : callable, optional
        PROPOSAL propagator instance.

    Returns
    -------
    tuple
        Tuple ``(events, records)`` where ``events`` is a list of per-event
        ``awkward.Array`` objects and ``records`` is a list of corresponding
        ``MCRecord`` objects.
    """
    rng = np.random.RandomState(seed)
    key = random.PRNGKey(seed)

    events = []
    records = []

    for i in trange(nsamples):
        pos = sample_cylinder_surface(cylinder_height, cylinder_radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(log_emin, log_emax, size=1))
        # determine the surface normal vectors given the samples position
        # surface normal always points out

        if pos[2] == cylinder_height / 2:
            # upper cap
            area_norm = np.array([0, 0, 1])
        elif pos[2] == -cylinder_height / 2:
            # lower cap
            area_norm = np.array([0, 0, -1])
        else:
            area_norm = np.array(pos, copy=True)
            area_norm[2] = 0
            area_norm /= np.linalg.norm(area_norm)

        orientation = 1
        # Rejection sampling to generate only inward facing tracks
        while orientation > 0:
            direc = sample_direction(1, rng).squeeze()
            orientation = np.dot(area_norm, direc)

        # shift pos back by half the length:
        # pos = pos - track_length / 2 * direc

        isec = track_isects_cyl(det._outer_cylinder[0], det._outer_cylinder[1], pos, direc)
        track_length = 3000
        if (isec[0] != np.nan) and (isec[1] != np.nan):
            track_length = isec[1] - isec[0] + 300

        event_data = {
            "pos": pos,
            "dir": direc,
            "energy": energy,
            "time": 0,
            "length": track_length,
        }

        key, subkey = random.split(key)
        result = generate_realistic_track(
            det,
            event_data,
            key=subkey,
            proposal_prop=proposal_prop,
            pprop_func=pprop_func,
        )

        event, record = result
        event, _ = simulate_noise(det, event)

        events.append(event)
        records.append(record)

    return events, records


def generate_realistic_starting_tracks(
    det,
    cylinder_height,
    cylinder_radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    pprop_func,
    proposal_prop=None,
):
    """Generate realistic starting tracks (cascade + track).

    Parameters
    ----------
    det : Detector
        Detector instance used for propagation.
    cylinder_height : float
        Cylinder height for source sampling.
    cylinder_radius : float
        Cylinder radius for source sampling.
    nsamples : int
        Number of tracks to generate.
    seed : int
        RNG seed.
    log_emin : float
        Log10 of minimum energy.
    log_emax : float
        Log10 of maximum energy.
    pprop_func : callable
        Photon propagation function.
    proposal_prop : callable, optional
        PROPOSAL propagator instance.

    Returns
    -------
    tuple
        Tuple ``(events, records)`` where ``events`` is a list of per-event
        ``awkward.Array`` objects and ``records`` is a list of corresponding
        ``MCRecord`` objects.
    """
    rng = np.random.RandomState(seed)
    key, subkey = random.split(random.PRNGKey(seed))
    # Safe length to that tracks will appear infinite
    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    events = []
    records = []

    for i in trange(nsamples):
        pos = sample_cylinder_volume(cylinder_height, cylinder_radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(log_emin, log_emax))
        direc = sample_direction(1, rng).squeeze()
        inelas = rng.uniform(1e-6, 1 - 1e-6)

        event_data = {
            "pos": pos,
            "dir": direc,
            "energy": inelas * energy,
            "time": 0,
            "length": track_length,
        }

        track, track_record = generate_realistic_track(
            det,
            event_data,
            key=subkey,
            proposal_prop=proposal_prop,
            pprop_func=pprop_func,
        )

        event_data = {
            "pos": pos,
            "dir": direc,
            "energy": (1 - inelas) * energy,
            "time": 0,
            "length": track_length,
            "particle_id": 211,
        }

        cascade, cascade_record = generate_cascade(
            det,
            event_data,
            subkey,
            pprop_func,
            functools.partial(make_realistic_cascade_source, moliere_rand=True, resolution=0.2),
        )

        if (ak.count(track) == 0) & (ak.count(cascade) == 0):
            event = ak.Array([])

        elif ak.count(track) == 0:
            event = cascade
        elif (ak.count(cascade)) == 0:
            event = track
        else:
            event = ak.sort(ak.concatenate([track, cascade], axis=1))
        record = track_record + cascade_record

        event, _ = simulate_noise(det, event)
        events.append(event)
        records.append(record)

    return events, records
