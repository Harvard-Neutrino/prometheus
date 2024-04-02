"""Event Generators."""
import functools
import logging

import awkward as ak
import jax.numpy as jnp
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
from .photon_propagation.utils import source_array_to_sources
from .photon_source import PhotonSource
from .utils import track_isects_cyl

logger = logging.getLogger(__name__)


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
):
    """
    Generate a single cascade with given amplitude and position and return time of detected photons.

    Parameters:
        det: Detector
            Instance of Detector class
        event_data: dict
            Container of the event data
        seed: int
        pprop_func: function
            Function to calculate the photon signal
        converter_func: function
            Function to calculate number of photons as function of energy
        splitter: int
            Subset of the modules to use per run. This is for memory

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

    record = MCRecord(
        "cascade",
        source_array_to_sources(source_pos, source_dir, source_time, source_nphotons),
        event_data,
    )

    # splitting for memory efficiency
    if det.module_coords.shape[0] > splitter:
        det_subsets_coords = np.array_split(
            det.module_coords,
            det.module_coords.shape[0] % splitter
        )
        det_subsets_eff = np.array_split(
            det.module_efficiencies,
            det.module_coords.shape[0] % splitter
        )
        propagation_result = [
            pprop_func(
            det_subsets_coords[id_set],
            det_subsets_eff[id_set],
            source_pos,
            source_dir,
            source_time,
            source_nphotons,
            seed=k2,
        ) for id_set, _ in enumerate(det_subsets_coords)
        ]
        propagation_result = ak.concatenate(propagation_result)
    else:
        propagation_result = pprop_func(
        det.module_coords,
        det.module_efficiencies,
        source_pos,
        source_dir,
        source_time,
        source_nphotons,
        seed=k2,
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
    """Generate a sample of cascades, randomly sampling the positions in a cylinder of given radius and length."""
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
    init_state.position = pp.Cartesian3D(
        position[0] * 100, position[1] * 100, position[2] * 100
    )
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

        """
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        
        p = position + dist * direction
        t = dist / Constants.c_vac + time
        """

        p = np.asarray([loss.position.x, loss.position.y, loss.position.z]) / 100
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        t = np.linalg.norm(p - position) / Constants.c_vac + time

        loss_type_name = pp.particle.Interaction_Type(loss.type).name
        ptype = loss_map[loss_type_name]

        if e_loss < 1e3:
            spos, sdir, stime, sph = make_pointlike_cascade_source(
                p, t, dir, e_loss, ptype
            )
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
    if len (loss_dists) == 0:
        cont_loss_sum = 1.
        total_dist = 1.1
        loss_dists = np.array([0., 1.])
    e_loss = cont_loss_sum / len(loss_dists)

    for ld in loss_dists:
        p = ld * direction + position
        t = np.linalg.norm(p - position) / Constants.c_vac + time

        spos, sdir, stime, sph = make_pointlike_cascade_source(
            p, t, direction, e_loss, 11
        )

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
    det,
    event_data,
    key,
    pprop_func,
    proposal_prop,
    splitter=100000
):
    """
    Generate a realistic track using energy losses from PROPOSAL.

    Parameters:
        det: Detector
            Instance of Detector class
        event_data: dict
            Container of the event data
        seed: PRNGKey
        pprop_func: function
            Function to calculate the photon signal
        proposal_prop: function
            Propoposal propagator
        splitter: int
            Splits the detector modules in splitter sized chunks for memory efficiency
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

    # early mask sources that are out of reach

    dist_matrix = np.linalg.norm(
        source_pos[:, np.newaxis, ...] - det.module_coords[np.newaxis, ...], axis=-1
    )

    mask = np.any(dist_matrix < 300, axis=1)
    source_pos = source_pos[mask]
    source_dir = source_dir[mask]
    source_time = source_time[mask]
    source_photons = source_photons[mask]

    record = MCRecord(
        "realistic_track",
        source_array_to_sources(source_pos, source_dir, source_time, source_photons),
        event_data,
    )
    # splitting for memory efficiency
    if det.module_coords.shape[0] > splitter:
        det_subsets_coords = np.array_split(
            det.module_coords,
            det.module_coords.shape[0] % splitter
        )
        det_subsets_eff = np.array_split(
            det.module_efficiencies,
            det.module_coords.shape[0] % splitter
        )
        propagation_result = [
            pprop_func(
                det_subsets_coords[id_set],
                det_subsets_eff[id_set],
                source_pos,
                source_dir,
                source_time,
                source_photons,
                seed=k2,
            ) for id_set, _ in enumerate(det_subsets_coords)
        ]
        propagation_result = ak.concatenate(propagation_result)
    else:
        propagation_result = pprop_func(
                det.module_coords,
                det.module_efficiencies,
                source_pos,
                source_dir,
                source_time,
                source_photons,
                seed=k2,
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
    """Generate realistic muon tracks."""
    rng = np.random.RandomState(seed)
    key = random.PRNGKey(seed)

    events = []
    records = []

    for i in trange(nsamples):
        pos = sample_cylinder_surface(
            cylinder_height, cylinder_radius, 1, rng
        ).squeeze()
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

        isec = track_isects_cyl(
            det._outer_cylinder[0], det._outer_cylinder[1], pos, direc
        )
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
    """Generate realistic starting tracks (cascade + track)."""
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
            functools.partial(
                make_realistic_cascade_source, moliere_rand=True, resolution=0.2
            ),
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
