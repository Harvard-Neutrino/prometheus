import logging
from typing import Iterable, Optional

import numpy as np

from .. import Particle, PropagatableParticle
from ..genie_parser import genie_loader
from ..injection_event.injection_event import InjectionEvent
from ..interactions import Interactions
from .injection import Injection

logger = logging.getLogger(__name__)


class GENIEInjection(Injection):
    """Injection constructed from GENIE output."""

    def __init__(self, events: Iterable[InjectionEvent]):
        if not all(isinstance(e, InjectionEvent) for e in events):
            raise ValueError("GENIEInjection requires InjectionEvent instances")
        super().__init__(events)


_M_PI0 = 0.134977  # GeV — π⁰ mass
_pi0_decay_warned = False


def _direction_from_p3(p3: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(p3)
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return p3 / norm


def _decay_pi0(
    rng: np.random.Generator,
    energy: float,
    p3: np.ndarray,
    position: np.ndarray,
    parent: Particle,
) -> list:
    """Decay π⁰ → γγ and return two PropagatableParticle photons.

    The π⁰ is assumed to decay instantaneously at the event vertex.
    Prometheus does not propagate the π⁰ itself; the decay is applied
    at injection time so that only the resulting photons enter the
    simulation pipeline.

    .. warning::
        This is an instantaneous-decay approximation. The π⁰ flight
        length before decay (cτ ≈ 25 nm) is negligible for detector
        scales and is therefore ignored.

    The decay angle is sampled isotropically in the π⁰ rest frame and
    boosted to the lab frame via the relativistic aberration formula.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for the rest-frame decay angle.
    energy : float
        π⁰ energy in the lab frame in GeV.
    p3 : np.ndarray
        π⁰ three-momentum in the lab frame in GeV/c.
    position : np.ndarray
        Decay vertex position in metres.
    parent : Particle
        Parent particle of the π⁰ (used to set the ``parent`` field on the photons).

    Returns
    -------
    list of PropagatableParticle
        Two photon particles produced by the decay.
    """
    global _pi0_decay_warned
    if not _pi0_decay_warned:
        logger.warning(
            "π⁰ (PDG 111) is decayed instantaneously to γγ at the event vertex. "
            "Prometheus does not propagate π⁰ directly. "
            "(This message is shown once per run.)"
        )
        _pi0_decay_warned = True

    p3 = np.asarray(p3, dtype=float)
    p_mag = np.linalg.norm(p3)

    # Degenerate case: π⁰ nearly at rest — split energy equally along z.
    if p_mag < 1e-9 or energy <= _M_PI0:
        dir0 = np.array([0.0, 0.0, 1.0])
        return [
            PropagatableParticle(
                22, energy / 2.0, position.copy(), dir0.copy(), None, parent, time=0.0
            ),
            PropagatableParticle(22, energy / 2.0, position.copy(), -dir0, None, parent, time=0.0),
        ]

    # Boost parameters.
    beta = p_mag / energy
    gamma_lor = energy / _M_PI0

    # π⁰ direction — defines the boost axis.
    pi0_dir = p3 / p_mag

    # Isotropic decay in the π⁰ rest frame.
    cos_star = rng.uniform(-1.0, 1.0)
    phi_star = rng.uniform(0.0, 2.0 * np.pi)

    # Energies in the lab frame via relativistic Doppler formula.
    e_star = _M_PI0 / 2.0
    e1 = gamma_lor * e_star * (1.0 + beta * cos_star)
    e2 = gamma_lor * e_star * (1.0 - beta * cos_star)

    # Lab-frame polar angles via relativistic aberration.
    cos1 = (cos_star + beta) / (1.0 + beta * cos_star)
    cos2 = (-cos_star + beta) / (1.0 - beta * cos_star)
    sin1 = np.sqrt(max(1.0 - cos1**2, 0.0))
    sin2 = np.sqrt(max(1.0 - cos2**2, 0.0))

    # Orthonormal frame with pi0_dir as the z-axis.
    ref = np.array([1.0, 0.0, 0.0]) if abs(pi0_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_ax = np.cross(pi0_dir, ref)
    x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(pi0_dir, x_ax)

    dir1 = cos1 * pi0_dir + sin1 * (np.cos(phi_star) * x_ax + np.sin(phi_star) * y_ax)
    phi2 = phi_star + np.pi
    dir2 = cos2 * pi0_dir + sin2 * (np.cos(phi2) * x_ax + np.sin(phi2) * y_ax)

    dir1 /= np.linalg.norm(dir1)
    dir2 /= np.linalg.norm(dir2)

    return [
        PropagatableParticle(22, e1, position.copy(), dir1, None, parent, time=0.0),
        PropagatableParticle(22, e2, position.copy(), dir2, None, parent, time=0.0),
    ]


def _interaction_from_descr(descr: str) -> Interactions:
    if "CC" in descr:
        return Interactions.CHARGED_CURRENT
    if "NC" in descr:
        return Interactions.NEUTRAL_CURRENT
    return Interactions.OTHER


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a rotation matrix drawn uniformly from SO(3).

    Sampled via a uniform random unit quaternion (a normalised 4-vector of
    standard normals is uniform on the 3-sphere).

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Shape (3, 3) rotation matrix.
    """
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _sample_cylinder(rng: np.random.Generator, radius: float, half_height: float) -> np.ndarray:
    r = radius * np.sqrt(rng.uniform())
    theta = rng.uniform(0.0, 2.0 * np.pi)
    z = rng.uniform(-half_height, half_height)
    return np.array([r * np.cos(theta), r * np.sin(theta), z])


def injection_from_genie_output(
    genie_root_file: str,
    *,
    simulation_config=None,
    detector=None,
    detector_offset: Optional[np.ndarray] = None,
    **kwargs,
) -> GENIEInjection:
    """Build a GENIEInjection by reading a GENIE ROOT file directly.

    Parameters
    ----------
    genie_root_file : str
        Path to the GENIE gRooTracker ROOT file.
    simulation_config : GENIESimConfig, optional
        Placement and seed configuration. Uses ``placement='fixed'`` at the
        origin when not provided.
    detector : Detector, optional
        Detector object, required for ``placement='random'``.
    detector_offset : np.ndarray, optional
        Detector centre in metres, used as the origin for random placement.
        Defaults to zero if not provided.

    Returns
    -------
    GENIEInjection
        Injection object ready for the Prometheus simulation pipeline.
    """
    placement = "fixed"
    positions = None
    n_simulate = None
    seed = None
    interaction_filter = None
    direction_mode = "as-is"

    if simulation_config is not None:
        placement = getattr(simulation_config, "placement", "fixed")
        positions = getattr(simulation_config, "positions", None)
        n_simulate = getattr(simulation_config, "n_events", None)
        seed = getattr(simulation_config, "random_state_seed", None)
        interaction_filter = getattr(simulation_config, "interaction_filter", None)
        direction_mode = getattr(simulation_config, "direction_mode", "as-is") or "as-is"

    if direction_mode not in ("as-is", "isotropic"):
        raise ValueError(
            f"Unknown direction_mode: {direction_mode!r}. Use 'as-is' or 'isotropic'."
        )

    offset = np.zeros(3) if detector_offset is None else np.asarray(detector_offset, dtype=float)

    logger.info("Loading GENIE events from %s (placement=%s)", genie_root_file, placement)
    events_df = genie_loader(genie_root_file)
    n_file = len(events_df)
    logger.info("Loaded %d GENIE events from file", n_file)

    if interaction_filter is not None:
        mask = events_df["event_descr"].str.contains(interaction_filter, case=False)
        events_df = events_df[mask].reset_index(drop=True)
        logger.info(
            "Filtered to %d events matching interaction_filter=%r (from %d)",
            len(events_df), interaction_filter, n_file,
        )
        if len(events_df) == 0:
            raise ValueError(
                f"No GENIE events remaining after filtering for {interaction_filter!r}. "
                "Check the ROOT file and interaction_filter value."
            )

    # Use post-filter count as the pool size for resampling.
    n_pool = len(events_df)

    rng = np.random.default_rng(seed)

    if n_simulate is not None and n_simulate != n_pool:
        indices = rng.choice(n_pool, size=n_simulate, replace=True)
        events_df = events_df.iloc[indices].reset_index(drop=True)
        logger.info(
            "Resampled to %d events (with replacement) from %d source events",
            n_simulate,
            n_file,
        )

    n_events = len(events_df)

    if placement == "random":
        if detector is None:
            raise ValueError(
                "detector must be provided to Prometheus when using placement='random'"
            )
        radius, height = detector.outer_cylinder
        half_height = height / 2.0
        vertices = [_sample_cylinder(rng, radius, half_height) + offset for _ in range(n_events)]
    elif placement == "fixed":
        if positions is None:
            vertices = [offset.copy() for _ in range(n_events)]
        else:
            pos_arr = np.asarray(positions, dtype=float)
            if pos_arr.ndim == 1:
                vertices = [pos_arr.copy() for _ in range(n_events)]
            else:
                if len(pos_arr) != n_events:
                    raise ValueError(
                        f"positions has {len(pos_arr)} entries but ROOT file has {n_events} events"
                    )
                vertices = [pos_arr[i].copy() for i in range(n_events)]
    else:
        raise ValueError(f"Unknown placement: {placement!r}. Use 'random' or 'fixed'.")

    injection_events = []
    for (_, row), vertex in zip(events_df.iterrows(), vertices):
        init_p3 = np.asarray(row["init_inj_p"], dtype=float)
        final_p3s = [np.asarray(p3, dtype=float) for p3 in row["final_p"]]

        # One common rotation per event keeps all intra-event angles intact,
        # so kinematics are unchanged while the event orientation becomes
        # isotropic (exact for interactions on an unpolarized target).
        if direction_mode == "isotropic":
            rotation = _random_rotation(rng)
            init_p3 = rotation @ init_p3
            final_p3s = [rotation @ p3 for p3 in final_p3s]

        init_dir = _direction_from_p3(init_p3)
        initial_state = Particle(
            int(row["init_inj_id"]),
            float(row["init_inj_e"]),
            vertex.copy(),
            init_dir,
            None,
        )

        final_states = []
        for pdg, energy, p3 in zip(row["final_ids"], row["final_e"], final_p3s):
            if int(pdg) == 111:
                final_states.extend(
                    _decay_pi0(rng, float(energy), p3, vertex, initial_state)
                )
            else:
                direction = _direction_from_p3(p3)
                final_states.append(
                    PropagatableParticle(
                        int(pdg),
                        float(energy),
                        vertex.copy(),
                        direction,
                        None,
                        initial_state,
                        time=0.0,  # interaction-vertex particle starts at t=0
                    )
                )

        interaction = _interaction_from_descr(str(row["event_descr"]))
        injection_events.append(
            InjectionEvent(
                initial_state=initial_state,
                final_states=final_states,
                interaction=interaction,
                vertex_x=float(vertex[0]),
                vertex_y=float(vertex[1]),
                vertex_z=float(vertex[2]),
            )
        )

    logger.info("Built %d GENIEInjectionEvents", len(injection_events))
    return GENIEInjection(injection_events)
