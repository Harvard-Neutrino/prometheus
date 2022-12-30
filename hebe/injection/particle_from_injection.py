import numpy as np

from .injection import Injection
from ..particle import Particle

def particle_from_injection(injection: Injection, key:str, idx: int) -> Particle:
    pdg_code = getattr(injection, f"{key}_type")[idx]
    pos = np.array([
        getattr(injection, f"{key}_position_x")[idx],
        getattr(injection, f"{key}_position_y")[idx],
        getattr(injection, f"{key}_position_z")[idx],
    ])
    zen = getattr(injection, f"{key}_direction_theta")[idx]
    azi = getattr(injection, f"{key}_direction_phi")[idx]
    direction = [
        np.cos(azi)*np.sin(zen),
        np.sin(azi)*np.sin(zen),
        np.cos(zen)
    ]

    primary_particle = Particle(
        pdg_code,
        getattr(injection, f"{key}_energy")[idx],
        pos,
        direction,
        theta=zen,
        phi=azi
    )
    return primary_particle
