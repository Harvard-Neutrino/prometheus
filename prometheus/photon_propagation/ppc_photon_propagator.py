import numpy as np
from typing import Tuple
import os
import subprocess

from .photon_propagator import PhotonPropagator
from .utils import should_propagate, parse_ppc
from ..lepton_propagation import LeptonPropagator, Loss
from ..detector import Detector
from ..particle import Particle
from ..utils import serialize_to_f2k, PDG_to_f2k


def ppc_sim(
    particle: Particle,
    det: Detector,
    lp: LeptonPropagator,
    ppc_config: dict
) -> Tuple[None, None]:
    """
    """
    geo_tmpfile = f"{ppc_config['paths']['ppctables']}/geo-f2k"
    ppc_file = f"{ppc_config['paths']['ppc_tmpfile']}_{str(particle)}"
    f2k_file = f"{ppc_config['paths']['f2k_tmpfile']}_{str(particle)}"
    command = f"{ppc_config['paths']['ppc_exe']} {ppc_config['simulation']['device']} < {f2k_file} > {ppc_file}"
    if ppc_config["simulation"]["supress_output"]:
        command += " 2>/dev/null"
    # TODO This could all be factored out into a LP step
    if abs(int(particle)) in [12, 14, 16]: # It's a neutrino
        return None, None
    # TODO put this in config
    r_inice = det.outer_radius + 1000
    if abs(int(particle)) in [11, 13, 15]: # It's a charged lepton
        lp.energy_losses(particle, det)
        for child in particle.children:
            # TODO put this in config
            if child.e > 1: # GeV
                ppc_sim(child, det, lp, ppc_config)
    # All of these we consider as point depositions
    elif abs(int(particle))==111: # It's a neutral pion
        # TODO handle this correctl by converting to photons after prop
        return None, None
    elif abs(int(particle))==211 or abs(int(particle))==321: # It's a charged pion
        if np.linalg.norm(particle.position-det.offset) <= r_inice:
            loss = Loss(int(particle), particle.e, particle.position)
            particle.add_loss(loss)
    elif abs(int(particle))==311: # It's a neutral kaon
        # TODO handle this correctl by converting to photons after prop
        return None, None
    elif int(particle)==-2000001006 or int(particle)==2212: # Hadrons
        if np.linalg.norm(particle.position-det.offset) <= r_inice:
            loss = Loss(int(particle), particle.e, particle.position)
            particle.add_loss(loss)
    else:
        print(repr(particle))
        raise ValueError("Unrecognized particle")

    if not should_propagate(particle):
        return None, None
    serialize_to_f2k(particle, f2k_file)
    det.to_f2k(
        geo_tmpfile,
        serial_nos=[m.serial_no for m in det.modules]
    )
    tenv = os.environ.copy()
    tenv["PPCTABLESDIR"] = ppc_config["paths"]["ppctables"]

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, env=tenv)
    process.wait()
    particle._hits = parse_ppc(ppc_file)
    return None, None

class PPCPhotonPropagator(PhotonPropagator):
    """Interface for simulating energy losses and light propagation using PPC"""
    def __init__(
        self,
        lepton_propagator: LeptonPropagator,
        detector: Detector,
        photon_prop_config: dict
    ):
        """Initialize the PhotonPropagator object
        
        params
        ______
        lepton_propagator: Prometheus LeptonPropagator object which will be used
            to generate losses from the particles
        detector: Prometheus detector object in which the light will be
            propagated
        """
        super().__init__(lepton_propagator, detector, photon_prop_config)

    def propagate(self, particle: Particle) -> Tuple[None, None]:
        """Propagate input particle using PPC. This returns None for consistency with
        Olympus. Instead it modifies the state of the input Particle. We should make this
        more consistent but that is a problem for another day...

        params
        ______
        particle: Prometheus particle to propagate
        """
        return ppc_sim(particle, self.detector, self.lepton_propagator, self.config)
