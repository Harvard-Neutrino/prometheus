import glob
import logging
import os
import shutil
import subprocess

import numpy as np

from ..detector import Detector, write_om_dirs
from ..lepton_propagation import LeptonPropagator, Loss
from ..particle import Particle
from ..utils import serialize_to_f2k
from .photon_propagator import PhotonPropagator
from .utils import parse_ppc, should_propagate

logger = logging.getLogger(__name__)

# Collect subprocess statuses for run summaries
subprocess_statuses = []

_PPC_VERSIONS = ("bundled", "official")


def _uses_official_ppc(det: Detector, ppc_config: dict) -> bool:
    """Return whether the configured PPC is the official one.

    Raises
    ------
    ValueError
        If the setting is unknown, or multi-PMT modules are used with the
        bundled PPC, which does not support them.
    """
    version = ppc_config["simulation"].get("ppc_version", "bundled")
    if version not in _PPC_VERSIONS:
        raise ValueError(f"ppc_version must be one of {_PPC_VERSIONS!r}, got {version!r}")
    if det.needs_nextgen() and version != "official":
        raise ValueError(
            "Multi-PMT modules need the official PPC. Set ppc_exe to it and the simulation "
            "setting ppc_version to 'official'; see examples/README.md."
        )
    return version == "official"


def _write_module_files(det: Detector, paths: dict) -> None:
    """Put the files PPC needs for multi-PMT modules in its working directory."""
    tmpdir = paths["ppc_tmpdir"]
    det.to_om_conf(os.path.join(tmpdir, "om.conf"))
    det.to_om_map(os.path.join(tmpdir, "om.map"))

    om_dirs = paths.get("om_dirs", "")
    in_tables = os.path.join(paths["ppctables"], "om.dirs")
    target = os.path.join(tmpdir, "om.dirs")
    if om_dirs:
        if not os.path.exists(om_dirs):
            raise FileNotFoundError(f"om_dirs file {om_dirs} does not exist")
        shutil.copy(om_dirs, target)
    elif not os.path.exists(in_tables):
        write_om_dirs(target)
    elif not (os.path.exists(target) and os.path.samefile(in_tables, target)):
        shutil.copy(in_tables, target)

    for module_type in sorted({m.module_type for m in det.modules} - {-1}):
        name = f"om.wv_{module_type}.*"
        pattern = os.path.join(glob.escape(tmpdir), name)
        om_tables = paths.get("om_tables", "")
        if om_tables:
            for table in glob.glob(os.path.join(glob.escape(om_tables), name)):
                shutil.copy(table, tmpdir)
        # PPC only warns on stderr and then drops every hit in such modules.
        if not glob.glob(pattern):
            raise FileNotFoundError(
                f"No om.wv_{module_type}.* in {tmpdir}; PPC needs a wavelength "
                f"table for module type {module_type}. Add it to ppctables or om_tables."
            )


def ppc_sim(particle: Particle, det: Detector, lp: LeptonPropagator, ppc_config: dict) -> None:
    """Simulate the propagation of a particle and of any photons resulting from its energy losses.

    Parameters
    ----------
    particle : Particle
        Particle to propagate.
    det : Detector
        Detector object to simulate within.
    lp : LeptonPropagator
        Prometheus LeptonPropagator used to simulate any charged leptons.
    ppc_config : dict
        Dictionary containing the configuration settings for the photon
        propagation.
    """
    # TODO I think this could be factored out into a separate energy loss section
    # But that is not a now problem
    if abs(int(particle)) in [12, 14, 16]:  # It's a neutrino
        return
    # TODO put this in config
    r_inice = det.outer_radius + 1000
    if abs(int(particle)) in [11, 13, 15]:  # It's a charged lepton
        lp.energy_losses(particle, det)
    # All of these we consider as point depositions
    elif abs(int(particle)) in (211, 311, 321) or int(particle) == 111:  # pion or kaon
        # Point-deposit a pion or kaon. A particle and its antiparticle deposit
        # the same cascade light, so we normalise to the positive code with
        # abs(); the f2k cascade type then follows from int_type_to_str via
        # Loss.__str__:
        #   111 -> "epair" (EM cascade; pi0 -> gamma gamma)
        #   211 / 311 / 321 -> "hadr" (hadronic cascade)
        # pi0 is its own antiparticle, so only +111 is physical (a nonsense
        # -111 falls through to the else: raise below). Depositing here (instead
        # of the old early return) is the Issue #2 fix: it stops neutral-hadron
        # (pi0/K0) decay-product light from being silently dropped.
        if np.linalg.norm(particle.position - det.offset) <= r_inice:
            loss = Loss(abs(int(particle)), particle.e, particle.position)
            particle.losses.append(loss)
    elif int(particle) == -2000001006 or int(particle) == 2212:  # Hadrons
        if np.linalg.norm(particle.position - det.offset) <= r_inice:
            loss = Loss(int(particle), particle.e, particle.position)
            particle.losses.append(loss)
    else:
        # TODO make this into a custom error
        logger.error("Unrecognized particle: %r", particle)
        raise ValueError("Unrecognized particle")
    geo_tmpfile = f"{ppc_config['paths']['ppc_tmpdir']}/geo-f2k"
    ppc_tmpfile = (
        f"{ppc_config['paths']['ppc_tmpdir']}/{ppc_config['paths']['ppc_tmpfile']}_{str(particle)}"
    )
    f2k_tmpfile = (
        f"{ppc_config['paths']['ppc_tmpdir']}/{ppc_config['paths']['f2k_tmpfile']}_{str(particle)}"
    )
    command = (
        f"{ppc_config['paths']['ppc_exe']} {ppc_config['simulation']['device']}"
        f" < {f2k_tmpfile} > {ppc_tmpfile}"
    )
    # NOTE: stderr is intentionally NOT redirected to /dev/null here. It is
    # captured in Python below so a nonzero exit (bad ppc_exe path, missing
    # tables, etc.) is raised instead of silently looking like "no photon hits".

    if not should_propagate(particle):
        return
    official_ppc = _uses_official_ppc(det, ppc_config)
    serialize_to_f2k(particle, f2k_tmpfile, official_ppc=official_ppc)
    det.to_f2k(geo_tmpfile, serial_nos=[m.serial_no for m in det.modules])

    ppc_tmpdir = ppc_config["paths"]["ppc_tmpdir"]

    if det.needs_nextgen():
        _write_module_files(det, ppc_config["paths"])

    tenv = os.environ.copy()
    tenv["PPCTABLESDIR"] = ppc_tmpdir
    # PPC prefers this directory for om.conf and om.map, so a value inherited from
    # the user's shell would replace the files written above.
    tenv["NEXTGENDIR"] = ppc_tmpdir

    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=tenv
    )
    # communicate() (not wait()) so the captured stderr pipe can't deadlock.
    _, stderr_data = process.communicate()
    rc = process.returncode
    subprocess_statuses.append({"cmd": command, "returncode": rc})
    stderr_text = (stderr_data or b"").decode("utf-8", "replace")
    if rc != 0:
        logger.error(
            "PPC exited with code %d for command %r\nstderr:\n%s", rc, command, stderr_text
        )
        raise RuntimeError(f"PPC failed with exit code {rc}; see logged stderr")
    if stderr_text and not ppc_config["simulation"]["supress_output"]:
        logger.info("PPC stderr:\n%s", stderr_text)
    particle.hits = parse_ppc(ppc_tmpfile, official_ppc=official_ppc)
    if det.needs_nextgen() and any(h.pmt_id is None for h in particle.hits):
        raise RuntimeError(
            f"{ppc_config['paths']['ppc_exe']} ignored om.conf and returned hits without PMT "
            "IDs. Multi-PMT modules need the official PPC; see examples/README.md."
        )
    for f in [geo_tmpfile, f2k_tmpfile, ppc_tmpfile]:
        os.remove(f)

    for child in particle.children:
        # TODO put this in config
        if child.e < 1:  # GeV
            continue
        ppc_sim(child, det, lp, ppc_config)


from .registry import register_propagator  # noqa: E402


@register_propagator("ppc")
@register_propagator("ppc_cuda")
class PPCPhotonPropagator(PhotonPropagator):
    """Interface for simulating energy losses and light propagation using ppc."""

    def propagate(self, particle: Particle, rng_key=None) -> None:
        """Propagate an input particle using ppc.

        This modifies the state of the input particle in-place.

        Parameters
        ----------
        particle : Particle
            Prometheus particle to propagate.
        rng_key : Any or None
            The parameter is ignored and accepted for interface compatibility; ppc uses its own internal RNG.
        """  # noqa: E501
        return ppc_sim(particle, self.detector, self.lepton_propagator, self.config)
