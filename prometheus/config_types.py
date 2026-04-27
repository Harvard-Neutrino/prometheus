# config_types.py
# Typed configuration dataclasses for Prometheus.
#
# All config objects inherit from ConfigBase which provides dict-style access
# (config["key"]) as a backward-compatibility shim.  Keys are normalised to
# Python identifiers by lower-casing and replacing spaces/hyphens with
# underscores before looking up the matching field.  A _KEY_MAP class variable
# handles the handful of cases where that rule is not sufficient (e.g.
# "LeptonInjector" -> "lepton_injector").

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import ClassVar, Optional

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"


# ---------------------------------------------------------------------------
# Base class providing the dict-compatibility shim
# ---------------------------------------------------------------------------


class ConfigBase:
    """Mixin that lets dataclass instances be accessed like dicts.

    ``obj["key"]`` normalises ``key`` to a Python identifier via
    ``key.lower().replace(" ", "_").replace("-", "_")`` and then falls back to
    the optional ``_KEY_MAP`` class variable for any remaining exceptions.
    """

    _KEY_MAP: ClassVar[dict[str, str]] = {}

    def _normalize(self, key: str) -> str:
        if key in self._KEY_MAP:
            return self._KEY_MAP[key]
        return key.lower().replace(" ", "_").replace("-", "_")

    def __getitem__(self, key: str):
        attr = self._normalize(key)
        try:
            return getattr(self, attr)
        except AttributeError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value) -> None:
        attr = self._normalize(key)
        if not hasattr(self, attr):
            raise KeyError(
                f"Unknown config key {key!r} (normalised: {attr!r}). "
                f"Valid keys: {[f.name for f in fields(self)]}"
            )
        setattr(self, attr, value)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False

    def keys(self):
        return [f.name for f in fields(self)]

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def to_dict(self) -> dict:
        """Return a plain nested dict (suitable for ``json.dumps``).

        Returns
        -------
        dict
            Nested dictionary representation of the config.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class RunConfig(ConfigBase):
    """Run-related configuration.

    Parameters
    ----------
    run_number : int
        Unique run identifier.
    nevents : int
        Number of events to simulate (must be > 0).
    verbosity : str or int
        Logging verbosity. Allowed string values: 'DEBUG', 'INFO', 'WARNING', 'ERROR',
        'CRITICAL' (case-insensitive) or a numeric logging level.
    logfile : str, optional
        Path to a file to write logs. If None, logs go to console.
    log_format : str, optional
        Logging format string for handlers.
    storage_prefix : str
        Base directory for output files.
    outfile : str, optional
        Explicit output file path (parquet). If None, a path under storage_prefix will be used.
    random_state_seed : int, optional
        Seed for RNGs used in injection/propagation.
    summary_mode : str
        Reporting mode. Allowed values: 'user' (default) and 'debug'.
        'user' prints a user-friendly, compact summary to the console and collapses
        noisy third-party output.
        'debug' prints a developer-oriented summary at DEBUG level with verbose logging
        of captured warnings and native prints.
    banner : bool
        Show ASCII banner from assets when True.
    compact : bool
        Emit a compact single-line summary for batch runs when True.
    summary_json : bool
        If True, write a machine-readable JSON summary alongside the output file.
    summary_json_path : str, optional
        Explicit path for JSON summary (overrides default outfile + '.summary.json').
    progress_threshold : int
        Minimum number of events required to display progress bars (tqdm).

    Notes
    -----
    To enable the debug summary when running examples that do not expose CLI flags, set::

        from prometheus import config
        config.run.summary_mode = 'debug'
        config.run.verbosity = 'DEBUG'  # optional: enable logger DEBUG-level output

    The ``verbosity`` parameter accepts either standard logging level names or integers
    (see Python's ``logging`` module).
    """

    run_number: int = 1337
    nevents: int = 10
    verbosity: str = "WARNING"
    logfile: Optional[str] = None
    log_format: Optional[str] = None
    storage_prefix: str = "./output/"
    outfile: Optional[str] = None
    random_state_seed: Optional[int] = None
    summary_mode: str = "user"
    banner: bool = False
    compact: bool = False
    summary_json: bool = False
    summary_json_path: Optional[str] = None
    progress_threshold: int = 10

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "run number": "run_number",
        "storage prefix": "storage_prefix",
        "random state seed": "random_state_seed",
        "log file": "logfile",
        "log format": "log_format",
        "summary mode": "summary_mode",
        "banner": "banner",
        "compact": "compact",
        "summary json": "summary_json",
        "summary json path": "summary_json_path",
        "progress threshold": "progress_threshold",
    }

    def __post_init__(self):
        if self.nevents is not None and self.nevents <= 0:
            raise ValueError(f"run.nevents must be > 0, got {self.nevents}")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


@dataclass
class DetectorConfig(ConfigBase):
    """Detector configuration."""

    geo_file: Optional[str] = None
    offset: Optional[list] = None

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "geo file": "geo_file",
    }


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


@dataclass
class LIPathsConfig(ConfigBase):
    """LeptonInjector file path configuration."""

    install_location: str = field(
        default_factory=lambda: (
            f"{sys.prefix}/lib/python"
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f"/site-packages"
        )
    )
    xsec_dir: str = field(default_factory=lambda: str(RESOURCES_DIR / "cross_section_splines"))
    earth_model_location: Optional[str] = None
    injection_file: Optional[str] = None
    lic_file: Optional[str] = None
    diff_xsec: Optional[str] = None
    total_xsec: Optional[str] = None

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "install location": "install_location",
        "xsec dir": "xsec_dir",
        "earth model location": "earth_model_location",
        "injection file": "injection_file",
        "lic file": "lic_file",
        "diff xsec": "diff_xsec",
        "total xsec": "total_xsec",
    }


@dataclass
class LISimulationConfig(ConfigBase):
    """LeptonInjector simulation parameters configuration."""

    final_state_1: str = "MuMinus"
    final_state_2: str = "Hadrons"
    minimal_energy: float = 1e2
    maximal_energy: float = 1e6
    power_law: float = 1.0
    min_zenith: float = 0.0
    max_zenith: float = 180.0
    min_azimuth: float = 0.0
    max_azimuth: float = 360.0
    is_ranged: Optional[bool] = None
    injection_radius: Optional[float] = None
    endcap_length: Optional[float] = None
    cylinder_radius: Optional[float] = None
    cylinder_height: Optional[float] = None
    nevents: Optional[int] = None
    random_state_seed: Optional[int] = None

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "final state 1": "final_state_1",
        "final state 2": "final_state_2",
        "minimal energy": "minimal_energy",
        "maximal energy": "maximal_energy",
        "power law": "power_law",
        "min zenith": "min_zenith",
        "max zenith": "max_zenith",
        "min azimuth": "min_azimuth",
        "max azimuth": "max_azimuth",
        "is ranged": "is_ranged",
        "injection radius": "injection_radius",
        "endcap length": "endcap_length",
        "cylinder radius": "cylinder_radius",
        "cylinder height": "cylinder_height",
        "random state seed": "random_state_seed",
    }


@dataclass
class LeptonInjectorConfig(ConfigBase):
    """Top-level LeptonInjector configuration."""

    inject: bool = True
    paths: LIPathsConfig = field(default_factory=LIPathsConfig)
    simulation: LISimulationConfig = field(default_factory=LISimulationConfig)


@dataclass
class SimpleInjectorPathsConfig(ConfigBase):
    """Simple injector file path configuration."""

    injection_file: Optional[str] = None

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "injection file": "injection_file",
    }


@dataclass
class SimpleInjectorConfig(ConfigBase):
    """Top-level simple injector configuration."""

    inject: bool = False
    paths: SimpleInjectorPathsConfig = field(default_factory=SimpleInjectorPathsConfig)
    simulation: dict = field(default_factory=dict)


@dataclass
class InjectionConfig(ConfigBase):
    """Injection configuration."""

    name: str = "LeptonInjector"
    lepton_injector: LeptonInjectorConfig = field(default_factory=LeptonInjectorConfig)
    prometheus_injector: SimpleInjectorConfig = field(default_factory=SimpleInjectorConfig)
    genie: SimpleInjectorConfig = field(default_factory=SimpleInjectorConfig)

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "LeptonInjector": "lepton_injector",
        "Prometheus": "prometheus_injector",
        "GENIE": "genie",
    }


# ---------------------------------------------------------------------------
# Lepton propagator
# ---------------------------------------------------------------------------


@dataclass
class ProposalPathsConfig(ConfigBase):
    """PROPOSAL file path configuration."""

    tables_path: str = field(default_factory=lambda: str(RESOURCES_DIR / "PROPOSAL_tables"))
    earth_model_location: Optional[str] = None

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "tables path": "tables_path",
        "earth model location": "earth_model_location",
    }


@dataclass
class ProposalSimConfig(ConfigBase):
    """PROPOSAL simulation parameters configuration."""

    vcut: float = 0.1
    ecut: float = 0.5
    interpolation: bool = True
    lpm_effect: bool = True
    continuous_randomization: bool = True
    soft_losses: bool = True
    interpolate: bool = True
    decay: bool = True
    scattering_model: str = "Moliere"
    propagation_padding: Optional[float] = None
    medium: Optional[str] = None

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "lpm effect": "lpm_effect",
        "continuous randomization": "continuous_randomization",
        "soft losses": "soft_losses",
        "scattering model": "scattering_model",
        "propagation padding": "propagation_padding",
    }


@dataclass
class NewProposalConfig(ConfigBase):
    """Configuration for the new PROPOSAL lepton propagator."""

    paths: ProposalPathsConfig = field(default_factory=ProposalPathsConfig)
    simulation: ProposalSimConfig = field(default_factory=ProposalSimConfig)


@dataclass
class OldProposalSimConfig(ConfigBase):
    """Old PROPOSAL simulation parameters configuration."""

    vcut: float = 1.0
    ecut: float = 0.1
    interpolation: bool = True
    lpm_effect: bool = True
    continuous_randomization: bool = True
    soft_losses: bool = True
    scattering_model: str = "Moliere"
    propagation_padding: Optional[float] = None
    medium: Optional[str] = None

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "lpm effect": "lpm_effect",
        "continuous randomization": "continuous_randomization",
        "soft losses": "soft_losses",
        "scattering model": "scattering_model",
        "propagation padding": "propagation_padding",
    }


@dataclass
class OldProposalPathsConfig(ConfigBase):
    """Old PROPOSAL file path configuration."""

    tables_path: str = "~/.local/share/PROPOSAL/tables"
    earth_model_location: Optional[str] = None

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "tables path": "tables_path",
        "earth model location": "earth_model_location",
    }


@dataclass
class OldProposalConfig(ConfigBase):
    """Configuration for the old PROPOSAL lepton propagator."""

    paths: OldProposalPathsConfig = field(default_factory=OldProposalPathsConfig)
    simulation: OldProposalSimConfig = field(default_factory=OldProposalSimConfig)


@dataclass
class LeptonPropagatorConfig(ConfigBase):
    """Top-level lepton propagator configuration."""

    name: str = "new proposal"
    version: Optional[str] = None
    new_proposal: NewProposalConfig = field(default_factory=NewProposalConfig)
    old_proposal: OldProposalConfig = field(default_factory=OldProposalConfig)

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "new proposal": "new_proposal",
        "old proposal": "old_proposal",
    }


# ---------------------------------------------------------------------------
# Photon propagator
# ---------------------------------------------------------------------------


@dataclass
class OlympusPathsConfig(ConfigBase):
    """Olympus file path configuration."""

    location: str = field(default_factory=lambda: str(RESOURCES_DIR / "olympus_resources"))
    photon_model: str = "pone_config.json"
    flow: str = "photon_arrival_time_nflow_params.pickle"
    counts: str = "photon_arrival_time_counts_params.pickle"

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "photon model": "photon_model",
    }


@dataclass
class OlympusSimConfig(ConfigBase):
    """Olympus simulation parameters configuration."""

    files: bool = True
    wavelength: int = 700
    splitter: int = 100000


@dataclass
class OlympusParticlesConfig(ConfigBase):
    """Olympus particle type configuration."""

    track_particles: list = field(default_factory=lambda: [13, -13])
    explicit: list = field(default_factory=lambda: [11, -11, 111, 211, 13, -13, 15, -15])
    replacement: int = 2212

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "track particles": "track_particles",
    }


@dataclass
class OlympusConfig(ConfigBase):
    """Top-level Olympus photon propagator configuration."""

    paths: OlympusPathsConfig = field(default_factory=OlympusPathsConfig)
    simulation: OlympusSimConfig = field(default_factory=OlympusSimConfig)
    particles: OlympusParticlesConfig = field(default_factory=OlympusParticlesConfig)


@dataclass
class PPCPathsConfig(ConfigBase):
    """ppc file path configuration."""

    location: str = field(default_factory=lambda: f"{RESOURCES_DIR}/PPC_executables/PPC/")
    force: bool = False
    ppc_tmpdir: str = "./.ppc_tmp"
    ppc_tmpfile: str = ".event_hits.ppc.tmp"
    f2k_tmpfile: str = ".event_losses.f2k.tmp"
    ppc_prefix: str = ""
    f2k_prefix: str = ""
    ppctables: str = "../resources/PPC_tables/south_pole/"
    ppc_exe: str = "../resources/PPC_executables/PPC/ppc"


@dataclass
class PPCSimConfig(ConfigBase):
    """ppc simulation parameters configuration."""

    device: int = 0
    supress_output: bool = True


@dataclass
class PPCConfig(ConfigBase):
    """Top-level ppc photon propagator configuration."""

    paths: PPCPathsConfig = field(default_factory=PPCPathsConfig)
    simulation: PPCSimConfig = field(default_factory=PPCSimConfig)


@dataclass
class PPCCudaPathsConfig(ConfigBase):
    """ppc CUDA file path configuration."""

    location: str = field(default_factory=lambda: f"{RESOURCES_DIR}/PPC_executables/PPC_CUDA/")
    force: bool = False
    ppc_tmpdir: str = "./.ppc_tmp"
    ppc_tmpfile: str = ".event_hits.ppc.tmp"
    f2k_tmpfile: str = ".event_losses.f2k.tmp"
    ppc_prefix: str = ""
    f2k_prefix: str = ""
    ppctables: str = "../resources/PPC_tables/south_pole/"
    ppc_exe: str = "../resources/PPC_executables/PPC_CUDA/ppc"


@dataclass
class PPCCudaConfig(ConfigBase):
    """Top-level ppc CUDA photon propagator configuration."""

    paths: PPCCudaPathsConfig = field(default_factory=PPCCudaPathsConfig)
    simulation: PPCSimConfig = field(default_factory=PPCSimConfig)


@dataclass
class PhotonPropagatorConfig(ConfigBase):
    """Top-level photon propagator configuration."""

    name: Optional[str] = None
    photon_field_name: str = "photons"
    olympus: OlympusConfig = field(default_factory=OlympusConfig)
    ppc: PPCConfig = field(default_factory=PPCConfig)
    ppc_cuda: PPCCudaConfig = field(default_factory=PPCCudaConfig)

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "photon field name": "photon_field_name",
        "PPC": "ppc",
        "PPC_CUDA": "ppc_cuda",
    }


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


def _deep_apply(obj: ConfigBase, data: dict) -> None:
    """Recursively apply *data* dict values onto a *ConfigBase* tree in-place.

    Parameters
    ----------
    obj : ConfigBase
        Destination config object to update.
    data : dict
        Mapping of values to apply onto ``obj``.
    """
    for key, value in data.items():
        try:
            attr = obj._normalize(key)
        except Exception:
            raise KeyError(f"Unknown config key {key!r} in {type(obj).__name__}") from None
        if not hasattr(obj, attr):
            raise KeyError(
                f"Unknown config key {key!r} (normalised: {attr!r}) in {type(obj).__name__}"
            )
        current = getattr(obj, attr)
        if isinstance(current, ConfigBase) and isinstance(value, dict):
            _deep_apply(current, value)
        else:
            setattr(obj, attr, value)


@dataclass
class PrometheusConfig(ConfigBase):
    """Top-level Prometheus simulation configuration."""

    run: RunConfig = field(default_factory=RunConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    lepton_propagator: LeptonPropagatorConfig = field(default_factory=LeptonPropagatorConfig)
    photon_propagator: PhotonPropagatorConfig = field(default_factory=PhotonPropagatorConfig)

    _KEY_MAP: ClassVar[dict[str, str]] = {
        "lepton propagator": "lepton_propagator",
        "photon propagator": "photon_propagator",
    }

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def from_dict(self, user_dict: dict) -> None:
        """Apply *user_dict* over the current defaults in-place.

        Parameters
        ----------
        user_dict : dict
            Mapping to apply onto the current defaults.
        """
        _deep_apply(self, user_dict)

    def from_yaml(self, yaml_file: str) -> None:
        """Load a YAML file and apply it over the current defaults in-place.

        Parameters
        ----------
        yaml_file : str
            Path to a YAML file to load.
        """
        import yaml

        with open(yaml_file) as fh:
            data = yaml.load(fh, Loader=yaml.SafeLoader)
        if data:
            _deep_apply(self, data)

    def from_toml(self, toml_file: str) -> None:
        """Load a TOML file and apply it over the current defaults in-place.

        Parameters
        ----------
        toml_file : str
            Path to a TOML file to load.
        """
        import tomllib

        with open(toml_file, "rb") as fh:
            data = tomllib.load(fh)
        if data:
            _deep_apply(self, data)

    def to_dict(self) -> dict:
        """Return a plain nested dict (suitable for ``json.dumps``).

        Returns
        -------
        dict
            Nested dictionary representation of the config.
        """
        return asdict(self)

    def __repr__(self) -> str:
        return f"PrometheusConfig(run={self.run!r}, detector={self.detector!r}, ...)"
