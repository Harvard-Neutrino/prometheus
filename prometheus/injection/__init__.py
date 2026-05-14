from dataclasses import dataclass
from typing import Callable

from ..particle import Particle, PropagatableParticle
from .injection import Injection, LIInjection, injection_from_LI_output
from .injection import GENIEInjection, injection_from_genie_output
from .lepton_injector_utils import make_new_LI_injection
from .genie_injector import make_new_genie_injection
from .registered_injectors import RegisteredInjectors
from .registry import get_injector, register_injector


@dataclass(frozen=True)
class InjectorPlugin:
    """Pairs an injector runner with its injection constructor.

    Parameters
    ----------
    runner : callable
        Runs or validates injection (e.g. invokes the LI binary or checks a
        GENIE ROOT file). Signature:
        ``runner(paths, simulation_config, *, detector_offset, detector, **kwargs)``.
    constructor : callable
        Reads injection data and returns an ``Injection`` object. Signature:
        ``constructor(injection_file, *, simulation_config, detector,
        detector_offset, **kwargs)``.
    """

    runner: Callable
    constructor: Callable


INJECTORS: dict[RegisteredInjectors, InjectorPlugin] = {
    RegisteredInjectors.LEPTONINJECTOR: InjectorPlugin(
        runner=make_new_LI_injection,
        constructor=injection_from_LI_output,
    ),
    RegisteredInjectors.GENIE: InjectorPlugin(
        runner=make_new_genie_injection,
        constructor=injection_from_genie_output,
    ),
}
