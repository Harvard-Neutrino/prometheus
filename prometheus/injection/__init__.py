from ..particle import Particle, PropagatableParticle
from .registered_injectors import RegisteredInjectors
from .injection import Injection, LIInjection, injection_from_LI_output
from .lepton_injector_utils import make_new_LI_injection

INJECTOR_DICT = {
    RegisteredInjectors.LEPTONINJECTOR: make_new_LI_injection
}

INJECTION_CONSTRUCTOR_DICT = {
    RegisteredInjectors.LEPTONINJECTOR: injection_from_LI_output
}
