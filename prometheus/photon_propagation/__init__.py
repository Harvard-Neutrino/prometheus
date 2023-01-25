from .photon_propagator import PhotonPropagator
from .registered_propagators import (
    RegisteredPropagators as RegisteredPhotonPropagators
)

# We're doing this because there is a lot of import overhead associated with 
# importing Olympus
def get_photon_propagator(name: str) -> PhotonPropagator:
    name = name.lower()
    if name not in "ppc ppc_cuda olympus".split():
        raise ValueError("Unknown photon propagator")
    if name in "ppc ppc_cuda".split():
        from .ppc_photon_propagator import PPCPhotonPropagator
        return PPCPhotonPropagator
    else:
        from .olympus_photon_propagator import OlympusPhotonPropagator
        return OlympusPhotonPropagator
