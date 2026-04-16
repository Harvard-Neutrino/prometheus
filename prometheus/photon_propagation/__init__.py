from .photon_propagator import PhotonPropagator
from .registry import get_propagator, register_propagator
from .registered_propagators import (
    RegisteredPropagators as RegisteredPhotonPropagators
)

# Legacy alias — prefer get_propagator() for new code.
get_photon_propagator = get_propagator
