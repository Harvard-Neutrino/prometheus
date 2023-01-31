from .loss import Loss
from .lepton_propagator import LeptonPropagator
from .registered_propagators import RegisteredPropagators as RegisteredLeptonPropagators

import proposal as pp
if int(pp.__version__.split(".")[0]) <= 6:
    from .old_proposal_lepton_propagator import OldProposalLeptonPropagator
else:
    from .new_proposal_lepton_propagator import NewProposalLeptonPropagator
