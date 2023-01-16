from .loss import Loss
from .lepton_propagator import LeptonPropagator

import proposal as pp
if int(pp.__version__.split(".")[0]) <= 6:
    print("Using an older version of PROPOSAL")
    from .old_proposal_lepton_propagator import OldProposalLeptonPropagator
else:
    print("Using a newer version of PROPOSAL")
    from .new_proposal_lepton_propagator import NewProposalLeptonPropagator
