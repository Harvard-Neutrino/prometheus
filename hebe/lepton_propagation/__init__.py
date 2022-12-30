from .loss import Loss
from .lepton_propagator import LeptonPropagator
from .old_proposal_lepton_propagator import OldProposalLeptonPropagator
from .new_proposal_lepton_propagator import NewProposalLeptonPropagator

LP_DICT = {
    "old proposal": OldProposalLeptonPropagator,
    "new proposal": NewProposalLeptonPropagator
}
