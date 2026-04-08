from .loss import Loss
from .lepton_propagator import LeptonPropagator

# NewProposalLeptonPropagator is intentionally not imported here.
# It imports `proposal` at module level, so importing it eagerly would cause
# every `import prometheus` to trigger a full PROPOSAL import.
# Use: from prometheus.lepton_propagation.new_proposal_lepton_propagator import NewProposalLeptonPropagator
