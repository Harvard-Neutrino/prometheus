Module prometheus.lepton_propagation.lepton_propagator
======================================================

Classes
-------

`LeptonPropagator(config)`
:   Interface class to the different lepton propagators

    ### Descendants

    * prometheus.lepton_propagation.new_proposal_lepton_propagator.NewProposalLeptonPropagator
    * prometheus.lepton_propagation.old_proposal_lepton_propagator.OldProposalLeptonPropagator

    ### Instance variables

    `config: dict`
    :   Get the configuration dictionary used to make this

    ### Methods

    `energy_losses(self, particle: prometheus.particle.particle.Particle) ‑> None`
    :   Propagates particle with energy losses. The losses will be
            stored in `particle.losses`
        
        particle: Prometheus particle to propagate