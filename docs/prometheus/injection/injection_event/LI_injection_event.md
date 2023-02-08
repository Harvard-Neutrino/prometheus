Module prometheus.injection.injection_event.LI_injection_event
==============================================================

Classes
-------

`LIInjectionEvent(initial_state: prometheus.particle.particle.Particle, final_states: Iterable[prometheus.particle.particle.Particle], interaction: prometheus.injection.interactions.Interactions, vertex_x: float, vertex_y: float, vertex_z: float, bjorken_x: float, bjorken_y: float, column_depth: float)`
:   Injection event for LeptonInjector injection
    
    params
    ______
    bjorken_x: Bjorken-x variable of the interaction, i.e. fraction
        of the incident neutrino momentum imparted to the hadronic
        shower
    bjorken_y: Bjorken-y variable of the interaction, i.e. fraction
        of the incident neutrino energy imparted to the hadronic
        shower
    column_depth: column depth traversed by the neutrino before
        interacting in M.W.E.

    ### Ancestors (in MRO)

    * prometheus.injection.injection_event.injection_event.InjectionEvent

    ### Class variables

    `bjorken_x: float`
    :

    `bjorken_y: float`
    :

    `column_depth: float`
    :