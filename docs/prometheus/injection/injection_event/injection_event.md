Module prometheus.injection.injection_event.injection_event
===========================================================

Classes
-------

`InjectionEvent(initial_state: prometheus.particle.particle.Particle, final_states: Iterable[prometheus.particle.particle.Particle], interaction: prometheus.injection.interactions.Interactions, vertex_x: float, vertex_y: float, vertex_z: float)`
:   Dataclass for handing injection events
    
    params
    ______
    initial_state: Incident neutrino
    final_states: All particles which results from the interaction
    vertex_x: x-position of the interaction vertex
    vertex_y: x-position of the interaction vertex
    vertex_z: x-position of the interaction vertex

    ### Descendants

    * prometheus.injection.injection_event.LI_injection_event.LIInjectionEvent

    ### Class variables

    `final_states: Iterable[prometheus.particle.particle.Particle]`
    :

    `initial_state: prometheus.particle.particle.Particle`
    :

    `interaction: prometheus.injection.interactions.Interactions`
    :

    `vertex_x: float`
    :

    `vertex_y: float`
    :

    `vertex_z: float`
    :