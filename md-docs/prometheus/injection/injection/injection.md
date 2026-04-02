Module prometheus.injection.injection.injection
===============================================

Functions
---------

    
`recursive_getattr(x: Any, attr: str) ‑> Any`
:   Get an attribute that is farther down, e.g. 
    recursive_getattr(obj, "a.b")==getattr(getattr(obj, "a"), "b")
    
    params
    ______
    x: base object
    attr: period-delimited string of attributes to grab

    
`recursively_get_final_property(particles: Iterable[prometheus.particle.particle.Particle], attr: str, idx: Optional[None] = None) ‑> numpy.ndarray`
:   Helper for getting the attributes from particles. This is busted, sorry.
    
    params
    ______
    particles: Iterable with particles that you want to get the same attribute from
    attr: period-delimited string of attributes to grab
    idx: If the final attribute is an iterable, and you only want the value
        from a specific index, use this. This is useful for e.g. getting the 
        x-position from a 3-vector
    
    returns
    _______
    A numpy array with the requested attr for each particle. The shape of this
        array will be equal to the length of the input particles

Classes
-------

`Injection(events: Iterable[prometheus.injection.injection_event.injection_event.InjectionEvent])`
:   Base class for Prometheus injection
    
    params
    ______
    events: A list of injection events

    ### Descendants

    * prometheus.injection.injection.LI_injection.LIInjection

    ### Instance variables

    `events: List[prometheus.injection.injection_event.injection_event.InjectionEvent]`
    :

    ### Methods

    `to_awkward(self) ‑> awkward.highlevel.Array`
    :   Convert all the properties of the injection to an Awkward array

    `to_dict(self) ‑> dict`
    :   Convert all the properties of the injection to a dict