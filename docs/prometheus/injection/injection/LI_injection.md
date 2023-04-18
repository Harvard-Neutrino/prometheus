Module prometheus.injection.injection.LI_injection
==================================================

Functions
---------

    
`injection_event_from_LI(injection: h5py._hl.group.Group, idx: int) ‑> prometheus.injection.injection_event.LI_injection_event.LIInjectionEvent`
:   Create an injection event from LI h5 group and index
    
    params
    ______
    injection: Group from h5 file to make injection from
    idx: index in that gorup to make event
    
    returns
    _______
    event: Prometheus LIInjectionEvent corresponding to input

    
`injection_from_LI_output(LI_file: str) ‑> prometheus.injection.injection.LI_injection.LIInjection`
:   Creates injection object from a saved LI file

Classes
-------

`LIInjection(events: Iterable[prometheus.injection.injection_event.LI_injection_event.LIInjectionEvent])`
:   Base class for Prometheus injection
    
    params
    ______
    events: A list of injection events

    ### Ancestors (in MRO)

    * prometheus.injection.injection.injection.Injection