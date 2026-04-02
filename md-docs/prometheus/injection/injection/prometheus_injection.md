Module prometheus.injection.injection.prometheus_injection
==========================================================

Functions
---------

    
`injection_from_prometheus(file: str, injection_cls: prometheus.injection.injection.injection.Injection = prometheus.injection.injection.LI_injection.LIInjection, event_converter: Callable = <function prometheus_inj_to_li_injection_event>)`
:   Make an Injection object from Prometheus output. If not using
    output that was not generated from LeptonInjector, you will need
    to tell this which kind of injection you want to use, and how to
    convert each item to the appropriate InjectionEvent

    
`prometheus_inj_to_li_injection_event(truth: awkward.highlevel.Record) ‑> prometheus.injection.injection_event.LI_injection_event.LIInjectionEvent`
: