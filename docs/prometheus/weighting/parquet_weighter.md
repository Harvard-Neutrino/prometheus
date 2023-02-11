Module prometheus.weighting.parquet_weighter
============================================

Classes
-------

`ParquetWeighter(lic_file, xs_prefix='./', nu_cc_xs='dsdxdy-numu-N-cc-HERAPDF15NLO_EIG_central.fits', nubar_cc_xs='dsdxdy-numubar-N-cc-HERAPDF15NLO_EIG_central.fits', nu_nc_xs='dsdxdy-numu-N-nc-HERAPDF15NLO_EIG_central.fits', nubar_nc_xs='dsdxdy-numubar-N-nc-HERAPDF15NLO_EIG_central.fits', nevents=1)`
:   Base class for weighting injection events with LeptonWeighter
    
    params
    ______
    xs_dir: Path to differential cross sections. This can usually be found in 
             `/LeptonWeighter/resources/data/`
    lic_file: Path to lic_file created by LeptonInjector
    nevents: (1) Events generated to rescale weight by. Helpful if you have non-uniform
             events per file.

    ### Ancestors (in MRO)

    * prometheus.weighting.weighter.Weighter

    ### Methods

    `get_event_oneweight(self, event: awkward.highlevel.Record) ‑> float`
    :   Function that returns oneweight for event. Oneweight * flux / n_gen_events = rate
        
        params
        ______
        event: Prometheus output event
        
        returns
        _______
        oneweight: Oneweight for event [GeV sr m^2]