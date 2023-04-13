Module prometheus.weighting.weighter
====================================

Classes
-------

`Weighter(lic_file, xs_prefix='./', nu_cc_xs='dsdxdy-numu-N-cc-HERAPDF15NLO_EIG_central.fits', nubar_cc_xs='dsdxdy-numubar-N-cc-HERAPDF15NLO_EIG_central.fits', nu_nc_xs='dsdxdy-numu-N-nc-HERAPDF15NLO_EIG_central.fits', nubar_nc_xs='dsdxdy-numubar-N-nc-HERAPDF15NLO_EIG_central.fits', nevents=1)`
:   Base class for weighting injection events with LeptonWeighter
    
    params
    ______
    xs_dir: Path to differential cross sections. This can usually be found in 
             `/LeptonWeighter/resources/data/`
    lic_file: Path to lic_file created by LeptonInjector
    nevents: (1) Events generated to rescale weight by. Helpful if you have non-uniform
             events per file.

    ### Descendants

    * prometheus.weighting.h5_weighter.H5Weighter
    * prometheus.weighting.parquet_weighter.ParquetWeighter

    ### Instance variables

    `lic_file: str`
    :

    `nevents: int`
    :

    `nu_cc_xs: str`
    :

    `nu_nc_xs: str`
    :

    `nubar_cc_xs: str`
    :

    `nubar_nc_xs: str`
    :

    ### Methods

    `get_event_oneweight(self, event) ‑> float`
    :