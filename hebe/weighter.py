import awkward as ak
import h5py as h5
import LeptonWeighter as LW

from abc import abstractmethod

class CrossSectionFilesNotFoundError(Exception):

    def __init__(self, xs_dir):
        from glob import glob
        self.message = f"Can't find correct spline files in {xs_dir}.\n"
        self.message += "You may need change the xs files in the code to match yours.\n"
        self.message += "Sorry about it"
        super().__init__(self.message)

def xs_files_exist(xs_dir):
    from os.path import isfile
    return (
        isfile(f"{xs_dir}/dsdxdy-numu-N-cc-HERAPDF15NLO_EIG_central.fits") and
        isfile(f"{xs_dir}/dsdxdy-numubar-N-cc-HERAPDF15NLO_EIG_central.fits") and
        isfile(f"{xs_dir}/dsdxdy-numu-N-nc-HERAPDF15NLO_EIG_central.fits") and
        isfile(f"{xs_dir}/dsdxdy-numubar-N-nc-HERAPDF15NLO_EIG_central.fits")
    )

DEFAULT_XS = {
    "nu_cc_xs": "dsdxdy-numu-N-cc-HERAPDF15NLO_EIG_central.fits",
    "nubar_cc_xs": "dsdxdy-numubar-N-cc-HERAPDF15NLO_EIG_central.fits",
    "nu_nc_xs": "dsdxdy-numu-N-nc-HERAPDF15NLO_EIG_central.fits",
    "nubar_nc_xs": "dsdxdy-numubar-N-nc-HERAPDF15NLO_EIG_central.fits"
}

class Weighter:

    """
    Base class for weighting injection events with LeptonWeighter

    params
    ______
    xs_dir: Path to differential cross sections. This can usually be found in 
             `/LeptonWeighter/resources/data/`
    lic_file: Path to lic_file created by LeptonInjector
    nevents: (1) Events generated to rescale weight by. Helpful if you have non-uniform
             events per file. 
    """
    def __init__(
        self, 
        xs_dir, 
        lic_file,
        nevents=1,
        **kwargs
        ):
        #if not xs_files_exist(xs_dir):
        #    raise CrossSectionFilesNotFoundError(xs_dir)
        self._xs_dir = xs_dir
        self._lic_file = lic_file
        self._nevents = nevents

        for k, v in DEFAULT_XS.items():
            if k not in kwargs.keys():
                kwargs[k] = v
        print(f"{xs_dir}/{kwargs['nubar_nc_xs']}",)
            
        self._xs = LW.CrossSectionFromSpline(
            f"{xs_dir}/{kwargs['nu_cc_xs']}",
            f"{xs_dir}/{kwargs['nubar_cc_xs']}",
            f"{xs_dir}/{kwargs['nu_nc_xs']}",
            f"{xs_dir}/{kwargs['nubar_nc_xs']}",
        )

        self._generators = LW.MakeGeneratorsFromLICFile(lic_file)
        self._weighter = LW.Weighter(self._xs, self._generators)

    @property
    def xs_dir(self) -> str:
        return self._xs_dir

    @property
    def lic_file(self) -> str:
        return self._lic_file

    @property
    def nevents(self) -> int:
        return self._nevents

    @abstractmethod
    def get_event_oneweight(self, event) -> float:
        pass


class ParquetWeighter(Weighter):

    def __init__(self, xs_dir: str, lic_file: str, nevents: int=1):
        """
        Class for weighting parquet events with LeptonWeighter

        params
        ______
        xs_dir: Path to differential cross sections. This can usually be found in 
                 `/LeptonWeighter/resources/data/`
        lic_file: Path to lic_file created by LeptonInjector
        nevents: (1) Events generated to rescale weight by. Helpful if you have non-uniform
                 events per file. 
        """
        super().__init__(xs_dir, lic_file, nevents=nevents)

    def get_event_oneweight(self, event:ak.Record) -> float:
        """
        Function that returns oneweight for event. Oneweight * flux / n_gen_events = rate

        params
        ______
        event: Prometheus output event

        returns
        _______
        oneweight: Oneweight for event [GeV sr m^2]
        """
        lw_event = LW.Event()
        injection = event.mc_truth
        lw_event.energy = injection.injection_energy
        lw_event.zenith = injection.injection_zenith
        lw_event.azimuth = injection.injection_azimuth
        lw_event.interaction_x = injection.injection_bjorkenx
        lw_event.interaction_y = injection.injection_bjorkeny
        lw_event.final_state_particle_0 = LW.ParticleType(injection.primary_lepton_1_type)
        lw_event.final_state_particle_1 = LW.ParticleType(injection.primary_hadron_1_type)
        lw_event.primary_type = LW.ParticleType(injection.injection_type)
        lw_event.total_column_depth = injection.injection_column_depth
        lw_event.x = injection.injection_position_x
        lw_event.y = injection.injection_position_y
        lw_event.z = injection.injection_position_z
        return self._weighter.get_oneweight(lw_event) * self.nevents

class H5Weighter(Weighter):

    def __init__(self, xs_dir: str, lic_file: str, nevents: int=1, **kwargs):
        """
        Class for weighting h5 events with LeptonWeighter

        params
        ______
        xs_dir: Path to differential cross sections. This can usually be found in 
                 `/LeptonWeighter/resources/data/`
        lic_file: Path to lic_file created by LeptonInjector
        nevents: (1) Events generated to rescale weight by. Helpful if you have non-uniform
                 events per file. 
        """
        super().__init__(xs_dir, lic_file, nevents=nevents, **kwargs)

    def get_event_oneweight(self, event_properties:h5.Dataset) -> float:
        """
        Function that returns oneweight for event. Oneweight * flux / n_gen_events = rate

        params
        ______
        event: Prometheus output event

        returns
        _______
        oneweight: Oneweight for event [GeV sr m^2]
        """
        lw_event = LW.Event()
        lw_event.energy = event_properties["totalEnergy"]
        lw_event.zenith = event_properties["zenith"]
        lw_event.azimuth = event_properties["azimuth"]
        lw_event.interaction_x = event_properties["finalStateX"]
        lw_event.interaction_y = event_properties["finalStateY"]
        lw_event.final_state_particle_0 = LW.ParticleType(event_properties["finalType1"])
        lw_event.final_state_particle_1 = LW.ParticleType(event_properties["finalType2"])
        lw_event.primary_type = LW.ParticleType(event_properties["initialType"])
        lw_event.total_column_depth = event_properties["totalColumnDepth"]
        lw_event.x = event_properties["x"]
        lw_event.y = event_properties["y"]
        lw_event.z = event_properties["z"]
        return self._weighter.get_oneweight(lw_event) * self.nevents
