import awkward as ak
import LeptonWeighter as LW

class Weighter:

    def __init__(self, xs_path, lic_file):
        self._xs_path = xs_path
        self._lic_file = lic_file
        self._generators = LW.MakeGeneratorsFromLICFile(lic_file)
        self._xs = LW.CrossSectionFromSpline(
            f"{xs_path}/dsdxdy-numu-N-cc-HERAPDF15NLO_EIG_central.fits",
            f"{xs_path}/dsdxdy-numubar-N-cc-HERAPDF15NLO_EIG_central.fits",
            f"{xs_path}/dsdxdy-numu-N-nc-HERAPDF15NLO_EIG_central.fits",
            f"{xs_path}/dsdxdy-numubar-N-nc-HERAPDF15NLO_EIG_central.fits"
        )
        self._weighter = LW.Weighter(self._xs, self._generators)

    @property
    def xs_path(self):
        return self._xs_path

    @property
    def lic_file(self):
        return self._lic_file

    def get_event_oneweight(self, event:ak.Record, nevents:int=1):
        """
        Function that returns oneweight for event with option to specify how many events we in file.
        This is useful when you don't have the same number of events per file

        params
        ______
        event: Prometheus output event
        nevents: (1) Events generated to rescale weight by
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
        return self._weighter.get_oneweight(lw_event) * nevents
