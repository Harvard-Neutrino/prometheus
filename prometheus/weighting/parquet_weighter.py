import awkward as ak
import LeptonWeighter as LW

from .weighter import Weighter

class ParquetWeighter(Weighter):

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

