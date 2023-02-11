import h5py as h5
import LeptonWeighter as LW

from .weighter import Weighter

class H5Weighter(Weighter):

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

