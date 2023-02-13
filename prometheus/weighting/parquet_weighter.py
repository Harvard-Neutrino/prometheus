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
        raise ValueError("This is busted sorry :-/")
        lw_event = LW.Event()
        injection = event["mc_truth"]
        lw_event.energy = injection["initial_state_energy"]
        lw_event.zenith = injection["initial_state_zenith"]
        lw_event.azimuth = injection["initial_state_azimuth"]
        lw_event.interaction_x = injection["bjorken_x"]
        lw_event.interaction_y = injection["bjorken_y"]
        initial_idxs = np.where(injection["final_state_parent"]==0)[0]
        final_state_1_idx = initial_idxs[0]
        final_state_2_idx = initial_idxs[1]
        lw_event.final_state_particle_0 = LW.ParticleType(injection["final_state_type"][final_state_1_idx])
        lw_event.final_state_particle_1 = LW.ParticleType(injection["final_state_type"][final_state_2_idx])
        lw_event.primary_type = LW.ParticleType(injection["initial_state_type"])
        lw_event.total_column_depth = injection["column_depth"]
        lw_event.x = injection["initial_state_x"]
        lw_event.y = injection["initial_state_y"]
        lw_event.z = injection["initial_state_z"]
        return self._weighter.get_oneweight(lw_event) * self.nevents

