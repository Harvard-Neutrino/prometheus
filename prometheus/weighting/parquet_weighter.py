import json
import numpy as np
import awkward as ak
import LeptonWeighter as LW
import pyarrow.parquet as pq

from typing import Optional

from .weighter import Weighter

WARNMSG = "It looks like the lic file provided does not match that in the Parquet file. You may want to check this."

class ParquetWeighter(Weighter):

    def __init__(
        self,
        parquet_file:str,
        nu_cc_xs: str = "dsdxdy_nu_CC_iso.fits",
        nubar_cc_xs: str = "dsdxdy_nubar_CC_iso.fits",
        nu_nc_xs: str = "dsdxdy_nu_NC_iso.fits",
        nubar_nc_xs: str = "dsdxdy_nubar_NC_iso.fits",
        xs_prefix: Optional[str] = None,
        lic_file: Optional[str] = None,
        offset: Optional[np.ndarray] = None
    ):

        config = json.loads(
            pq.read_metadata(parquet_file).metadata[b"config_prometheus"]
        )

        if config["injection"]["name"]!="LeptonInjector":
            raise ValueError("Weighting only available for LeptonInjector")

        exp_lic_file = config["injection"]["LeptonInjector"]["paths"]["lic file"]
        
        if (lic_file is not None) and (exp_lic_file!=lic_file):
            from warnings import warn
            warn(WARNMSG, UserWarning)

        elif lic_file is None:
            lic_file = exp_lic_file

        self._data = ak.from_parquet(parquet_file)
        print(config)
        if offset is None:
            try:
                self._offset = np.array(config["detector"]["offset"])
            except KeyError:
                from ..detector import detector_from_geo
                det = detector_from_geo(config["detector"]["geo file"])
                self._offset = np.array(det.offset)

        super().__init__(
            lic_file,
            xs_prefix=xs_prefix,
            nu_cc_xs=nu_cc_xs,
            nubar_cc_xs=nubar_cc_xs,
            nu_nc_xs=nu_nc_xs,
            nubar_nc_xs=nubar_nc_xs,
            nevents=len(self._data["mc_truth"])
        )

    def _get_event_oneweight(self, event:ak.Record) -> float:
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
        injection = event["mc_truth"]
        lw_event.energy = injection["initial_state_energy"]
        lw_event.zenith = injection["initial_state_zenith"]
        azimuth = injection["initial_state_azimuth"]
        lw_event.azimuth = azimuth if azimuth >= 0 else 2 * np.pi + azimuth
        lw_event.interaction_x = injection["bjorken_x"]
        lw_event.interaction_y = injection["bjorken_y"]
        initial_idxs = np.where(injection["final_state_parent"]==0)[0]
        final_state_1_idx = initial_idxs[0]
        final_state_2_idx = initial_idxs[1]
        lw_event.final_state_particle_0 = LW.ParticleType(injection["final_state_type"][final_state_1_idx])
        lw_event.final_state_particle_1 = LW.ParticleType(injection["final_state_type"][final_state_2_idx])
        lw_event.primary_type = LW.ParticleType(injection["initial_state_type"])
        lw_event.total_column_depth = injection["column_depth"]
        lw_event.x = injection["initial_state_x"] - self._offset[0]
        lw_event.y = injection["initial_state_y"] - self._offset[1]
        lw_event.z = injection["initial_state_z"] - self._offset[2]
        return self._weighter.get_oneweight(lw_event) * self.nevents / 1e4 # m

    def weight_events(self):
        return np.array([self._get_event_oneweight(x) for x in self._data])

