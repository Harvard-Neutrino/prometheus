import awkward as ak
import LeptonWeighter as LW

from abc import abstractmethod

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
        lic_file: str,
        xs_prefix: str = None,
        nu_cc_xs: str = "dsdxdy_nu_CC_iso.fits",
        nubar_cc_xs: str = "dsdxdy_nubar_CC_iso.fits",
        nu_nc_xs: str = "dsdxdy_nu_NC_iso.fits",
        nubar_nc_xs: str = "dsdxdy_nubar_NC_iso.fits",
        nevents: int = 1
    ):
        if xs_prefix is None:
            import os
            import prometheus
            path = os.path.dirname(prometheus.__file__)
            xs_prefix = os.path.abspath(f"{path}/../resources/cross_section_splines/")


        self._lic_file = lic_file
        self._nevents = nevents

        self._nu_cc_xs = f"{xs_prefix}/{nu_cc_xs}"
        self._nubar_cc_xs = f"{xs_prefix}/{nubar_cc_xs}"
        self._nu_nc_xs = f"{xs_prefix}/{nu_nc_xs}"
        self._nubar_nc_xs = f"{xs_prefix}/{nubar_nc_xs}"
            
        self._xs = LW.CrossSectionFromSpline(
            self.nu_cc_xs,
            self.nubar_cc_xs,
            self.nu_nc_xs,
            self.nubar_nc_xs,
        )

        self._generators = LW.MakeGeneratorsFromLICFile(lic_file)
        self._weighter = LW.Weighter(self._xs, self._generators)

    @property
    def nu_cc_xs(self) -> str:
        return self._nu_cc_xs

    @property
    def nubar_cc_xs(self) -> str:
        return self._nubar_cc_xs

    @property
    def nu_nc_xs(self) -> str:
        return self._nu_nc_xs

    @property
    def nubar_nc_xs(self) -> str:
        return self._nubar_nc_xs

    @property
    def lic_file(self) -> str:
        return self._lic_file

    @property
    def nevents(self) -> int:
        return self._nevents

    @abstractmethod
    def get_event_oneweight(self, event) -> float:
        pass
