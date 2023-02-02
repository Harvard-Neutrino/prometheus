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
        lic_file,
        xs_prefix = "./",
        nu_cc_xs = "dsdxdy-numu-N-cc-HERAPDF15NLO_EIG_central.fits",
        nubar_cc_xs = "dsdxdy-numubar-N-cc-HERAPDF15NLO_EIG_central.fits",
        nu_nc_xs = "dsdxdy-numu-N-nc-HERAPDF15NLO_EIG_central.fits",
        nubar_nc_xs = "dsdxdy-numubar-N-nc-HERAPDF15NLO_EIG_central.fits",
        nevents=1,
    ):
        self._lic_file = lic_file
        self._nevents = nevents

        self._nu_cc_xs = f"{xs_prefix}/{nu_cc_xs}"
        self._nubar_cc_xs= f"{xs_prefix}/{nubar_cc_xs}"
        self._nu_nc_xs = f"{xs_prefix}/{nu_nc_xs}"
        self._nubar_nc_xs= f"{xs_prefix}/{nubar_nc_xs}"
            
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
