from typing import List, Optional, Tuple

import numpy as np

#: Default IceCube DOM radius [m], matches PPC's OMR constant.
_OMR = 0.16510

#: Valid output modes understood by the serializer.
_VALID_OUTPUT_MODES = ("minimal", "standard", "extended")


class Module:
    """Detector optical module."""

    def __init__(
        self,
        pos: np.ndarray,
        key: Tuple[int, int],
        noise_rate: int = 1e3,
        efficiency: float = 0.2,
        serial_no: Optional[str] = None,
        module_type: int = -1,
        Rr: float = _OMR,
        Rz: float = _OMR,
        beta: float = 0.49,
        area: float = 1.0,
        n_pmts: int = 1,
        pmt_dirs: Optional[List[Tuple[float, float]]] = None,
        cable_azimuth: Optional[float] = None,
    ):
        """Initialize a module.

        Parameters
        ----------
        pos : np.ndarray
            Position of the optical module in meters.
        key : tuple of int
            Tuple to look up module by. (string index, om index) is the
            convention.
        noise_rate : int or float, optional
            Noise of the module in GHz.
        efficiency : float, optional
            Quantum efficiency of module.
        serial_no : str or None, optional
            Serial number for the OM.
        module_type : int, optional
            PPC om.conf type ID. ``-1`` means legacy angular sensitivity
            from ``as.dat``; any other value references an entry that will be
            written to ``om.conf``.
        Rr : float, optional
            Horizontal semi-axis of the module [m].
        Rz : float, optional
            Vertical semi-axis of the module [m]. Negative values indicate a
            cylindrical geometry with radius ``Rr`` and half-height ``|Rz|``.
        beta : float, optional
            PMT angular sensitivity shape parameter (see PPC documentation).
        area : float, optional
            Overall efficiency scaling factor written to om.conf.
        n_pmts : int, optional
            Number of PMTs within this module.
        pmt_dirs : list of (float, float) or None, optional
            Pointing direction of each PMT as ``(zenith_deg, azimuth_deg)``.
            Defaults to ``[(180.0, 0.0)]`` (single downward-facing PMT).
        cable_azimuth : float or None, optional
            Azimuthal direction to the cable in the module frame [deg].
            Omitted from om.conf when ``None``.

        Raises
        ------
        ValueError
            If ``len(pmt_dirs) != n_pmts``.
        """
        if pmt_dirs is None:
            pmt_dirs = [(180.0, 0.0)]

        if len(pmt_dirs) != n_pmts:
            raise ValueError(f"len(pmt_dirs)={len(pmt_dirs)} does not match n_pmts={n_pmts}")

        self.pos = pos
        self.noise_rate = noise_rate
        self.efficiency = efficiency
        self.key = key
        self.serial_no = serial_no
        self.module_type = module_type
        self.Rr = Rr
        self.Rz = Rz
        self.beta = beta
        self.area = area
        self.n_pmts = n_pmts
        self.pmt_dirs = pmt_dirs
        self.cable_azimuth = cable_azimuth

    def __repr__(self):
        """Return string representation."""
        return repr(
            f"Module {self.key}, {self.pos} [m], {self.noise_rate * 1e-9} [Hz], {self.efficiency}"
        )
