# -*- coding: utf-8 -*-

from .config import config
from .detector import Detector, __init__
from .injection import __init__
from .lepton_propagation import __init__
from .particle import __init__
from .photon_propagation import __init__
from .prometheus import Prometheus
from .utils import __init__

__all__ = (Prometheus, config)
__version__ = "1.0.0"
