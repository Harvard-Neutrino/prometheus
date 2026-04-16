# -*- coding: utf-8 -*-
# Name: config.py
# Copyright (C) 2022 Stephan Meighen-Berger
# Config file for the prometheus package.
#
# The singleton ``config`` is now a typed ``PrometheusConfig`` dataclass
# instance.  All old dict-style access (``config["run"]["nevents"]``) still
# works through the ``ConfigBase.__getitem__`` shim.
#
# Preferred new style:
#   config.run.nevents = 100
#   config.detector.geo_file = "resources/geofiles/demo_water.geo"
#
# Legacy style still works:
#   config["run"]["nevents"] = 100
#   config["detector"]["geo file"] = "resources/geofiles/demo_water.geo"

from .config_types import PrometheusConfig

config = PrometheusConfig()

# Keep the old name around in case external code imports it.
ConfigClass = PrometheusConfig

# Dummy _baseconfig — retained in case any external code imported it.
_baseconfig = None
