# -*- coding: utf-8 -*-
# hebe_ui.py
# Authors: Stephan Meighen-Berger
# Collection of custom errors for Prometheus

import sys
import inspect


class NoTraceBackWithLineNumber(Exception):
    """Custom error messages for exceptions"""
    def __init__(self, msg):
        try:

            ln = sys.exc_info()[-1].tb_lineno
        except AttributeError:
            ln = inspect.currentframe().f_back.f_lineno
            stack = inspect.stack()
            the_class = stack[1][0].f_locals["self"].__class__.__name__
            the_method = stack[1][0].f_code.co_name
            print("An error was raised in {}.{}()".format(the_class, the_method))
        self.args = "{0.__name__} (line {1}): {2}".format(type(self), ln, msg),
        sys.exit(self)


class UnknownInjectorError(NoTraceBackWithLineNumber):
    """ the injector defined in config isn't supported"""
    pass


class UnknownLeptonPropagatorError(NoTraceBackWithLineNumber):
    """ the lepton propagator defined in the config is not supported"""
    pass


class UnknownPhotonPropagatorError(NoTraceBackWithLineNumber):
    """ the photon propagator defined in the config is not supported"""
    pass


class NoInjectionError(NoTraceBackWithLineNumber):
    """ the injection has not been set"""
    pass


class InjectorNotImplementedError(NoTraceBackWithLineNumber):
    """ this type of injection has not been implemented"""
    pass


class CannotLoadDetectorError(NoTraceBackWithLineNumber):
    """Raised when detector not provided and cannot be determined from config"""
    pass
