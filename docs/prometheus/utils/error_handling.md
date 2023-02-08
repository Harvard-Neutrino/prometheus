Module prometheus.utils.error_handling
======================================

Classes
-------

`CannotLoadDetectorError(msg)`
:   Raised when detector not provided and cannot be determined from config

    ### Ancestors (in MRO)

    * prometheus.utils.error_handling.NoTraceBackWithLineNumber
    * builtins.Exception
    * builtins.BaseException

`InjectorNotImplementedError(msg)`
:   this type of injection has not been implemented

    ### Ancestors (in MRO)

    * prometheus.utils.error_handling.NoTraceBackWithLineNumber
    * builtins.Exception
    * builtins.BaseException

`NoInjectionError(msg)`
:   the injection has not been set

    ### Ancestors (in MRO)

    * prometheus.utils.error_handling.NoTraceBackWithLineNumber
    * builtins.Exception
    * builtins.BaseException

`NoTraceBackWithLineNumber(msg)`
:   Custom error messages for exceptions

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

    ### Descendants

    * prometheus.utils.error_handling.CannotLoadDetectorError
    * prometheus.utils.error_handling.InjectorNotImplementedError
    * prometheus.utils.error_handling.NoInjectionError
    * prometheus.utils.error_handling.UnknownInjectorError
    * prometheus.utils.error_handling.UnknownLeptonPropagatorError
    * prometheus.utils.error_handling.UnknownPhotonPropagatorError

`UnknownInjectorError(msg)`
:   the injector defined in config isn't supported

    ### Ancestors (in MRO)

    * prometheus.utils.error_handling.NoTraceBackWithLineNumber
    * builtins.Exception
    * builtins.BaseException

`UnknownLeptonPropagatorError(msg)`
:   the lepton propagator defined in the config is not supported

    ### Ancestors (in MRO)

    * prometheus.utils.error_handling.NoTraceBackWithLineNumber
    * builtins.Exception
    * builtins.BaseException

`UnknownPhotonPropagatorError(msg)`
:   the photon propagator defined in the config is not supported

    ### Ancestors (in MRO)

    * prometheus.utils.error_handling.NoTraceBackWithLineNumber
    * builtins.Exception
    * builtins.BaseException