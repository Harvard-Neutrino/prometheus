################################################################################
# Module to find SuiteSparse                                                   #
#                                                                              #
# This module defines:                                                         #
#                                                                              #
#   SUITESPARSE_FOUND                                                          #
#   SUITESPARSE_VERSION                                                        #
#   SUITESPARSE_LIBRARIES                                                      #
#   SUITESPARSE_INCLUDE_DIR                                                    #
#   SUITESPARSE_LIB_DIR                                                        #
#   SUITESPARSE_CPPFLAGS                                                       #
#   SUITESPARSE_LDFLAGS                                                        #
################################################################################

SET (SUITESPARSE_FIND_QUIETLY TRUE)
SET (SUITESPARSE_FIND_REQUIRED FALSE)
SET (SUITESPARSE_REQUIRED_LIBS "camd" "ccolamd" "spqr" "cholmod" "amd" "colamd")

# Assume that that if TBB is present, SuiteSparse was linked against it,
# and we will also need to link against C++ libraries
INCLUDE (TBB)
IF (TBB_FOUND)
    ENABLE_LANGUAGE(CXX)
ENDIF (TBB_FOUND)
INCLUDE (METIS)

IF (NOT SUITESPARSE_FOUND)

  # Search user environment for headers, then default paths; extract version
  FIND_PATH (SUITESPARSE_INCLUDE_DIR cholmod.h
    PATHS $ENV{SUITESPARSEROOT}/include $ENV{SUITESPARSEROOT}/include/suitesparse
    NO_DEFAULT_PATH)
  FIND_PATH (SUITESPARSE_INCLUDE_DIR cholmod.h)
  IF( NOT SUITESPARSE_INCLUDE_DIR )
    FIND_PATH (SUITESPARSE_INCLUDE_DIR suitesparse/cholmod.h)
    IF (SUITESPARSE_INCLUDE_DIR)
      SET(SUITESPARSE_INCLUDE_DIR "${SUITESPARSE_INCLUDE_DIR}/suitesparse" CACHE PATH "SuiteSparse includes" FORCE)
    ENDIF (SUITESPARSE_INCLUDE_DIR)
  ENDIF( NOT SUITESPARSE_INCLUDE_DIR )
  GET_FILENAME_COMPONENT (SUITESPARSEROOT ${SUITESPARSE_INCLUDE_DIR} PATH)

  SET (SUITESPARSE_VERSION 0)
  IF (SUITESPARSE_INCLUDE_DIR)
    IF (EXISTS "${SUITESPARSE_INCLUDE_DIR}/SuiteSparse_config.h")
      FILE (READ "${SUITESPARSE_INCLUDE_DIR}/SuiteSparse_config.h" _SUITESPARSE_VERSION)
    ENDIF (EXISTS "${SUITESPARSE_INCLUDE_DIR}/SuiteSparse_config.h")
    IF (EXISTS "${SUITESPARSE_INCLUDE_DIR}/UFconfig.h")
      FILE (READ "${SUITESPARSE_INCLUDE_DIR}/UFconfig.h" _SUITESPARSE_VERSION)
    ENDIF (EXISTS "${SUITESPARSE_INCLUDE_DIR}/UFconfig.h")
    STRING (REGEX MATCH "define SUITESPARSE_MAIN_VERSION +([0-9]+)" _SS_M "${_SUITESPARSE_VERSION}")
    IF (_SS_M)
      STRING (REGEX REPLACE ".*([0-9]+)$" "\\1" SUITESPARSE_VERSION "${_SS_M}")
    ELSE ()
      SET (SUITESPARSE_VERSION 4)
    ENDIF ()
  ENDIF (SUITESPARSE_INCLUDE_DIR)
  
  IF (${SUITESPARSE_VERSION} GREATER 3)
    LIST (APPEND SUITESPARSE_REQUIRED_LIBS "suitesparseconfig")
  ENDIF (${SUITESPARSE_VERSION} GREATER 3)

  # Search user environment for libraries, then default paths
  SET (SUITESPARSE_LIBRARIES)
  FOREACH (_lib ${SUITESPARSE_REQUIRED_LIBS})
    FIND_LIBRARY (my_${_lib}
      NAMES ${_lib}
      PATHS $ENV{SUITESPARSEROOT}/lib
      NO_DEFAULT_PATH)
    FIND_LIBRARY (my_${_lib}
      NAMES ${_lib})
    LIST (APPEND SUITESPARSE_LIBRARIES ${my_${_lib}})
  ENDFOREACH ()

  # Set SUITESPARSE_FOUND and error out if required libraries are missing
  INCLUDE (FindPackageHandleStandardArgs)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS (SUITESPARSE
    DEFAULT_MSG SUITESPARSE_LIBRARIES SUITESPARSE_INCLUDE_DIR)
  ADD_DEFINITIONS ("-I${SUITESPARSE_INCLUDE_DIR}")

  IF (SUITESPARSE_FOUND)
    # Set flags and print a status message
    MESSAGE (STATUS "SuiteSparse version ${SUITESPARSE_VERSION} found:")

    SET (SUITESPARSE_CPPFLAGS "-I${SUITESPARSE_INCLUDE_DIR}")
    SET (SUITESPARSE_LDFLAGS "${SUITESPARSE_LIBRARIES}")
    
    MESSAGE (STATUS "  * includes: ${SUITESPARSE_INCLUDE_DIR}")
    MESSAGE (STATUS "  * libs:     ${SUITESPARSE_LIBRARIES}")
  ENDIF (SUITESPARSE_FOUND)

ENDIF (NOT SUITESPARSE_FOUND)

