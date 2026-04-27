################################################################################
# Module to find METIS                                                         #
#                                                                              #
# This module defines:                                                         #
#                                                                              #
#   METIS_FOUND                                                                #
#   METIS_LIBRARIES                                                            #
#   METIS_LIB_DIR                                                              #
#   METIS_CPPFLAGS                                                             #
#   METIS_LDFLAGS                                                              #
################################################################################

SET (METIS_FIND_QUIETLY TRUE)
SET (METIS_FIND_REQUIRED FALSE)
SET (METIS_REQUIRED_LIBS "metis")

IF (NOT METIS_FOUND)

  # Search user environment for libraries, then default paths
  SET (METIS_LIBRARIES)
  FOREACH (_lib ${METIS_REQUIRED_LIBS})
    FIND_LIBRARY (my_${_lib}
      NAMES ${_lib}
      PATHS $ENV{METISROOT}/lib
      NO_DEFAULT_PATH)
    FIND_LIBRARY (my_${_lib}
      NAMES ${_lib})
    LIST (APPEND METIS_LIBRARIES ${my_${_lib}})
  ENDFOREACH ()

  # Set METIS_FOUND and error out if required libraries are missing
  INCLUDE (FindPackageHandleStandardArgs)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS (METIS
    DEFAULT_MSG METIS_LIBRARIES)

  IF (METIS_FOUND)
    # Set flags and print a status message
    MESSAGE (STATUS "METIS found:")

    SET (METIS_LDFLAGS "${METIS_LIBRARIES}")
    
    MESSAGE (STATUS "  * libs:     ${METIS_LIBRARIES}")
  ENDIF (METIS_FOUND)

ENDIF (NOT METIS_FOUND)

