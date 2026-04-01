cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)
cmake_policy(VERSION 3.3.2)


#TODO
# + add support for Apple systems (needs an apple part??)
# + make sure it correctly builds on the cobalt testbed
# + verify python build.

## project(lepton_injector VERSION 1.0.0 DESCRIPTION "lepton injector library")
project(lepton_injector VERSION 1.0.0 LANGUAGES CXX C)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_C_STANDARD 99)
## enable_language(cxx)
SET (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
                       ${CMAKE_SOURCE_DIR}/cmake/Packages)

INCLUDE (SuiteSparse)
INCLUDE (Python)

set(CMAKE_CXX_FLAGS_DEBUG  "-g -O0")

if(CMAKE_VERSION VERSION_LESS 3.12.0)
    if(Python_ROOT_DIR)
        MESSAGE(WARNING "Python_ROOT_DIR is set but will be ignored by this version of CMake; set PYTHON_EXECUTABLE instead")
	endif()
	find_package(PythonInterp)
	find_package(PythonLibs)
	set(PYTHON_FOUND PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
else()
    if(PYTHON_EXECUTABLE)
        MESSAGE(WARNING "PYTHON_EXECUTABLE is set but will be ignored by this version of CMake; set Python_ROOT_DIR instead")
	endif()
    find_package(Python COMPONENTS Interpreter Development)
    set(PYTHON_FOUND Python_Interpreter_FOUND AND Python_Development_FOUND)
    set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
	set(PYTHON_INCLUDE_DIRS "${Python_INCLUDE_DIRS}")
	set(PYTHON_LIBRARIES "${Python_LIBRARIES}")
	set(PYTHON_VERSION_MAJOR "${Python_VERSION_MAJOR}")
	set(PYTHON_VERSION_MINOR "${Python_VERSION_MINOR}")
endif()
if(PYTHON_FOUND)
    MESSAGE("-- Found python at ${PYTHON_INCLUDE_DIRS}")
    MESSAGE("-- Will build python!")
else()
	MESSAGE(WARNING "!! Python Not Found !!")
    MESSAGE("!! Skipping Python Module !!")
endif()
if(DEFINED ENV{SROOT})
    set(HDF5_ROOT ${SROOT})
endif()
find_package(HDF5 REQUIRED)
if(HDF5_FOUND)
    MESSAGE("-- Found hdf5 at ${HDF5_INCLUDE_DIR}")
    include_directories(${HDF5_INCLUDE_DIR})
    set(_hdf5_libs hdf5 hdf5_cpp)
endif()

# find boost
find_package(Boost REQUIRED)
if(Boost_FOUND)
    MESSAGE("-- Found Boost at ${Boost_INCLUDE_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    MESSAGE(FATAL_ERROR "Unable to find boost")
endif()

find_package(photospline REQUIRED)

if(photospline_FOUND)
    MESSAGE("-- photospline found")
endif()

include(GNUInstallDirs)

# need to write 
# some apple thing

#core files 
LIST (APPEND core_SOURCES
    ${CMAKE_SOURCE_DIR}/private/LeptonInjector/BasicInjectionConfiguration.cxx
    ${CMAKE_SOURCE_DIR}/private/LeptonInjector/LeptonInjector.cxx
    ${CMAKE_SOURCE_DIR}/private/LeptonInjector/Controller.cxx
    ${CMAKE_SOURCE_DIR}/private/LeptonInjector/Coordinates.cxx
    ${CMAKE_SOURCE_DIR}/private/LeptonInjector/DataWriter.cxx
    ${CMAKE_SOURCE_DIR}/private/LeptonInjector/EventProps.cxx
    ${CMAKE_SOURCE_DIR}/private/LeptonInjector/Particle.cxx
    ${CMAKE_SOURCE_DIR}/private/LeptonInjector/Random.cxx
    ${CMAKE_SOURCE_DIR}/private/earthmodel-service/EarthModelCalculator.cxx
    ${CMAKE_SOURCE_DIR}/private/earthmodel-service/EarthModelService.cxx
    ${CMAKE_SOURCE_DIR}/private/phys-services/LICrossSection.cxx
)


add_library( lepton_injector SHARED ${core_SOURCES} )
target_include_directories(lepton_injector PUBLIC
    ${CMAKE_SOURCE_DIR}/public/
)

set_target_properties(lepton_injector PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME LeptonInjector)

MESSAGE("")
MESSAGE("Linking ...")
MESSAGE("    hdf5 ${HDF5_LIBRARIES}")
MESSAGE("    boost ${Boost_LIBRARIES}")
target_link_libraries(lepton_injector ${HDF5_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(lepton_injector photospline)
MESSAGE("")

install(TARGETS lepton_injector
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY "${CMAKE_SOURCE_DIR}/public/"
    DESTINATION "include"
    FILES_MATCHING PATTERN "*.h")

# write and install pkg-config file
file(WRITE "LeptonInjector.pc" "prefix=${CMAKE_INSTALL_PREFIX}\n")
file(APPEND "LeptonInjector.pc" "libdir=${CMAKE_INSTALL_PREFIX}/lib\n")
file(APPEND "LeptonInjector.pc" "includedir=${CMAKE_INSTALL_PREFIX}/include\n")
file(APPEND "LeptonInjector.pc" "Name: LeptonInjector\n")
file(APPEND "LeptonInjector.pc" "Description: Generates neutrino events for large volume cherenkov telescopes\n")
file(APPEND "LeptonInjector.pc" "URL: https://github.com/IceCubeOpenSource/LeptonInjector\n")
file(APPEND "LeptonInjector.pc" "Version: 1.0.0\n")
file(APPEND "LeptonInjector.pc" "Requires: photospline\n")
file(APPEND "LeptonInjector.pc" "Libs: -L${CMAKE_INSTALL_PREFIX}/lib\n")
file(APPEND "LeptonInjector.pc" "Cflags: -I${CMAKE_INSTALL_PREFIX}/include\n")
  
install(FILES "LeptonInjector.pc" DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig")

# build and install pyhton module
if(DEFINED ENV{SROOT})
    set(Python_ROOT_DIR ${SROOT})
endif()
find_package(PythonInterp)
find_package(PythonLibs)
set(PYTHON_FOUND PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
find_package(Boost REQUIRED COMPONENTS python)
if(PYTHON_FOUND)
    MESSAGE("-- Found python at ${PYTHON_INCLUDE_DIR}")
    MESSAGE("-- Will build python!")
else()
    MESSAGE("!! Python Not Found !!")
    MESSAGE("!! Skipping Python Module !!")
endif()



IF(PYTHON_FOUND)
    if(${PYTHON_VERSION_MAJOR} EQUAL 3)
        MESSAGE("-- Searching for python3 compatible boost_python")
        find_package(Boost COMPONENTS python3 QUIET)
        if(NOT Boost_FOUND)
            # Maybe it isn't labeled as python3? Search without a version and hope for the best
	    find_package(Boost COMPONENTS python QUIET)
        endif()
    else()
        MESSAGE("-- Searching for python2 compatible boost_python")
	find_package(Boost COMPONENTS python QUIET)
    endif()

    LIST( APPEND python_private
        ${CMAKE_SOURCE_DIR}/private/pybindings/LeptonInjector.cxx
    )

    add_library( LeptonInjector SHARED ${python_private})
    set_target_properties( LeptonInjector PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME "LeptonInjector" PREFIX "" SUFFIX ".so")
    target_include_directories( LeptonInjector PRIVATE ${PYTHON_INCLUDE_DIR})
    target_link_libraries(LeptonInjector PRIVATE lepton_injector ${Boost_LIBRARIES} ${Python_LIBRARIES})
    include_directories(${Boost_INCLUDE_DIRS})
    
    EXECUTE_PROCESS( COMMAND ${PYTHON_EXECUTABLE} -c
        "from distutils.sysconfig import get_python_lib; print(get_python_lib(prefix=''))"
        OUTPUT_VARIABLE PYTHON_MODULE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PYTHON_MODULE_DIR)
        INSTALL(TARGETS LeptonInjector LIBRARY DESTINATION ${PYTHON_MODULE_DIR})
    elseif(Python_SITELIB)
        INSTALL(TARGETS LeptonInjector LIBRARY DESTINATION ${Python_SITELIB})
    endif()

    LIST( APPEND python_earth_private
        ${CMAKE_SOURCE_DIR}/private/pybindings/earthmodel-service.cxx
    )

    add_library( EarthModelService SHARED ${python_earth_private})
    set_target_properties( EarthModelService PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME "EarthModelService" PREFIX "" SUFFIX ".so")
    target_include_directories( EarthModelService PRIVATE ${PYTHON_INCLUDE_DIR})
    target_link_libraries(EarthModelService PRIVATE lepton_injector ${Boost_LIBRARIES} ${Python_LIBRARIES})
    include_directories(${Boost_INCLUDE_DIRS})
    
    EXECUTE_PROCESS( COMMAND ${PYTHON_EXECUTABLE} -c
        "from distutils.sysconfig import get_python_lib; print(get_python_lib(prefix=''))"
        OUTPUT_VARIABLE PYTHON_MODULE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PYTHON_MODULE_DIR)
        INSTALL(TARGETS EarthModelService LIBRARY DESTINATION ${PYTHON_MODULE_DIR})
    elseif(Python_SITELIB)
        INSTALL(TARGETS EarthModelService LIBRARY DESTINATION ${Python_SITELIB})
    endif()

ELSE(NOT PYTHON FOUND)
    MESSAGE("-- no python. skipping")
ENDIF(PYTHON_FOUND)


MESSAGE("")
MESSAGE("Done!")
MESSAGE("Run `make` then `make install`")
