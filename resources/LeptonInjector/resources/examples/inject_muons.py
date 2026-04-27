# Benjamin Smithers
# benjamin.smithers@mavs.uta.edu

# this example script ...
#   + imports the LeptonInjector libraries
#   + creates two injectors, and their operator
#   + tells the operator to execute the process

from math import pi
from pathlib import Path

import EarthModelService as em
import LeptonInjector as LI

# use this if you are on the Cobalt testbed
# xs_folder = "/cvmfs/icecube.opensciencegrid.org/data/neutrino-generator/cross_section_data/csms_differential_v1.0"

# for now, just use the example cross sections that come pre-installed.
# this looks in the parent folder to this file's containing folder
xs_folder = str(Path(__file__).resolve().parent.parent)

# Now, we'll make a new injector for muon tracks
n_events = 55000
diff_xs = xs_folder + "/test_xs.fits"
total_xs = xs_folder + "/test_xs_total.fits"
is_ranged = True
final_1 = LI.Particle.ParticleType.MuMinus
final_2 = LI.Particle.ParticleType.Hadrons
the_injector = LI.Injector(n_events, final_1, final_2, diff_xs, total_xs, is_ranged)


deg = pi / 180.0

# define some defaults
minE = 1000.0  # [GeV]
maxE = 100000.0  # [GeV]
gamma = 2.0
minZenith = 80.0 * deg
maxZenith = 180.0 * deg
minAzimuth = 0.0 * deg
maxAzimuth = 180.0 * deg

# construct the controller
controller = LI.Controller(
    the_injector, minE, maxE, gamma, minAzimuth, maxAzimuth, minZenith, maxZenith
)

# specify the output, earth model
earth_params = str(Path(__file__).resolve().parent.parent / "earthparams") + "/"
earth = em.EarthModelService(
    "earth",
    earth_params,
    ["PREM_mmc"],
    ["Standard"],
    "SimpleIceCap",
    20 * LI.Constants.degrees,
    3500 * LI.Constants.m,
)

controller.SetEarthModelService(earth)
controller.NameOutfile("./data_output.h5")
controller.NameLicFile("./config.lic")

# run the sim
controller.Execute()
