from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="file",
        type=str, required=True,
        help="Which file do you want to parse?")
args = parser.parse_args()

filename = args.file

"""
This is an example converter script.
It takes in an HDF5 file produced by ICOS-LI and converts it into a file format that is compatible withe IceCube IceTray framework. 
"""


from I3Tray import *
from icecube import icetray, dataclasses
from icecube import dataio #infinite source of DAQ frames
from icecube.icetray import I3Units

import h5py as h5
import numpy as np


def make_i3_particle( particle_data ):
    """
    This takes one of the lists/tuples from the hdf5 datatable for a particle.
    It converts it into an I3Particle 
    """
    pt = dataclasses.I3Particle()
    pt.pdg_encoding = int(particle_data[1])
    pt.pos = dataclasses.I3Position( particle_data[2][0], particle_data[2][1], particle_data[2][2])
    pt.dir = dataclasses.I3Direction(particle_data[3][0], particle_data[3][1])
    pt.energy = particle_data[4]*I3Units.GeV
    return(pt)

def get_prop_dict(props, ranged):
    """
    This function takes the list/tuple of a an event's properties.
    It converts it into an I3MapstringDouble (basically a dictionary)
    """
    pdict = {}
    pdict['totalEnergy'] = props[0]*I3Units.GeV
    pdict['zenith'] = props[1]*I3Units.rad
    pdict['azimuth'] = props[2]*I3Units.rad
    pdict['finalStateX'] = props[3]
    pdict['finalStateY'] = props[4]
    pdict['finalType1'] = props[5]
    pdict['finalType2'] = props[6]
    pdict['initialType'] = props[7]
    if ranged:
        pdict['impactParameters']=props[8]
        pdict['totalColumnDepth']=props[9]
    else:
        pdict['radius'] = props[8]
        pdict['z'] = props[9]
        pdict['totalColumnDepth']=props[10]
    for key in pdict:
        pdict[key] = float(pdict[key])

    return(dataclasses.I3MapStringDouble(pdict))


class li_parser(icetray.I3Module):
    """
    Here we define a module which will parse the hdf5 file event by event
    """
    def __init__(self, context):
        """
        Upon creating this module, we load in the file.
        Each time a DAQ frame is passed to this function, it reads the next event.
        Once it reads the last event, it tells the tray to stop. 
        """
        icetray.I3Module.__init__(self,context)
        
        self.datafile = h5.File(filename, "r")
        self.keys = list(self.datafile.keys())

        self.keyno = 0
        self.event_no = 0

    def DAQ(self, frame):
        """
        Check if we need to move on to the next injector, and check then if we've reached the end of the file. 
        """
        while self.event_no>=len(self.datafile[self.keys[self.keyno]]['final_1']):
            self.keyno+=1
            self.event_no=0
            if self.keyno >= len(self.keys):
                print("Requesting Suspension")
                self.RequestSuspension()
                return
            else: 
                print("Moving to key {}".format(self.keys[self.keyno]))
        
        active = self.datafile[self.keys[self.keyno]]
        frame['I3MCTree'] = dataclasses.I3MCTree()
        dataclasses.I3MCTree.add_primary( frame['I3MCTree'], make_i3_particle( active['initial'][self.event_no]))
        dataclasses.I3MCTree.append_child( frame['I3MCTree'], frame['I3MCTree'][0].id, make_i3_particle( active['final_1'][self.event_no]))
        dataclasses.I3MCTree.append_child( frame['I3MCTree'], frame['I3MCTree'][0].id, make_i3_particle( active['final_2'][self.event_no]))
        
        frame['I3EventHeader'] = dataclasses.I3EventHeader()
        frame['EventProperties'] = get_prop_dict(active['properties'][self.event_no], 'Ranged' in self.keys[self.keyno])

        self.event_no += 1
        self.PushFrame(frame)

# we really only want to rename the file
outname = filename.split(".")[0] + ".i3.zst"

# prepare a tray, load in a DAQ source, and run our converter module 
tray = I3Tray()
tray.Add("I3InfiniteSource")
tray.Add(lambda frame: True, streams=[icetray.I3Frame.DAQ])
tray.AddModule(li_parser)
tray.AddModule("I3Writer", Filename=outname, Streams=[icetray.I3Frame.DAQ])
tray.Execute()
tray.Finish()
