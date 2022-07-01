# make icecube gen2 f2k

import numpy as np
import hebe.detector as dt

def xy_list(fname):
    """ Returns 2xn array of xy coordinates from web plot digitizer data
    """
    temp = []
    with open(fname) as dfile:
        for line in dfile.readlines():
            line = line.strip().split(',')
            line[0] = 1000*float(line[0])
            line[1] = 1000*float(line[1])
            temp.append(np.array([line[0],line[1]]))
        out = np.vstack([a for a in temp])
    return out

def make_ic2(fname):
    """
    Adds a line detector at each xy in fname to icecube
    Returns resultant detector
    """
    xylist = xy_list(fname)
    ic1_det = dt.detector_from_f2k('../data/icecube_clean-f2k')

    for xy in xylist:
        x = xy[0]; sy = xy[1]
        line_det = dt.make_line(x,y,120,16)
        ic1_det += line_det

    return ic1_det

make_ic2('../data/ic2_data')