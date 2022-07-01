# make icecube gen2 f2k

import numpy as np
import hebe.detector as dt
import f2k_utils as fk

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
        x = xy[0]; y = xy[1]
        line_mods = dt.make_line(x,y,80,16,1,1,1)
        line_det = dt.Detector(line_mods)
        ic1_det += line_det

    return ic1_det

def center_ic(fname):
    det = dt.detector_from_f2k(fname)
    offset = fk.offset(fk.get_xyz(fname))
    det.module_coords += offset
    return det

#centered_det = center_ic('../data/icecube_clean-f2k')
#centered_det.to_f2k('config_settings')
ic2 = make_ic2('../data/ic2_data')
ic2.to_f2k('../data/icecube_gen2-f2k')