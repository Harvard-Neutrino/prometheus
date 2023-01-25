# f2k_utils.py
# Authors: David Kim
# Util functions for f2k files

import numpy as np

padding = 200

def from_f2k(fname):
     # List for the (x,y,z) positions of each OM
    pos = []
    # List for the optical module keys
    keys = []
    # List for the serial numbers
    sers = []
    with open(fname) as f2k_in:
        for line in f2k_in.readlines():
            line = line.strip("\n").split("\t")
            sers.append(line[1])
            pos.append(
                np.array([float(line[2]), float(line[3]),
                float(line[4])]))
            keys.append((int(line[5]), int(line[6])))
        pos_out = np.vstack([a for a in pos])
    return pos_out, keys, sers

def clean_icecube(fname):
    ''' Removes IceTop modules from icecube-f2k
    '''
    with open(fname,'r') as f2k_in:
        line_list = f2k_in.readlines()
    with open(fname,'w') as f2k_in:
        count = 1
        for line in line_list[:2]:
            f2k_in.write(line)
        for line in line_list[2:]:
            if count >= 65:
                count = 1
            if count not in list(range(61,66)):
                f2k_in.write(line)
            count += 1

#clean_icecube('../data/copy-f2k')

def get_xyz(fname):
    # returns 3xn array
    det = from_f2k(fname)
    return det[0]

def offset(coords):
    # returns x st x+mean(x,y,z) = <o,o,o>
    xyz_avg = np.average(coords,0)
    return -1*xyz_avg

def get_cylinder(coords, epsilon = 5):
    # returns cylinder (radius, height) from 3xn array
    if max(np.average(coords,0)) > epsilon:
        coords = coords+offset(coords)
    
    out_cylinder = (
            np.linalg.norm(coords[:, :2], axis=1).max(),
            np.ptp(coords[:,2:])
        )
    
    return out_cylinder
    
def get_endcap(coords):
    cyl = get_cylinder(coords)
    r = cyl[0]; z = cyl[1]
    theta = (np.pi/2)-2*np.arctan(2*r/z)
    endcap_len = padding + np.cos(theta)*(get_injRadius(coords)-padding)
    return endcap_len

def get_injRadius(coords):
    cyl = get_cylinder(coords)
    injRad = padding + (np.sqrt(cyl[0]**2+(0.5*cyl[1])**2))
    return injRad
