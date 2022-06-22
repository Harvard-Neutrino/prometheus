# f2k_utils
# David Kim

import numpy as np

padding = 150

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
            #keys.append(tuple(int(line[5:])))
        pos_out = np.vstack([a for a in pos])
    return pos_out, keys, sers

def get_xyz(fname):
    # return 3xn array
    det = from_f2k(fname)
    return det[0]

def offset(array):
    # return x st x+mean(x,y,z) = <o,o,o>
    xyz_avg = np.average(array,0)
    return -1*xyz_avg

def get_cylinder(array):
    # return cylinder (radius, height)
    if max(np.average(array,0)) > 5:
        array = array+offset(array)
    
    out_cylinder = (
            np.linalg.norm(array[:, :2], axis=1).max(),
            np.ptp(array[:,2:]),
        )
    
    return out_cylinder
    
def get_endcap(array):
    # return endcap len
    cyl = get_cylinder(array)
    r = cyl[0]; z = cyl[1]
    theta = (np.pi/2)-2*np.arctan(2*r/z)
    endcap_len = np.cos(theta)*(get_injRadius(array)-padding)
    return endcap_len

def get_injRadius(array):
    cyl = get_cylinder(array)
    injRad = 0.5*(np.sqrt(cyl[0]**2+cyl[1]**2))+padding
    return injRad

print('success!')
