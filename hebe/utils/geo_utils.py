# geo_utils.py
# Authors: David Kim
# Util functions for geo files

import numpy as np
from .f2k_utils import from_f2k
from .iter_or_rep import iter_or_rep

ice_padding = 200
water_padding = 30

def from_geo(fname):
    """Returns positions, keys, and medium from detector geometry file
    """
    pos = []; keys = []; meta_data = []
    with open(fname) as geo_in:
        read_lines = geo_in.readlines()
        meta_i = read_lines.index("### Metadata ###\n")
        modules_i = read_lines.index("### Modules ###\n")   

        for line in read_lines[meta_i+1:modules_i]:
            line = line.strip("\n").split("\t")
            meta_data.append(line[0]); meta_data.append(line[1])
        medium_i = meta_data.index("Medium:")
        medium = meta_data[medium_i+1]

        for line in read_lines[modules_i+1:]:
            line = line.strip("\n").split("\t")
            pos.append(
                np.array([float(line[0]), float(line[1]),
                float(line[2])]))
            pos_out = np.array(pos)
            keys.append((int(line[3]),int(line[4])))
    return pos_out, keys, medium

def geo_from_f2k(fname, out_path, medium = "ice", dom_radius = 30):
    """Generates a detector geo file from an f2k
    """
    positions, keys, sers = from_f2k(fname)
    with open(out_path, "w") as geo_out:
        geo_out.write(f'### Metadata ###\nMedium:\t{medium}\nDOM Radius [cm]:\t{dom_radius}\n### Modules ###\n')
        for pos, key in zip(positions,keys):
            geo_out.write(f'{pos[0]}\t{pos[1]}\t{pos[2]}\t{key[0]}\t{key[1]}\n')

geo_from_f2k("../data/deepcore-f2k","../data/deepcore-geo")
geo_from_f2k("../data/upgrade-f2k","../data/upgrade-geo")

def get_xyz(fname):
    # returns 3xn array
    det = from_geo(fname)
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

def get_volume(coords, is_ice = True):
    out_cyl = get_cylinder(coords)
    if is_ice:
        padding = ice_padding
    else:
        padding = water_padding
    return (out_cyl[0]+padding, out_cyl[1]+2*padding)

def get_endcap(coords, is_ice = True):
    if is_ice:
        padding = ice_padding 
    else:
        padding = water_padding
    cyl = get_cylinder(coords)
    r = cyl[0]; z = cyl[1]
    theta = (np.pi/2)-2*np.arctan(2*r/z)
    endcap_len = padding + np.cos(theta)*(get_injRadius(coords)-padding)
    return endcap_len

def get_injRadius(coords, is_ice = True):
    if is_ice:
        padding = ice_padding 
    else:
        padding = water_padding
    cyl = get_cylinder(coords)
    injRad = padding + (np.sqrt(cyl[0]**2+(0.5*cyl[1])**2))
    return injRad
