# example_detector_construction.py
# Authors: Stephan Meighen-Berger
# Shows how to generate a simple detector
# imports
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

# Here we are generating evenly spaced lines in a circle, e.g. KM3NeT
def sunflower(n, alpha=0, geodesic=False):
    points = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    angle_stride = 360 * phi if geodesic else 2 * np.pi / phi ** 2
    b = round(alpha * np.sqrt(n))  # number of boundary points
    for k in range(1, n + 1):
        r = radius(k, n, b)
        theta = k * angle_stride
        points.append((r * np.cos(theta), r * np.sin(theta)))
    return np.array(points)

def radius(k, n, b):
    if k > n - b:
        return 1.0
    else:
        return np.sqrt(k - 0.5) / np.sqrt(n - (b + 1) / 2)

if __name__ == "__main__":
    # Generating strings
    points = sunflower(75, alpha=0.2, geodesic=False) * 100.
    # Generating modules per line
    # So inefficient
    nz_list = []
    dist_z_list = []
    for line in points:
        nz_list.append(22)
        dist_z_list.append(9.09)
    # Combined array
    detector_specs = np.array([
        points[:, 0],
        points[:, 1],
        nz_list,
        dist_z_list
    ])
    np.savetxt('../prometheus/data/custom.txt', detector_specs)
