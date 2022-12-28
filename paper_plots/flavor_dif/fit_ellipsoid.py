import numpy as np

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        dest="infile",
        type=str,
        required=True
    )
    parser.add_argument(
        "-o",
        dest="outfile",
        type=str,
        required=True
    )
    args = parser.parse_args()
    return args

def cost(pts, rotation, ellipsoid):
    pts = pts.transpose() - np.mean(pts, axis=1)
    pts = np.matmul(rotation.as_matrix(), pts.transpose())
    norms = np.linalg.norm(pts, axis=0)
    thetas = np.arccos(pts[2] / norms)
    rs = ellipsoid.r(thetas)
    return np.mean(np.abs(np.unique(rs - norms)))

def f(params, pts, return_objs=False):
    from ellipsoid import Ellipsoid
    from scipy.spatial.transform import Rotation as R
    a, b, theta, phi, psi = params
    rotation = R.from_euler("zyx", [theta, phi, psi])
    ellipsoid = Ellipsoid(a, b)
    if return_objs:
        return cost(pts, rotation, ellipsoid), rotation, ellipsoid
    else:
        return cost(pts, rotation, ellipsoid)

def fit_ellipsoid(event, nhit=8):
    if len(event.total.t) <= nhit:
        return None
    else:
        from scipy.optimize import differential_evolution
        pts = np.array([
             event.total.sensor_pos_x.to_numpy(),
             event.total.sensor_pos_y.to_numpy(),
             event.total.sensor_pos_z.to_numpy()
        ])
        # pts = (pts.transpose() - np.mean(pts, axis=1)).transpose()
        reduced_f = lambda p: f(p, pts)
        res = differential_evolution(
            reduced_f,
            [
                (0, 500),
                (0, 500),
                (0, 2*np.pi),
                (0, 2*np.pi),
                (0, 2*np.pi),
            ]
        )
        rat = res.x[0] / res.x[1]
        if rat < 1:
            rat = 1 / rat
        return res

if __name__=="__main__":
    
    args = initialize_args()
    infile = args.infile
    outfile = args.outfile

    import awkward as ak
    events = ak.from_parquet(infile)
    rats = []
    for event in events:
        res = fit_ellipsoid(event)
        if res is not None:
            rat = res.x[0] / res.x[1]
            if rat < 1:
                rat = 1 / rat
            rats.append(rat)
    np.save(outfile, rats)
