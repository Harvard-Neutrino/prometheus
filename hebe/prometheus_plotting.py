import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hebe.utils import find_cog
from numbers import Number
#from utils.constants import *
#from utils.event import *

def is_empty(event):
    nhit = 0
    for val in event.values():
        nhit += len(val[0])
    return not bool(nhit)

def plot_event(
        parquet_set,
        detector,
        e_id=0,
        brightest_event=False,
        fig_name='event_display.pdf',
        save=True,
        show=False,
        channel='lepton',
        # Optional things to display
        show_doms=True,
        show_lepton=True,
        show_dust_layer=False,
        # Display tuning
        charge_mult=3,
        cut=1.0,
        loss_threshold=0.1,
        cmap='jet_r',
        tmethod='mean',
        elevation_angle=0.0,
        azi_angle=None
    ):
    # TODO: Add formatting check
    if brightest_event:
        for event in parquet_set[channel].sensor_pos_x:
            hit_counts.append(len(event))
        hit_counts = np.array(hit_counts)
        event_id = np.argmax(hit_counts)
        print('The brightest event has the id %d' % event_id)
        print('The energy of the lepton is %.1f' % parquet_set.mc_truth.lepton_energy[event_id])
    else:
        event_id = e_id
    sensor_comb = np.array([
        [
            parquet_set[channel].sensor_pos_x[event_id][i],
            parquet_set[channel].sensor_pos_y[event_id][i],
            parquet_set[channel].sensor_pos_z[event_id][i]
        ]
        for i in range(len(parquet_set[channel].sensor_pos_x[event_id]))
    ])
    fig = plt.figure(figsize=(10, 4))
    ax  = fig.add_subplot(111, projection='3d')
    if isinstance(event.photons_1.sensor_id[0], Number):
        om_ids = (
            [x for x in event.photons_1.sensor_id] +
            [x for x in event.photons_2.sensor_id]
        )
    else:
        om_ids = (
                [tuple(x) for x in event.photons_1.sensor_id if x[0]!=-1] +
                [tuple(x) for x in event.photons_2.sensor_id if x[0]!=-1]
        )
    times = np.array(
        [x for x in event.photons_1.t if x!=-1] +
        [x for x in event.photons_2.t if x!=-1]
    )
    tmin = np.min(times)
    deltat = np.max(times) - tmin
    if cut < 1:
        t_cut = np.quantile(times, cut)
        time_mask = times < t_cut
        # This sucks sorry
        om_ids = [om_id for om_id, t in zip(om_ids, times) if t < t_cut]
        times = times[time_mask]

    # This is ugly but I don't know what else to do...
    unique_ids = list(set(om_ids))

    xyz = np.array([detector[om_id].pos for om_id in unique_ids])
    if len(xyz) > 0:
        xyz += detector.offset
    
        reduced_times = np.zeros(len(unique_ids))
        lcharges = np.zeros(len(unique_ids))
        for i, om_id in enumerate(unique_ids):
            m = np.array([om_id==x for x in om_ids])
            hits = times[m]
            reduced_times[i] = getattr(np, tmethod)(hits)
            lcharges[i] = 1+np.log(len(hits))
        lcharges *= charge_mult

        cmap = getattr(plt.cm, cmap)
        c = cmap((reduced_times - tmin) / deltat)
        scat = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=c,
            alpha=0.4,
            s=lcharges,
            zorder=10
        )

    # Show the dust layer, if requested
    if show_dust_layer:
        N = 1000
        _xx = np.linspace(-500, 500, N) + detector.offset[0]
        _yy = np.linspace(-500, 500, N) + detector.offset[1]
        _zz = np.random.uniform(-32, -132, N) + detector.offset[2]
        ax.scatter(_xx, _yy, _zz, alpha=0.005, s=50 * np.random.rand(N), color='k')

    # Plot IceCube DOM locations in the background, if requested
    if show_doms:
        XYZ = np.array([m.pos for m in detector.modules]) + detector.offset
        # Plot all DOMs
        ax.scatter(
            XYZ[:,0],
            XYZ[:,1],
            XYZ[:,2],
            c='black',
            alpha=0.1,
            s=0.2
        )

    # Make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Make the grid lines transparent TODO make this work ????
    ax.xaxis._axinfo['grid']['color'] =  (1.0, 1.0, 1.0, 0.0)
    ax.yaxis._axinfo['grid']['color'] =  (1.0, 1.0, 1.0, 0.0)
    ax.zaxis._axinfo['grid']['color'] =  (1.0, 1.0, 1.0, 0.0)
    
    # Rotate the axes and update
    if azi_angle is None:
        azi_angle = np.degrees(event.mc_truth.direction[0][1]+np.pi/2)
    ax.view_init(elevation_angle, azi_angle)
    plt.savefig(fig_name, bbox_inches='tight')

    # Show the plot if interactive view was requested
    if show:
        plt.show()
    plt.clf()
    plt.close()
