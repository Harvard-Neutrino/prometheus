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
    event,
    detector,
    fig_name='event_display.pdf',
    save=True,
    show=False,
    # Optional things to display
    show_doms=True,
    show_track=True,
    show_losses=False,
    show_cog=False,
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
    # Check if there were any photon hits
    if event.photons_1.t[0]==-1 and event.photons_2.t[0]==-1:
        return
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
    #cbar = fig.colorbar(scat, shrink=0.5, aspect=5)

    # TODO figure out how to extract energy from event
    #energy = mcinfo[0]
    #ax.text2D(0.05, 0.85, 'E = {E:0.2f} GeV'.format(E=energy), transform=ax.transAxes)

    ## Plot the particle's energy losses, if requested
    #if show_losses:
    #    markers = {'hadr':'.', 'epair':'^', 'delta':'v', 'brems':'o'}
    #    for loss in lossinfo:
    #        loss_type = loss[0]
    #        position = list(map(float, loss[1:4]))
    #        amount = float(loss[-1])
    #        if amount > loss_threshold:
    #            ax.scatter(
    #                position[0] - detector.offset[0],
    #                position[1] - detector.offset[1],
    #                position[2] - detector.offset[2],
    #                marker=markers[loss_type],
    #                s=np.sqrt(amount)
    #            )

    # TODO make this work
    ## Plot the path that the particle followed, if requested
    #if show_track:
    #    dir_ = tools.get_Cartesian_point(mcinfo[4], mcinfo[5], -1)
    #    point = np.array([
    #        mcinfo[1],
    #        mcinfo[2],
    #        mcinfo[3] - detector.offset[3]
    #    ])

    #    # Find point of closest approach to the detector to center the track
    #    opoint = np.array([0.0, 0.0, -1900.0])
    #    dir_norm = np.linalg.norm(dir_)**2
    #    t_0 = (- dir_[0] * (point[0] - opoint[0]) - dir_[1] * (point[1] - opoint[1]) - dir_[2] * (point[2] - opoint[2])) / dir_norm
    #    cpoint = point + dir_ * t_0

    #    # Make track
    #    tMC = np.linspace(-700.0, 700.0, 100)
    #    x_track = cpoint[0] + dir_[0] * tMC
    #    y_track = cpoint[1] + dir_[1] * tMC
    #    z_track = cpoint[2] + dir_[2] * tMC
    #    ax.scatter(x_track[0], y_track[0], z_track[0], color='black', s=15)
    #    ax.plot(x_track, y_track, z_track, color='gray', lw=0.75)

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

    # TODO make this work
    ## Plot center of gravity of event, if requested
    #if show_cog:
    #    cog = find_cog(event_dict, detector)
    #    ax.scatter(*cog)

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

## Make a display of each event in a hits file
#def plot_event_parquet(
#    event,
#    detector,
#    **kwargs
#):
#    '''
#        Plots the given event.
#        event_dict: dictionary of hits, where the dictionary key is the DOM index.
#        hitinfo: information about each hit
#        lossinfo: information about losses from PROPOSAL
#        mcinfo: true information of the trajectories
#        fig_name: output file name
#        show_doms: plots the IceCube array on the background
#        show_track: plots the true trajectory direction
#    '''
#    # Check whether the event had any hits
#    if is_empty(event):
#        return
#
#    if plot_keys is None:
#        plot_keys = event.keys()
#
#    ## Construct a dictionary whose keys are om keys and vals are list of times
#    #event_dict = {}
#    #for key in plot_keys:
#    #    for hit in event[key][0]:
#    #        om_key = (hit[0], hit[1])
#    #        if om_key not in event_dict.keys():
#    #            event_dict[om_key] = [hit[2]]
#    #        else:
#    #            event_dict[om_key].append(hit[2])
#    
#    fig = plt.figure(figsize=(15, 15))
#    ax  = fig.add_subplot(111, projection='3d')
#    # TODO figure out how to extract energy from event
#    #energy = mcinfo[0]
#    #ax.text2D(0.05, 0.85, 'E = {E:0.2f} GeV'.format(E=energy), transform=ax.transAxes)
#
#    if isinstance(event.photons_1.sensor_id[0], Number):
#        om_ids = (
#                [x for x in event.photons_1.sensor_id] +
#                [x for x in event.photons_2.sensor_id]
#        )
#    else:
#        om_ids = (
#                [tuple(x) for x in event.photons_1.sensor_id] +
#                [tuple(x) for x in event.photons_2.sensor_id]
#        )
#    # This is ugly but I don't know what else to do...
#    unique_ids = list(set(om_ids))
#    xyz = np.array([detector[om_id].pos for key in keys_])
#    times = np.array([getattr(np, tmethod)(event_dict[key]) for key in keys_])
#    lcharges = np.array([1+np.log(len(event_dict[key])) for key in keys_])
#    lcharges *= charge_mult
#    t_cut = np.quantile(times, cut)
#    m = times <= t_cut
#    scat = ax.scatter(xyz[m, 0], xyz[m, 1], xyz[m, 2], c=times[m], alpha=0.5, s=lcharges[m], cmap=cmap, zorder=10)
#    cbar = fig.colorbar(scat, shrink=0.5, aspect=5)
#
#    # Plot the path that the particle followed, if requested
#    if show_track:
#        z_offset = 1948.07
#        dir_ = tools.get_Cartesian_point(mcinfo[4], mcinfo[5], -1)
#        point = np.array([mcinfo[1], mcinfo[2], mcinfo[3] - z_offset])
#
#        # Find point of closest approach to the detector to center the track
#        opoint = np.array([0.0, 0.0, -1900.0])
#        dir_norm = np.linalg.norm(dir_)**2
#        t_0 = (- dir_[0] * (point[0] - opoint[0]) - dir_[1] * (point[1] - opoint[1]) - dir_[2] * (point[2] - opoint[2])) / dir_norm
#        cpoint = point + dir_ * t_0
#
#        # Make track
#        tMC = np.linspace(-700.0, 700.0, 100)
#        x_track = cpoint[0] + dir_[0] * tMC
#        y_track = cpoint[1] + dir_[1] * tMC
#        z_track = cpoint[2] + dir_[2] * tMC
#        ax.scatter(x_track[0], y_track[0], z_track[0], color='black', s=15)
#        ax.plot(x_track, y_track, z_track, color='gray', lw=0.75)
#
#    # Show the dust layer, if requested
#    if show_dust_layer:
#        N = 1000
#        _xx = np.linspace(-500, 500, N)
#        _yy = np.linspace(-500, 500, N)
#        _zz = np.random.uniform(-32, -132, N)
#        ax.scatter(_xx, _yy, _zz, alpha=0.02, s=500 * np.random.rand(N), color='k')
#
#    # Plot IceCube DOM locations in the background, if requested
#    if show_doms:
#        # 60 excludes IceTop3
#        XYZ = np.array([m.pos for m in detector.modules if m.key[1]<60])
#        # Plot all DOMs
#        ax.scatter(XYZ[:,0], XYZ[:,1], XYZ[:,2], c='black', alpha=0.2, s=2)
#
#    # Plot center of gravity of event, if requested
#    if show_cog:
#        cog = find_cog(event_dict, detector)
#        ax.scatter(*cog)
#
#    # Make the panes transparent
#    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#
#    # Make the grid lines transparent TODO make this work ????
#    ax.xaxis._axinfo['grid']['color'] =  (1.0, 1.0, 1.0, 0.0)
#    ax.yaxis._axinfo['grid']['color'] =  (1.0, 1.0, 1.0, 0.0)
#    ax.zaxis._axinfo['grid']['color'] =  (1.0, 1.0, 1.0, 0.0)
#
#    # Rotate the axes and update
#    #ax.view_init(elevation_angle, np.degrees(mcinfo[5]) + 90)
#    ax.view_init(elevation_angle, 0)
#    print(fig_name)
#    plt.savefig(fig_name, bbox_inches='tight')
#
#    # Show the plot if interactive view was requested
#    if show:
#        plt.show()
#    plt.clf()
#    plt.close()
