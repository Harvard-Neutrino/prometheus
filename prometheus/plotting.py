# -*- coding: utf-8 -*-
# prometheus_plotting.py
# Copyright (C) 2022 Jeffrey Lazar, Stephan Meighen-Berger,
# Script to create event views

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .utils import find_cog

def is_empty(event):
    nhit = 0
    for val in event.values():
        nhit += len(val[0])
    return not bool(nhit)

def plot_brightest(
    events,
    det,
    brightest_event=False,
    figname='event_display.pdf',
    save=True,
    show=False,
    channel='total',
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
    hit_counts = []
    for event in events[channel].sensor_pos_x:
        hit_counts.append(len(event))
    hit_counts = np.array(hit_counts)
    event_id = np.argmax(hit_counts)
    event = events[event_id]
    plot_event(
        event, det, brightest_event=brightest_event, figname=figname,
        save=save, show=show, channel=channel, show_doms=show_doms,
        show_lepton=show_lepton, show_dust_layer=show_dust_layer,
        charge_mult=charge_mult, cut=cut, loss_threshold=loss_threshold,
        cmap=cmap, tmethod=tmethod, elevation_angle=elevation_angle, azi_angle=azi_angle
    )

def plot_event(
    event,
    detector,
    brightest_event=False,
    figname='event_display.pdf',
    save=True,
    show=False,
    channel='total',
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
    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection='3d')
    #particle_fields = [field for field in event.fields if field not in "mc_truth event_id".split()]
    #om_ids = []
    #times = []
    #for field in particle_fields:
    try:
        om_ids = [(int(x[0]), int(x[1])) for x in zip(getattr(event, channel).sensor_string_id, getattr(event, channel).sensor_id)]
    except:
        om_ids = [(int(x[0]), int(x[1])) for x in zip(getattr(event, channel).string_id, getattr(event, channel).sensor_id)]
    times = np.array([x for x in getattr(event, channel).t if x != -1])
    if len(times)==0:
        return 
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
        azi_angle = np.degrees(event.mc_truth.initial_azimuth+np.pi/2)
    ax.view_init(elevation_angle, azi_angle)
    plt.savefig(figname, bbox_inches='tight')

    # Show the plot if interactive view was requested
    if show:
        plt.show()
    plt.clf()
    plt.close()
