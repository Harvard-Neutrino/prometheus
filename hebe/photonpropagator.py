# -*- coding: utf-8 -*-
# photonpropagator.py
# Authors: Christian Haack, Jeffrey Lazar, Stephan Meighen-Berger,
# Interface class to the different photon propagators

# imports
import functools
from .config import config
from .lepton_prop import LP
import sys
import json
import awkward as ak
import numpy as np
import plotly.graph_objects as go
from .utils import write_to_f2k_format

from .detector_handler import DH
# sys.path.append(config['photon propagator']['location'])
sys.path.append('../')

from olympus.event_generation.photon_propagation.norm_flow_photons import (  # noqa
    make_generate_norm_flow_photons
)
from olympus.event_generation.event_generation import (  # noqa: E402
    generate_cascade,
    generate_realistic_track,
    simulate_noise,
)
from olympus.event_generation.lightyield import make_realistic_cascade_source # noqa
from olympus.event_generation.utils import sph_to_cart_jnp  # noqa: E402

from hyperion.medium import medium_collections  # noqa: E402
from hyperion.constants import Constants  # noqa: E402

# A dict for converting PDG names to f2k names
# TODO momve this outside
PDG_type = [22, 11, -11, 13, -13, 2212]
f2k_type = ['gamma', 'e-', 'e+', 'mu-', 'mu+', 'hadr']
PDG_to_f2k = {x:y for x, y in zip(PDG_type, f2k_type)}

def _parse_ppc(ppc_f):
    res_result = [] # timing and module
    hits = []
    with open(ppc_f) as ppc_out:
        for line in ppc_out:
            #if "EE" in line: # Event is over
            #    hits = []
            if "HIT" in line: # Photon was seen
                l = line.split()
                '''(string number, dom number, time, wavelength, 
                dom zenith, dom azimuth, photon theta, photon azimuth)
                (N, n, ns, nm, radian, radian, radian, radian)
                '''
                hits.append(
                    (int(l[1]), int(l[2]), float(l[3]), float(l[4]),
                    float(l[5]), float(l[6]), float(l[7]), float(l[8]))
                )
    return hits


def _ppc_sim(
        event,
        dh,
        det,
        lp,
        ppc_config,
        padding
        ):
    """ Utilizes ppc to propagate light for the injected object

    Parameters
    ---------
    event : Dict
        Event to be simulated
    det : Detector
        Detector object in which to simulate
    pprop_func : function
        Function to calculate the photon signal
    proposal_prop : function
            Propoposal propagator

    Returns
    -------
    res_event : 
        
    res_record : 
        
    """
    # TODO figure out what format the losses should have and make that
    # Compute losses and write them to the temporary outfile
    print("=================")
    print(ppc_config['f2k_tmpfile'])
    print("=================")

    f2k_file = ppc_config['f2k_tmpfile'].replace("event", f"{ppc_config['f2k_prefix']}event")
    with open(f2k_file, "w") as ppc_f:
        # The particle is a charged lepton and should be handled by PROPOSAL
        if abs(event['particle_id']) in [11,13,15]:
            event_dir = sph_to_cart_jnp(
                event["theta"],
                event["phi"]
            )
            event_dir *= -1
            event["dir"] = event_dir
            r = np.linalg.norm(event["pos"])
            # This is probably where we would factor it out
            losses, total_loss = lp.j_losses(event, r+padding)
        # The event is a hadronic shower (I think?)
        else:
            key = PDG_to_f2k[event['particle_id']]
            losses = {key:[[np.log10(event['energy']), event['pos']]]}
            total_loss = None
        # Make array with energy loss information and write it into the output file
        # print(losses)
        output_info = [np.cos(event['theta']), event['phi'], event['energy'], event["pos"], losses]
        write_to_f2k_format(output_info, ppc_f, 0)
    # Propagate with PPC
    # This file path is hardcoded into PPC. Don't change
    geo_tmpfile = f'{ppc_config["ppctables"]}/geo-f2k'
    # Write the geometry out to an f2k file
    #dh.to_f2k(det, geo_tmpfile, serial_nos=[m.serial_no for m in det.modules])
    ppc_file = ppc_config['ppc_tmpfile'].replace("event", f"{ppc_config['ppc_prefix']}_event")
    command = f"{ppc_config['ppc_exe']} {ppc_config['device']} < {f2k_file} > {ppc_file}"
    print(command)
    import os
    tenv = os.environ.copy()
    tenv["PPCTABLESDIR"] = ppc_config["ppctables"]

    import subprocess
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, env=tenv)
    process.wait()
    hits = _parse_ppc(ppc_file)
    ## Cleanup f2k_tmpfile
    os.remove(ppc_file)
    os.remove(f2k_file)
    return hits, None


def _olympus_sim(injection_event, det, key, pprop_func, proposal_prop):
    """ Utilizes olympus to propagate light for the injected object

    Parameters
    ---------
    injection_event : Dict
        Event to be simulated
    det : Detector
        Detector object in which to simulate
    key : PRNGKey
    pprop_func : function
        Function to calculate the photon signal
    proposal_prop : function
            Propoposal propagator

    Returns
    -------
    res_event : 
        
    res_record : 
        
    """
    event_dir = sph_to_cart_jnp(
        injection_event["theta"],
        injection_event["phi"]
    )
    injection_event["dir"] = event_dir
    # Tracks
    if injection_event['event id'] in config['particles'][
            'track particles']:
        res_event, res_record = (
            generate_realistic_track(
                det,
                injection_event,
                key=key,
                pprop_func=pprop_func,
                proposal_prop=proposal_prop
            )
        )
    # Cascades
    else:
        res_event, res_record = generate_cascade(
            det,
            injection_event,
            seed=config['runtime']['random state jax'],
            converter_func=functools.partial(
                make_realistic_cascade_source,
                moliere_rand=True,
                # TODO should this be an kwarg ?
                resolution=0.2),
            pprop_func=pprop_func
        )
        if config['run']['noise']:
            res_event, _ = simulate_noise(det, res_event)
    return res_event, res_record


class PP(object):
    ''' interface class between the different photon propagators

    Parameters
    ----------
    lp : LP object
        A lepton propagation instance
    dh : dictionary
        A detector dictionary
    '''
    def __init__(self, lp: LP, det):
        print('--------------------------------------------')
        print('Constructing the photon propagator')
        if config['photon propagator']['name'] == 'olympus':
            self._local_conf = config['photon propagator']['olympus']
            self.__lp = lp
            self.__det = det
            self._olympus_setup()
            self._plotting = self._olympus_plot_event
            self._sim = self._olympus_sim
        elif config['photon propagator']['name'] == 'PPC':
            self._local_conf = config['photon propagator']['PPC']
            self.__lp = lp
            self.__det = det
            self.__dh = DH()
            # TODO add in event displays
            # self._plotting = plot_event
            # TODO make a function for propagating photons
            self._sim = lambda event: _ppc_sim(
                    event,
                    self.__dh,
                    self.__det,
                    self.__lp,
                    self._local_conf,
                    config['lepton propagator']['propagation padding'],
            )
        elif config['photon propagator']['name'] == 'PPC_CUDA':
            self._local_conf = config['photon propagator']['PPC_CUDA']
            self.__lp = lp
            self.__det = det
            self.__dh = DH()
            # TODO add in event displays
            #self._plotting = plot_event
            # TODO make a function for propagating photons
            self._sim = lambda event: _ppc_sim(
                    event, 
                    self.__dh,
                    self.__det,
                    self.__lp, 
                    self._local_conf,
                    config['lepton propagator']['propagation padding'],
            )
            
        else:
            raise ValueError(
                'Unknown photon propagator! Check the config file'
            )
        print('--------------------------------------------')

    #def _ppc_setup(self):
    #    ''' Sets up PPC.
    #    '''
    #    print('Using PPC')
    #    print('Setting up the medium')
    #    self._pprop_config = json.load(open(
    #        self._pprop_path))['photon_propagation']
    #    # The medium
    #    self._ref_ix_f, self._sca_a_f, self._sca_l_f = medium_collections[
    #        self._pprop_config['medium']
    #    ]
    #    print('Finished the medium')
    #    print('-------------------------------------------------------')
    #    print('Setting up the photon generator')
    #    if self._local_conf['files']:
    #        self._gen_ph = make_generate_norm_flow_photons(
    #            self._local_conf['location'] + self._local_conf['flow'],
    #            self._local_conf['location'] + self._local_conf['counts'],
    #            c_medium=self._c_medium_f(
    #                self._local_conf['wavelength']) / 1E9
    #        )
    #    else:
    #        ValueError('Currently only file runs for olympus are supported!')
    #    print('Finished the photon generator')

    def _olympus_setup(self):
        ''' Sets up olympus.
        '''
        print('Using olympus')
        print('Setting up the medium')
        self._pprop_path = (
            self._local_conf['location'] + self._local_conf['photon model']
        )
        self._pprop_config = json.load(open(
            self._pprop_path))['photon_propagation']
        # The medium
        self._ref_ix_f, self._sca_a_f, self._sca_l_f = medium_collections[
            self._pprop_config['medium']
        ]
        print('Finished the medium')
        print('-------------------------------------------------------')
        print('Setting up the photon generator')
        if self._local_conf['files']:
            self._gen_ph = make_generate_norm_flow_photons(
                self._local_conf['location'] + self._local_conf['flow'],
                self._local_conf['location'] + self._local_conf['counts'],
                c_medium=self._c_medium_f(
                    self._local_conf['wavelength']) / 1E9
            )
        else:
            ValueError('Currently only file runs for olympus are supported!')
        print('Finished the photon generator')

    def _olympus_sim(self, injection_event):
        """ Utilizes olympus to propagate light for the injected object
        """
        event_dir = sph_to_cart_jnp(
            injection_event["theta"],
            injection_event["phi"]
        )
        injection_event["dir"] = event_dir
        # Tracks
        if injection_event['event id'] in config['particles'][
                'track particles']:
            res_event, res_record = (
                generate_realistic_track(
                    self.__det,
                    injection_event,
                    key=config['runtime']['random state jax'],
                    pprop_func=self._gen_ph,
                    proposal_prop=self.__lp.prop
                )
            )
        # Cascades
        else:
            res_event, res_record = generate_cascade(
                self.__det,
                injection_event,
                seed=config['runtime']['random state jax'],
                converter_func=functools.partial(
                    make_realistic_cascade_source,
                    moliere_rand=True,
                    resolution=0.2),
                pprop_func=self._gen_ph
            )
            if config['run']['noise']:
                res_event, _ = simulate_noise(self._det, res_event)
        return res_event, res_record

    def _olympus_plot_event(
            self,
            det,
            hit_times,
            record=None,
            plot_tfirst=False,
            plot_hull=False):
        """ helper function to plot events
        """
        if plot_tfirst:
            plot_target = ak.fill_none(ak.firsts(hit_times, axis=1), np.nan)
        else:
            plot_target = np.log10(ak.count(hit_times, axis=1))

        mask = (plot_target > 0) & (plot_target != np.nan)

        traces = [
            go.Scatter3d(
                x=det.module_coords[mask, 0],
                y=det.module_coords[mask, 1],
                z=det.module_coords[mask, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=plot_target[mask],  # set color to an array/list
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                    showscale=True,
                ),
            ),
            go.Scatter3d(
                x=det.module_coords[~mask, 0],
                y=det.module_coords[~mask, 1],
                z=det.module_coords[~mask, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color="black",  # set color to an array/list
                    colorscale="Viridis",  # choose a colorscale
                    opacity=1.,
                ),
            ),
        ]

        if record is not None:
            positions = []
            sizes = []
            for source in record.sources:
                sizes.append(
                    np.asscalar((np.log10(source.n_photons) / 2) ** 2))
                positions.append(
                    [source.position[0],
                     source.position[1],
                     source.position[2]]
                )
            positions = np.asarray(positions)
            traces.append(
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode="markers",
                    marker=dict(
                        size=sizes, color="black",
                        opacity=0.5, line=dict(width=0)),
                )
            )
        if plot_hull:
            # Cylinder
            radius = det.outer_cylinder[0]
            height = det.outer_cylinder[1]
            z = np.linspace(-height / 2, height / 2, 100)
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid)
            y_grid = radius * np.sin(theta_grid)

            traces.append(
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=z_grid,
                    colorscale=[[0, "blue"], [1, "blue"]],
                    opacity=0.2,
                )
            )
        fig = go.Figure(
            data=traces,
        )
        fig.update_layout(
            showlegend=False,
            height=700,
            width=1400,
            coloraxis_showscale=True,
            scene=dict(
                xaxis=dict(range=config['plot']['xrange']),
                yaxis=dict(range=config['plot']['yrange']),
                zaxis=dict(range=config['plot']['zrange']),
            ),
        )
        fig.update_coloraxes(colorbar_title=dict(text="log10(det. photons)"))
        fig.show()
        return fig

    def _c_medium_f(self, wl):
        """ Speed of light in medium for wl (nm)
        """
        return Constants.BaseConstants.c_vac / self._ref_ix_f(wl)
