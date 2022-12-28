import awkward as ak
import h5py as h5
import numpy as np

from .injection import Injection
from .errors import (
    MultipleInjectionError,
    TooManyInjectorsError,
    DataNotLoadedError
)

INTERACTION_CONVERTER = {
    # Charged current
    (12, -2000001006, 11): 1,
    (14, -2000001006, 13): 1,
    (16, -2000001006, 15): 1,
    (12, 11, -2000001006): 1,
    (14, 13, -2000001006): 1,
    (16, 15, -2000001006): 1,
    (-12, -2000001006, -11): 1,
    (-14, -2000001006, -13): 1,
    (-16, -2000001006, -15): 1,
    (-12, -11, -2000001006): 1,
    (-14, -13, -2000001006): 1,
    (-16, -15, -2000001006): 1,
    # Neutral current
    (12, 12, -2000001006): 2,
    (14, 14, -2000001006): 2,
    (16, 16,-2000001006): 2,
    (12, -2000001006, 12): 2,
    (14, -2000001006, 14): 2,
    (16,-2000001006, 16): 2,
    (-12, -12,-2000001006): 2,
    (-14, -14,-2000001006): 2,
    (-16, -16,-2000001006): 2,
    (-12,-2000001006, -12): 2,
    (-14,-2000001006, -14): 2,
    (-16,-2000001006, -16): 2,
    # Glashow
    (-12, -2000001006, -2000001006): 0,
    (-12,-12, 11): 0,
    (-12,-14, 13): 0,
    (-12,-16, 15): 0,
    (-12, 11,-12): 0,
    (-12, 13,-14): 0,
    (-12, 15,-16): 0,
    # Dimuon
    (14, 13, -13): 3,
    (14, -13, 13): 3,
    (-14, -13, 13): 3,
    (-14, 13, -13): 3,
}

LEPTON_PDGS = [-16, -15, -14, -13, -12, -11, 11, 12, 13, 14, 15, 16]

def make_new_injection(path_dict, injection_specs):
    import os
    try:
        import LeptonInjector as LI
    except ImportError:
        raise ImportError('LeptonInjector not found!')
    print('Setting up the LI')
    print('Fetching parameters and setting paths')
    xs_folder = os.path.join(
        os.path.dirname(__file__),
        path_dict["xsec location"]
    )
    print('Setting the simulation parameters for the LI')
    n_events = injection_specs['nevents']
    diff_xs = f"{path_dict['xsec location']}/{path_dict['diff xsec']}"
    total_xs = f"{path_dict['xsec location']}/{path_dict['total xsec']}"
    is_ranged = injection_specs['is ranged']
    particles = []
    for id_name, names in enumerate([
        injection_specs['final state 1'], injection_specs['final state 2']
    ]):
        particles.append(getattr(LI.Particle.ParticleType, names))
    
    print('Setting up the LI object')
    the_injector = LI.Injector(
        injection_specs["nevents"],
        particles[0],
        particles[1],
        diff_xs,
        total_xs,
        is_ranged
    )
    print('Setting injection parameters')

    # define some defaults
    min_E = injection_specs['minimal energy']     # [GeV]
    max_E = injection_specs['maximal energy']    # [GeV]
    gamma = injection_specs['power law']
    min_zenith = np.radians(injection_specs['min zenith'])
    max_zenith = np.radians(injection_specs['max zenith'])
    min_azimuth = np.radians(injection_specs['min azimuth'])
    max_azimuth = np.radians(injection_specs['max azimuth'])
    injectRad = injection_specs["injection radius"]
    endcap_length = injection_specs["endcap length"]
    cyinder_radius = injection_specs["cylinder radius"]
    cyinder_height = injection_specs["cylinder height"]
    print('Building the injection handler')
    # construct the controller
    if is_ranged:
        controller = LI.Controller(
            the_injector, min_E, max_E, gamma, min_azimuth,
            max_azimuth, min_zenith, max_zenith, 
        )
    else:
        controller = LI.Controller(
            the_injector, min_E, max_E, gamma, min_azimuth,
            max_azimuth, min_zenith, max_zenith,
            injectRad, endcap_length, cyinder_radius, cyinder_height
        )
    print('Defining the earth model')
    path_to = os.path.join(
        os.path.dirname(__file__),
        xs_folder, path_dict['earth model location']
    )
    print(
        'Earth model location to use: ' +
        path_to + ' With the model ' + injection_specs['earth model']
    )
    print(injection_specs['earth model'], path_dict['earth model location'])
    controller.SetEarthModel(injection_specs['earth model'], path_to)
    print("Setting the seed")
    controller.setSeed(injection_specs["random state seed"])
    print('Defining the output location')
    print(path_dict['output name'])
    controller.NameOutfile(path_dict['output name'])
    controller.NameLicFile(path_dict['lic name'])

    # run the simulation
    controller.Execute()

class LeptonInjectorInjection(Injection):

    def __init__(self, injection_file):
        self._injection_file = injection_file
        self._data_loaded = False

    def load_data(self, injector_key=None):

        self._h5f = h5.File(self._injection_file, "r")

        if len(self._h5f.keys()) > 1 and injector_key is None:
            raise TooManyInjectorsError()
        elif injector_key is not None:
            self._injector_key = injector_key
        else:
            self._injector_key = list(self._h5f.keys())[0]
        self._injection = self._h5f[self._injector_key]

        # Make sure we only have one kind of interaction. Not sure why this
        # is necessary but whatever...
        if not (
            np.all(self._injection["final_1"]["ParticleType"]==self._injection["final_1"]["ParticleType"][0]) and
            np.all(self._injection["final_2"]["ParticleType"]==self._injection["final_2"]["ParticleType"][0])
        ):
            raise MultipleInjectionError()

        if self._injection["final_1"]["ParticleType"][0] in LEPTON_PDGS:
            self._lepton_key = "final_1"
            self._hadron_key = "final_2"
        else:
            self._lepton_key = "final_2"
            self._hadron_key = "final_1"
        self._data_loaded = True

    def __del__(self):
        if self._data_loaded:
            self._h5f.close()

    def __len__(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return len(self.injection_energy)

    @property
    def injection_file(self):
        return self._injection_file

    @property
    def injection_energy(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["totalEnergy"]

    @property
    def injection_type(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["initialType"]

    @property
    def injection_interaction_type(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        val = [
            INTERACTION_CONVERTER[tuple(t)] for t in self._injection["properties"][:][["initialType", "finalType1", "finalType2"]]
        ]
        return val

    @property
    def injection_zenith(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["zenith"]

    @property
    def injection_azimuth(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["azimuth"]

    @property
    def injection_bjorkenx(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["finalStateX"]

    @property
    def injection_bjorkeny(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["finalStateY"]

    @property
    def injection_position_x(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["x"]

    @property
    def injection_position_y(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["y"]

    @property
    def injection_position_z(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["z"]

    @property
    def injection_column_depth(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["totalColumnDepth"]

    @property
    def primary_particle_1_type(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._lepton_key]["ParticleType"]

    @property
    def primary_particle_1_energy(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._lepton_key]["Energy"]

    @property
    def primary_particle_1_position_x(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._lepton_key]["Position"][:, 0]

    @property
    def primary_particle_1_position_y(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._lepton_key]["Position"][:, 1]

    @property
    def primary_particle_1_position_z(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._lepton_key]["Position"][:, 2]

    @property
    def primary_particle_1_direction_theta(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._lepton_key]["Direction"][:, 0]

    @property
    def primary_particle_1_direction_phi(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._lepton_key]["Direction"][:, 1]

    @property
    def primary_particle_2_type(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._hadron_key]["ParticleType"]

    @property
    def primary_particle_2_energy(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._hadron_key]["Energy"]

    @property
    def primary_particle_2_position_x(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._hadron_key]["Position"][:, 0]

    @property
    def primary_particle_2_position_y(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._hadron_key]["Position"][:, 1]

    @property
    def primary_particle_2_position_z(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._hadron_key]["Position"][:, 2]

    @property
    def primary_particle_2_direction_theta(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._hadron_key]["Direction"][:, 0]

    @property
    def primary_particle_2_direction_phi(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection[self._hadron_key]["Direction"][:, 1]

    @property
    def total_energy(self):
        if not self._data_loaded:
            raise DataNotLoadedError()
        return self._injection["properties"]["totalEnergy"]


    def serialize_to_dict(self):
        all_fields = [
            'injection_energy',
            'injection_type',
            'injection_interaction_type',
            'injection_zenith',
            'injection_azimuth',
            'injection_bjorkenx',
            'injection_bjorkeny',
            'injection_position_x',
            'injection_position_y',
            'injection_position_z',
            'injection_column_depth',
            'primary_particle_1_type',
            'primary_particle_1_position_x',
            'primary_particle_1_position_y',
            'primary_particle_1_position_z',
            'primary_particle_1_direction_theta',
            'primary_particle_1_direction_phi',
            'primary_particle_1_energy',
            'primary_particle_2_type',
            'primary_particle_2_position_x',
            'primary_particle_2_position_y',
            'primary_particle_2_position_z',
            'primary_particle_2_direction_theta',
            'primary_particle_2_direction_phi',
            'primary_particle_2_energy',
            'total_energy',
        ]
    
        # We need to copy these since they are taken from
        # non-contiguous portions of the array, i.e.
        # we are slicing on the first index
        copy_fields = [
            'primary_particle_1_position_x',
            'primary_particle_1_position_y',
            'primary_particle_1_position_z',
            'primary_particle_1_direction_theta',
            'primary_particle_1_direction_phi',
            'primary_particle_2_position_x',
            'primary_particle_2_position_y',
            'primary_particle_2_position_z',
            'primary_particle_2_direction_theta',
            'primary_particle_2_direction_phi',
        ]
        # Sorry this is nasty :-(
        dict_ = {
            field:(np.copy(getattr(self, field)) if field in copy_fields else getattr(self, field)) 
            for field in all_fields
        }
        return dict_

    def serialize_to_awkward(self):
        return ak.Array(self.serialize_to_dict())

    def inject(self, injection_config):
        make_new_injection(
            injection_config["paths"],
            injection_config["simulation"]
        )
