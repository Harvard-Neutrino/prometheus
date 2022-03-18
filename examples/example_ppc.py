import sys
sys.path.append('../')
from hebe import HEBE, config
from jax.config import config as jconfig

jconfig.update("jax_enable_x64", True)

nevent = 2

config["general"]["random state seed"] = 1337
config['photon propagator']['name'] = 'PPC_CUDA'
config['run']['group name'] = 'VolumeInjector0'
config['run']['subset']['counts'] = nevent

pname = 'MuMinus'
            
config['lepton injector']['simulation']['nevents'] = nevent
config['lepton injector']['simulation']['is ranged'] = False
config['lepton injector']['simulation']['final state 1'] = pname
config['lepton injector']['simulation']['minimal energy'] = 1e5
config['lepton injector']['simulation']['maximal energy'] = 1e5
config['lepton injector']['simulation']['maxZenith'] = 180
config['lepton injector']['simulation']['minZenith'] = 90

config['lepton propagator']['lepton'] = pname

config["detector"]["file name"] = '../hebe/data/icecube-f2k'


hebe = HEBE(userconfig=config)
hebe.sim()
for idx in range(nevent):
    event = {'final_1':[hebe.results['final_1'][idx]], 'final_2':[hebe.results['final_2'][idx]]}
    hebe.ppc_event_plotting(event, fig_name=f'./{pname}_test_{idx}.pdf', show_track=False, show_dust_layer=True)
