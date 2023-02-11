import numpy as np

import sys
sys.path.append("../")

from prometheus.injection.injection.LI_injection import injection_from_LI_output

def test_li_injection_loading() -> None:
    injection = injection_from_LI_output("resources/MuMinus_Hadrons_seed_925_LI_output.h5")
    assert(len(injection)==50)
    assert(all([len(event.final_states)==2 for event in injection]))
    assert(all([event.interaction.name=="CHARGED_CURRENT" for event in injection]))
    assert(all([event.initial_state.pdg_code==14 for event in injection]))
    assert(all([event.final_states[0].pdg_code==13 for event in injection]))
    assert(all([event.final_states[1].pdg_code==-2000001006 for event in injection]))
    assert(sum([event.bjorken_x for event in injection])-5.38459932518872 < 1e-5)
    assert(sum([event.bjorken_y for event in injection])-15.304668192727698 < 1e-5)
    assert(sum([event.column_depth for event in injection])-5398104.596123365 < 1e-5)
    assert(sum([event.vertex_x for event in injection]) + 903.6668603488667 < 1e-5)
    assert(sum([event.vertex_y for event in injection]) - 1720.581858684919 < 1e-5)
    assert(sum([event.vertex_z for event in injection]) + 97267.71623169626 < 1e-5)
    assert(sum([event.initial_state.e for event in injection]) - 5805025.309993334 < 1e-5)
    assert(sum(([event.initial_state.phi for event in injection])) - 24.598147958495357 < 1e-5)
    assert(sum(([event.initial_state.theta for event in injection])) - 72.11030193309311 < 1e-5)
    assert(np.sum([event.initial_state.position for event in injection]) + 96450.80123336022 < 1e-5)
    assert(np.sum([event.initial_state.direction for event in injection]) - 9.578980695601722 < 1e-5)
    assert(np.sum([event.final_states[0].direction for event in injection]) - 1.4888529313548755 < 1e-5)
    assert(np.sum([event.final_states[1].direction for event in injection]) - 0.6596348700869346 < 1e-5)
    assert(
         np.sum([event.final_states[0].position for event in injection])-np.sum([event.initial_state.position for event in injection]) < 1e-5
    )
    assert(
         np.sum([event.final_states[1].position for event in injection])-np.sum([event.initial_state.position for event in injection]) < 1e-5
    )

