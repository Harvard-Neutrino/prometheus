# A dict for converting PDG names to f2k names from
# https://www.zeuthen.desy.de/~steffenp/f2000/f2000_1.5.html#SEC26
PDG_to_f2k = {
    11:"e-", 
    -11:"e+",
    12:"nu_e",
    -12:"~nu_e",
    13:"mu-",
    -13:"mu+",
    14:"nu_mu",
    -14:"~nu_mu",
    22:"gamma",
    111:'pi0', # This is technically not defined but...
    211:'hadr',
    -211:'hadr',
    311:'k0', # This is technically not defined but...
    2212:'p+',
    -2212:'p-',
}

f2k_to_PDG = {val:key for key, val in PDG_to_f2k.items()}

# mapping from https://github.com/icecube/LeptonInjector/blob/master/private/LeptonInjector/Particle.cxx
PDG_to_pstring = {
    0:"Unknown",
    11:"EMinus",
    12:"NuE",
    13:"MuMinus",
    14:"NuMu",
    15:"TauMinus",
    16:"NuTau",
    -11:"EPlus",
    -12:"NuEBar",
    -13:"MuPlus",
    -14:"NuMuBar",
    -15:"TauPlus",
    -16:"NuTaBar",
    2212:"Proton", # I'm not sure that these are defined either but...
    -2212:"AntiProton", # I'm not sure that these are defined either but...
    211:'PiPlus', # I'm not sure that these are defined either but...
    -211:'PiMinus', # I'm not sure that these are defined either but...
    311:'KZero', # I'm not sure that these are defined either but...
    111:'PiZero', # I'm not sure that these are defined either but...
    -2000001006:"Hadrons",
}

pstring_to_PDG = {val:key for key, val in PDG_to_pstring.items()}

# Mapping from https://github.com/tudo-astroparticlephysics/PROPOSAL/blob/master/src/PROPOSAL/PROPOSAL/particle/Particle.h
# to https://www.zeuthen.desy.de/~steffenp/f2000/f2000_1.5.html#SEC26
int_type_to_str = {
    211:"hadr",
    -211:"hadr",
    #1000000001:"",
    1000000002:"brems",
    1000000003:"delta",
    1000000004:"epair",
    # Should we be mapping photonuclear to hadr ?
    1000000005:"hadr",
    1000000006:"mupair",
    1000000007:"hadr",
    # What do we do with this ???????
    1000000008:"delta",
    -2000001006:"hadr",
    2212:"hadr",
}

str_to_int_type = {val:key for key, val in int_type_to_str.items()}
