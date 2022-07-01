# A dict for converting PDG names to f2k names
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
    2212:'p-',
    211:'pi+',
    -211:'pi0',
    111:'pi0',

}

f2k_to_PDG = {val:key for key, val in PDG_to_f2k.items()}

PDG_to_pstring = {
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
    2212:"Proton",
    211:'PiPlus',
    -211:'PiMinus',
    111:'PiZero',
}

pstring_to_PDG = {val:key for key, val in PDG_to_pstring.items()}

int_type_to_str = {
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
}

str_to_int_type = {val:key for key, val in int_type_to_str.items()}
