from enum import Enum


class Interactions(Enum):
    """Enum of known interaction types."""

    GLASHOW_RESONANCE = 0
    CHARGED_CURRENT = 1
    NEUTRAL_CURRENT = 2
    DIMUON = 3
    OTHER = 4


# Maps (final_state_1, final_state_2) name strings to interaction-type labels.
INTERACTION_DICT = {
    ("EMinus", "Hadrons"): "CC",
    ("MuMinus", "Hadrons"): "CC",
    ("TauMinus", "Hadrons"): "CC",
    ("EPlus", "Hadrons"): "CC",
    ("MuPlus", "Hadrons"): "CC",
    ("TauPlus", "Hadrons"): "CC",
    ("NuE", "Hadrons"): "NC",
    ("NuMu", "Hadrons"): "NC",
    ("NuTau", "Hadrons"): "NC",
    ("NuEBar", "Hadrons"): "NC",
    ("NuMuBar", "Hadrons"): "NC",
    ("NuTauBar", "Hadrons"): "NC",
}

# Maps (initial_pdg, final1_pdg, final2_pdg) tuples from LeptonInjector output
# to Interactions enum values.
INTERACTION_CONVERTER = {
    (12, -2000001006, 11): Interactions.CHARGED_CURRENT,
    (14, -2000001006, 13): Interactions.CHARGED_CURRENT,
    (16, -2000001006, 15): Interactions.CHARGED_CURRENT,
    (12, 11, -2000001006): Interactions.CHARGED_CURRENT,
    (14, 13, -2000001006): Interactions.CHARGED_CURRENT,
    (16, 15, -2000001006): Interactions.CHARGED_CURRENT,
    (-12, -2000001006, -11): Interactions.CHARGED_CURRENT,
    (-14, -2000001006, -13): Interactions.CHARGED_CURRENT,
    (-16, -2000001006, -15): Interactions.CHARGED_CURRENT,
    (-12, -11, -2000001006): Interactions.CHARGED_CURRENT,
    (-14, -13, -2000001006): Interactions.CHARGED_CURRENT,
    (-16, -15, -2000001006): Interactions.CHARGED_CURRENT,
    (12, 12, -2000001006): Interactions.NEUTRAL_CURRENT,
    (14, 14, -2000001006): Interactions.NEUTRAL_CURRENT,
    (16, 16, -2000001006): Interactions.NEUTRAL_CURRENT,
    (12, -2000001006, 12): Interactions.NEUTRAL_CURRENT,
    (14, -2000001006, 14): Interactions.NEUTRAL_CURRENT,
    (16, -2000001006, 16): Interactions.NEUTRAL_CURRENT,
    (-12, -12, -2000001006): Interactions.NEUTRAL_CURRENT,
    (-14, -14, -2000001006): Interactions.NEUTRAL_CURRENT,
    (-16, -16, -2000001006): Interactions.NEUTRAL_CURRENT,
    (-12, -2000001006, -12): Interactions.NEUTRAL_CURRENT,
    (-14, -2000001006, -14): Interactions.NEUTRAL_CURRENT,
    (-16, -2000001006, -16): Interactions.NEUTRAL_CURRENT,
    (-12, -2000001006, -2000001006): Interactions.GLASHOW_RESONANCE,
    (-12, -12, 11): Interactions.GLASHOW_RESONANCE,
    (-12, -14, 13): Interactions.GLASHOW_RESONANCE,
    (-12, -16, 15): Interactions.GLASHOW_RESONANCE,
    (-12, 11, -12): Interactions.GLASHOW_RESONANCE,
    (-12, 13, -14): Interactions.GLASHOW_RESONANCE,
    (-12, 15, -16): Interactions.GLASHOW_RESONANCE,
    (14, 13, -13): Interactions.DIMUON,
    (14, -13, 13): Interactions.DIMUON,
    (-14, -13, 13): Interactions.DIMUON,
    (-14, 13, -13): Interactions.DIMUON,
}
