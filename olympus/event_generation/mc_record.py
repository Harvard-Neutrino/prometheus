"""Implements MCRecord class."""


class MCRecord(object):
    """
    Stores MC Truth information.

    Properties:
    event_type: str
    sources: List[PhotonSource_]
        List of photon sources
    mc_info: List[Dict[str, Any]]
        List of dictionaries containing MCTruth information.
    """

    def __init__(self, event_type, sources, mc_info):
        """Initialize MCRecord."""
        self.event_type = event_type
        self.sources = sources
        if not isinstance(mc_info, list):
            mc_info = [mc_info]
        self.mc_info = mc_info

    def __add__(self, other):
        """Combine two MCRecords."""
        if isinstance(other, MCRecord):
            new_ev_type = self.event_type + other.event_type
            new_src = self.sources + other.sources
            new_mcinfo = self.mc_info + other.mc_info
            return MCRecord(new_ev_type, new_src, new_mcinfo)
        raise NotImplementedError("Can only combine with MCRecord.")
