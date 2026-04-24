def should_propagate(particle):
    """Check whether a particle (or any of its children) has energy losses to propagate.

    Parameters
    ----------
    particle : PropagatableParticle
        Particle to check.

    Returns
    -------
    result : bool
        True if the particle or any direct child has at least one energy loss.
    """
    if len(particle.losses) > 0:
        return True
    for child in particle.children:
        if len(child.losses) > 0:
            return True
    return False
