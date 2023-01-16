def should_propagate(particle):
    if len(particle.losses) > 0:
        return True
    for child in particle.children:
        if len(child.losses) > 0:
            return True
    return False
