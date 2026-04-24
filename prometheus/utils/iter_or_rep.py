from itertools import repeat

def iter_or_rep(arg):
    """Wrap a scalar in ``itertools.repeat`` or pass through an iterable unchanged.

    Parameters
    ----------
    arg : object
        Value to wrap. Tuples and lists of length > 1 are returned as-is.
        Everything else (scalar, single-element sequence, or existing ``repeat``)
        is wrapped in ``itertools.repeat``.

    Returns
    -------
    result : iterable
        An infinite iterator that yields the scalar, or the original sequence.
    """
    if isinstance(arg, (tuple, list)):
        if len(arg) == 1:
            return repeat(arg[0])
        else:
            return arg
    elif isinstance(arg, repeat):
        return arg
    else:
        return repeat(arg)
