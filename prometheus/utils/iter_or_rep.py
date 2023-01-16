from itertools import repeat

def iter_or_rep(arg):
    if isinstance(arg, (tuple, list)):
        if len(arg) == 1:
            return repeat(arg[0])
        else:
            return arg
    elif isinstance(arg, repeat):
        return arg
    else:
        return repeat(arg)
