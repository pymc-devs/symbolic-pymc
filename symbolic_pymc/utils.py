import numpy as np

import symbolic_pymc as sp


def _check_eq(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b


def meta_parts_unequal(x, y, pdb=False):  # pragma: no cover
    """Traverse meta objects and return the first pair of elements that are not equal."""
    res = None
    if type(x) != type(y):
        print("unequal types")
        res = (x, y)
    elif isinstance(x, sp.meta.MetaSymbol):
        if x.base != y.base:
            print("unequal bases")
            res = (x.base, y.base)
        else:
            for a, b in zip(x.rands(), y.rands()):
                z = meta_parts_unequal(a, b, pdb=pdb)
                if z is not None:
                    res = z
                    break
    elif isinstance(x, (tuple, list)):
        for a, b in zip(x, y):
            z = meta_parts_unequal(a, b, pdb=pdb)
            if z is not None:
                res = z
                break
    elif not _check_eq(x, y):
        res = (x, y)

    if res is not None:
        if pdb:
            import pdb

            pdb.set_trace()
        return res
