import numpy as np

import symbolic_pymc as sp

from collections import Hashable


class HashableNDArray(np.ndarray, Hashable):
    """A subclass of Numpy's ndarray that uses `tostring` hashing and `array_equal` equality testing.

    Usage
    -----
        >>> import numpy as np
        >>> from symbolic_pymc.utils import HashableNDArray
        >>> x = np.r_[1, 2, 3]
        >>> x_new = x.view(HashableNDArray)
        >>> assert hash(x_new) == hash(x.tostring())
        >>> assert x_new == np.r_[1, 2, 3]
    """

    def __hash__(self):
        return hash(self.tostring())

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        if self.__eq__(other):
            return False

        return NotImplemented


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
    elif not x == y:
        res = (x, y)

    if res is not None:
        if pdb:
            import pdb

            pdb.set_trace()
        return res
