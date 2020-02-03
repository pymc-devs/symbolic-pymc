import numpy as np

from operator import ne, attrgetter, itemgetter
from collections import namedtuple
from collections.abc import Hashable, Sequence, Mapping

from unification import isvar, Var

from toolz import compose

import symbolic_pymc as sp


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


UnequalMetaParts = namedtuple("UnequalMetaParts", ["path", "reason", "objects"])


def meta_diff_seq(x, y, loc, path, is_map=False, **kwargs):
    if len(x) != len(y):
        return (path, f"{loc} len", (x, y))
    else:
        for i, (a, b) in enumerate(zip(x, y)):
            if is_map:
                if a[0] != b[0]:
                    return (path, "map keys", (x, y))
                this_path = compose(itemgetter(a[0]), path)
                a, b = a[1], b[1]
            else:
                this_path = compose(itemgetter(i), path)

            z = meta_diff(a, b, path=this_path, **kwargs)
            if z is not None:
                return z


def meta_diff(x, y, pdb=False, ne_fn=ne, cmp_types=True, path=compose()):
    """Traverse meta objects and return information about the first pair of elements that are not equal.

    Returns a `UnequalMetaParts` object containing the object path, reason for
    being unequal, and the unequal object pair; otherwise, `None`.
    """
    res = None
    if cmp_types and ne_fn(type(x), type(y)) is True:
        res = (path, "types", (x, y))
    elif isinstance(x, sp.meta.MetaSymbol):
        if ne_fn(x.base, y.base) is True:
            res = (path, "bases", (x.base, y.base))
        else:
            try:
                x_rands = x.rands
                y_rands = y.rands
            except NotImplementedError:
                pass
            else:

                path = compose(attrgetter("rands"), path)

                res = meta_diff_seq(
                    x_rands, y_rands, "rands", path, pdb=pdb, ne_fn=ne_fn, cmp_types=cmp_types
                )

    elif isinstance(x, Mapping) and isinstance(y, Mapping):

        x_ = sorted(x.items(), key=itemgetter(0))
        y_ = sorted(y.items(), key=itemgetter(0))

        res = meta_diff_seq(
            x_, y_, "map", path, is_map=True, pdb=pdb, ne_fn=ne_fn, cmp_types=cmp_types
        )

    elif (
        isinstance(x, Sequence)
        and isinstance(y, Sequence)
        and not isinstance(x, str)
        and not isinstance(y, str)
    ):

        res = meta_diff_seq(x, y, "seq", path, pdb=pdb, ne_fn=ne_fn, cmp_types=cmp_types)

    elif ne_fn(x, y) is True:
        res = (path, "ne_fn", (x, y))

    if res is not None:
        if pdb:  # pragma: no cover
            import pdb

            pdb.set_trace()
        return UnequalMetaParts(*res)


def lvar_ignore_ne(x, y):
    if (isvar(x) and isvar(y)) or (
        isinstance(x, type) and isinstance(y, type) and issubclass(x, Var) and issubclass(y, Var)
    ):
        return False
    else:
        return ne(x, y)


def eq_lvar(x, y):
    """Perform an equality check that considers all logic variables equal."""
    return meta_diff(x, y, ne_fn=lvar_ignore_ne) is None
