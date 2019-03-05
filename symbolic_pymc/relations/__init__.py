import numpy as np
import theano.tensor as tt

from kanren.facts import Relation
from kanren.goals import goalify

from unification.utils import transitive_get as walk

from ..theano.meta import TheanoMetaConstant


# Hierarchical models that we recognize.
hierarchical_model = Relation("hierarchical")

# Conjugate relationships
conjugate = Relation("conjugate")


concat = goalify(lambda *args: "".join(args))


def constant_neq(lvar, val):
    """
    Assert that a constant graph variable is not equal to a specific value.

    Scalar values are broadcast across arrays.

    """

    def _goal(s):
        lvar_val = walk(lvar, s)
        if isinstance(lvar_val, (tt.Constant, TheanoMetaConstant)):
            data = lvar_val.data
            if (isinstance(val, np.ndarray) and not np.array_equal(data, val)) or not all(
                np.atleast_1d(data) == val
            ):
                yield s
        else:
            yield s

    return _goal
