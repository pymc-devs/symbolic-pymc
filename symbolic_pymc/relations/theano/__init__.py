import numpy as np
import theano.tensor as tt

from unification import var
from unification.utils import transitive_get as walk

from kanren import eq
from kanren.core import lall, Zzz
from kanren.facts import fact
from kanren.graph import applyo, walko
from kanren.assoccomm import commutative, associative
from kanren.constraints import neq

from etuples import etuple

from ...utils import HashableNDArray
from ...theano.meta import TheanoMetaConstant, mt

# Establish some Theano `Op`s as commutative and/or associative
fact(commutative, mt.add)
fact(commutative, mt.mul)
fact(associative, mt.add)
fact(associative, mt.mul)


def constant_neq(lvar, val):
    """Assert that a constant graph variable is not equal to a specific value.

    Scalar values are broadcast across arrays.

    XXX: This goal is non-relational.
    TODO: Rewrite this as an actual constraint.
    """

    if isinstance(val, np.ndarray):
        val = val.view(HashableNDArray)

    def constant_neq_goal(S):
        lvar_rf = walk(lvar, S)
        if isinstance(lvar_rf, (tt.Constant, TheanoMetaConstant)):
            # Although `neq` is an actual constraint, the preceding type check
            # and alternative success--when not a constant type--make the
            # entire goal non-relational/not a true constraint, since the
            # aforementioned check will only occur once per goal-stream/state
            # and never again.
            yield from neq(lvar_rf.data, val)(S)
        else:
            # When the value isn't a Theano constant, consider it satisfied
            yield S

    return constant_neq_goal


def non_obs_walko(relation, a, b):
    """Construct a goal that applies a relation to all nodes above an observed random variable.

    This is useful if you don't want to apply relations to an observed random
    variable, but you do want to apply them to every term above one and
    ultimately reproduce the entire graph (observed RV included).

    Parameters
    ----------
    relation: function
      A binary relation/goal constructor function
    a: lvar or meta graph
      The left-hand side of the relation.
    b: lvar or meta graph
      The right-hand side of the relation

    """
    obs_lv, obs_rv_lv = var(), var()
    rv_op_lv, rv_args_lv, obs_rv_lv = var(), var(), var()
    new_rv_args_lv, new_obs_rv_lv = var(), var()

    return lall(
        # Indicate the observed term (i.e. observation and RV)
        eq(a, mt.observed(obs_lv, obs_rv_lv)),
        # Deconstruct the observed random variable
        applyo(rv_op_lv, rv_args_lv, obs_rv_lv),
        # Apply relation to the RV's inputs
        Zzz(walko, relation, rv_args_lv, new_rv_args_lv),
        # map_anyo(partial(walko, relation), rv_args_lv, new_rv_args_lv),
        # Reconstruct the random variable
        applyo(rv_op_lv, new_rv_args_lv, new_obs_rv_lv),
        # Reconstruct the observation
        applyo(mt.observed, etuple(obs_lv, new_obs_rv_lv), b),
    )
