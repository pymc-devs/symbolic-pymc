from functools import partial

from unification import var

from kanren import eq
from kanren.core import lall

from .linalg import buildo
from ..graph import graph_applyo, seq_apply_anyo
from ...etuple import etuplize, etuple
from ...theano.meta import mt


def tt_graph_applyo(relation, a, b, preprocess_graph=partial(etuplize, shallow=True)):
    """Construct a `graph_applyo` goal that judiciously expands a Theano meta graph.

    This essentially allows `graph_applyo` to walk a non-`etuple`/S-exp Theano
    meta graph.

    Parameters
    ----------
    relation: function
      A binary relation/goal constructor function
    a: lvar, meta graph, or etuple
      The left-hand side of the relation.
    b: lvar, meta graph, or etuple
      The right-hand side of the relation
    preprocess_graph: function
      See `graph_applyo`.

    """

    def _expand_some_nodes(node):
        if isinstance(node, (mt.Apply, mt.TensorVariable)) or (
            isinstance(node, mt.TensorVariable) and node.owner is not None
        ):
            res = preprocess_graph(node)
        else:
            res = node

        return res

    return graph_applyo(relation, a, b, preprocess_graph=_expand_some_nodes)


def non_obs_graph_applyo(relation, a, b):
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
        (buildo, rv_op_lv, rv_args_lv, obs_rv_lv),
        # Apply relation to the RV's inputs
        seq_apply_anyo(
            partial(tt_graph_applyo, relation), rv_args_lv, new_rv_args_lv, skip_op=False
        ),
        # Reconstruct the random variable
        (buildo, rv_op_lv, new_rv_args_lv, new_obs_rv_lv),
        # Reconstruct the observation
        (buildo, mt.observed, etuple(obs_lv, new_obs_rv_lv), b),
    )
