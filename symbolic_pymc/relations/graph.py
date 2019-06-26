from functools import partial
from unification import var

from kanren import eq
from kanren.cons import is_cons, is_null
from kanren.core import condeseq
from kanren.goals import conso, fail

from ..etuple import etuplize


def lapply_anyo(relation, l_in, l_out, i_any=False):
    """Apply a relation to at least one pair of corresponding elements in two sequences."""

    l_car, l_cdr = var(), var()
    o_car, o_cdr = var(), var()
    # o_any = var()

    # We need the `cons` types to match in the end, which involves
    # using the same `cons`-null (i.e. terminating `cdr`).
    null_type = None
    if is_cons(l_out) or is_null(l_out):
        null_type = type(l_out)()
    if is_cons(l_in) or is_null(l_in):
        _null_type = type(l_in)()
        if null_type is not None and not null_type == _null_type:
            return fail
        else:
            null_type = _null_type

    return (
        condeseq,
        [
            [
                (conso, l_car, l_cdr, l_in),
                (conso, o_car, o_cdr, l_out),
                (
                    condeseq,
                    [
                        [
                            (relation, l_car, o_car),
                            # (eq, o_any, True),
                            (lapply_anyo, relation, l_cdr, o_cdr, True),
                        ],
                        [
                            (eq, l_car, o_car),
                            # (eq, o_any, i_any),
                            (lapply_anyo, relation, l_cdr, o_cdr, i_any),
                        ],
                    ],
                ),
                # (lapply_anyo, relation, l_cdr, o_cdr, o_any),
            ],
            [(eq, i_any, True), (eq, l_in, null_type), (eq, l_in, l_out)],
        ],
    )


def reduceo(relation, in_expr, out_expr):
    """Relate a term and the fixed-point of that term under a given relation.

    This includes the "identity" relation.
    """
    expr_rdcd = var()
    return (
        condeseq,
        [
            # The fixed-point is another reduction step out.
            [(relation, in_expr, expr_rdcd), (reduceo, relation, expr_rdcd, out_expr)],
            # The fixed-point is a single-step reduction.
            [(relation, in_expr, out_expr)],
        ],
    )


def graph_applyo(relation, in_graph, out_graph, preprocess_graph=partial(etuplize, shallow=True)):
    """Relate the fixed-points of two term-graphs under a given relation.

    Parameters
    ----------
    relation: callable
      A relation to apply on a graph and its subgraphs.
    in_graph: object
      The graph for which the left-hand side of a binary relation holds.
    out_graph: object
      The graph for which the right-hand side of a binary relation holds.
    preprocess_graph: callable
      A unary function that produces an iterable upon which `lapply_anyo`
      can be applied in order to traverse a graph's subgraphs.  The default
      function converts the graph to expression-tuple form.
    """
    in_rdc = var()

    if preprocess_graph in (False, None):

        def preprocess_graph(x):
            return x

    _gapplyo = partial(graph_applyo, relation, preprocess_graph=preprocess_graph)

    return (
        condeseq,
        [
            [
                (relation, in_graph, in_rdc),
                (condeseq, [[(_gapplyo, in_rdc, out_graph)], [(eq, in_rdc, out_graph)]]),
            ],
            [
                (lapply_anyo, _gapplyo, preprocess_graph(in_graph), in_rdc),
                (condeseq, [[(_gapplyo, in_rdc, out_graph)], [(eq, in_rdc, out_graph)]]),
            ],
        ],
    )
