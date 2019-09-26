from functools import partial
from unification import var, isvar

from unification import reify
from kanren.core import goaleval

from kanren import eq
from cons.core import ConsPair, ConsNull
from kanren.core import conde, lall
from kanren.goals import conso, fail

from ..etuple import etuplize, etuple, ExpressionTuple


def lapply_anyo(relation, l_in, l_out, null_type=False, skip_op=True):
    """Apply a relation to at least one pair of corresponding elements in two sequences.

    Parameters
    ----------
    null_type: optional
       An object that's a valid cdr for the collection type desired.  If
       `False` (i.e. the default value), the cdr will be inferred from the
       inputs, or defaults to an empty list.
    skip_op: boolean (optional)
       When both inputs are `etuple`s and this value is `True`, the relation
       will not be applied to the operators (i.e. the cars) of the inputs.
    """

    def _lapply_anyo(relation, l_in, l_out, i_any, null_type, skip_cars=False):
        def _goal(s):

            nonlocal i_any, null_type

            l_in_rf, l_out_rf = reify((l_in, l_out), s)

            i_car, i_cdr = var(), var()
            o_car, o_cdr = var(), var()

            conde_branches = []
            if i_any or (isvar(l_in_rf) and isvar(l_out_rf)):
                conde_branches.append([eq(l_in_rf, null_type), eq(l_in_rf, l_out_rf)])

            descend_branch = [
                goaleval(conso(i_car, i_cdr, l_in_rf)),
                goaleval(conso(o_car, o_cdr, l_out_rf)),
            ]

            conde_branches.append(descend_branch)

            conde_2_branches = [
                [eq(i_car, o_car), _lapply_anyo(relation, i_cdr, o_cdr, i_any, null_type)]
            ]

            if not skip_cars:
                conde_2_branches.append(
                    [relation(i_car, o_car), _lapply_anyo(relation, i_cdr, o_cdr, True, null_type)]
                )

            descend_branch.append(conde(*conde_2_branches))

            g = conde(*conde_branches)
            g = goaleval(g)

            yield from g(s)

        return _goal

    def goal(s):

        nonlocal null_type, skip_op

        # We need the `cons` types to match in the end, which involves
        # using the same `cons`-null (i.e. terminating `cdr`).
        if null_type is False:
            l_out_, l_in_ = reify((l_out, l_in), s)

            out_null_type = False
            if isinstance(l_out_, (ConsPair, ConsNull)):
                out_null_type = type(l_out_)()

            in_null_type = False
            if isinstance(l_in_, (ConsPair, ConsNull)):
                in_null_type = type(l_in_)()

                if out_null_type is not False and not type(in_null_type) == type(out_null_type):
                    yield from fail(s)
                    return

            null_type = (
                out_null_type
                if out_null_type is not False
                else in_null_type
                if in_null_type is not False
                else []
            )

        g = _lapply_anyo(
            relation,
            l_in,
            l_out,
            False,
            null_type,
            skip_cars=isinstance(null_type, ExpressionTuple) and skip_op,
        )
        g = goaleval(g)

        yield from g(s)

    return goal


def reduceo(relation, in_expr, out_expr):
    """Relate a term and the fixed-point of that term under a given relation.

    This includes the "identity" relation.
    """
    expr_rdcd = var()
    return conde(
        # The fixed-point is another reduction step out.
        [(relation, in_expr, expr_rdcd), (reduceo, relation, expr_rdcd, out_expr)],
        # The fixed-point is a single-step reduction.
        [(relation, in_expr, out_expr)],
    )


def graph_applyo(
    relation,
    in_graph,
    out_graph,
    preprocess_graph=partial(etuplize, shallow=True, return_bad_args=True),
    inside=False,
):
    """Relate the fixed-points of two term-graphs under a given relation.

    Parameters
    ----------
    relation: callable
      A relation to apply on a graph and its subgraphs.
    in_graph: object
      The graph for which the left-hand side of a binary relation holds.
    out_graph: object
      The graph for which the right-hand side of a binary relation holds.
    preprocess_graph: callable (optional)
      A unary function that produces an iterable upon which `lapply_anyo`
      can be applied in order to traverse a graph's subgraphs.  The default
      function converts the graph to expression-tuple form.
    inside: boolean (optional)
      Process the graph or sub-graphs first.
    """

    if preprocess_graph in (False, None):

        def preprocess_graph(x):
            return x

    def _gapplyo(s):

        nonlocal in_graph, out_graph, inside

        in_rdc = var()

        in_graph_rf, out_graph_rf = reify((in_graph, out_graph), s)

        expanding = isvar(in_graph_rf)

        _gapply = partial(
            graph_applyo,
            relation,
            preprocess_graph=preprocess_graph,
            inside=inside,  # expanding and (True ^ inside)
        )

        # This goal reduces the entire graph
        graph_reduce_gl = (relation, in_graph_rf, in_rdc)

        # This goal reduces children/arguments of the graph
        subgraphs_reduce_gl = lapply_anyo(
            _gapply,
            preprocess_graph(in_graph_rf),
            in_rdc,
            null_type=etuple() if expanding else False,
        )

        # Take only one step (e.g. reduce the entire graph and/or its
        # arguments)
        reduce_once_gl = eq(in_rdc, out_graph_rf)

        # Take another reduction step on top of the one(s) we already did
        # (i.e. recurse)
        reduce_again_gl = _gapply(in_rdc, out_graph_rf)

        # We want the fixed-point value first, but that requires
        # some checks.
        if expanding:
            # When the un-reduced expression is a logic variable (i.e. we're
            # "expanding" expressions), we can't go depth first.
            # We need to draw the association between (i.e. unify) the reduced
            # and expanded expressions ASAP, in order to produce finite
            # expanded graphs first and yield results.
            g = conde(
                [reduce_once_gl, graph_reduce_gl],
                [reduce_again_gl, graph_reduce_gl],
                [reduce_once_gl, subgraphs_reduce_gl],
                [reduce_again_gl, subgraphs_reduce_gl],
            )
        else:
            # TODO: With an explicit simplification order, could we determine
            # whether or not simplifying the sub-expressions or the expression
            # itself is more efficient?
            g = lall(
                conde([graph_reduce_gl], [subgraphs_reduce_gl]),
                conde([reduce_again_gl], [reduce_once_gl]),
            )

        g = goaleval(g)
        yield from g(s)

    return _gapplyo
