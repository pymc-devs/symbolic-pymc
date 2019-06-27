from operator import add, mul
from math import log, exp

import pytest

from unification import var

from kanren import run, eq
from kanren.core import condeseq

from symbolic_pymc.etuple import etuple
from symbolic_pymc.relations.graph import reduceo, lapply_anyo, graph_applyo


def reduces(in_expr, out_expr):
    """Create a relation for a couple math-based identities."""
    x_lv = var()
    return (condeseq, [
        [(eq, in_expr, etuple(add, x_lv, x_lv)),
         (eq, out_expr, etuple(mul, 2, x_lv))],
        [(eq, in_expr, etuple(log, etuple(exp, x_lv))),
         (eq, out_expr, x_lv)],
    ])


def math_reduceo(a, b):
    """Produce all results for repeated applications of the math-based relation."""
    return (reduceo, reduces, a, b)


def test_lapply_anyo_types():
    """Make sure that `applyo` preserves the types between its arguments."""
    q_lv = var()
    res = run(1, q_lv, (lapply_anyo, lambda x, y: (eq, x, y), [1], q_lv))
    assert res[0] == [1]
    res = run(1, q_lv, (lapply_anyo, lambda x, y: (eq, x, y), (1,), q_lv))
    assert res[0] == (1,)
    res = run(1, q_lv, (lapply_anyo, lambda x, y: (eq, x, y), etuple(1,), q_lv))
    assert res[0] == etuple(1,)
    res = run(1, q_lv, (lapply_anyo, lambda x, y: (eq, x, y), q_lv, (1,)))
    assert res[0] == (1,)
    res = run(1, q_lv, (lapply_anyo, lambda x, y: (eq, x, y), q_lv, [1]))
    assert res[0] == [1]
    res = run(1, q_lv, (lapply_anyo, lambda x, y: (eq, x, y), q_lv, etuple(1)))
    assert res[0] == etuple(1)


@pytest.mark.parametrize(
    'test_input, test_output',
    [([], ()),
     ([1], ()),
     ([etuple(add, 1, 1),],
      ([etuple(mul, 2, 1)],)),
     ([1, etuple(add, 1, 1)],
      ([1, etuple(mul, 2, 1)],)),
     ([etuple(add, 1, 1), 1],
      ([etuple(mul, 2, 1), 1],)),
     ([etuple(mul, 2, 1), etuple(add, 1, 1), 1],
      ([etuple(mul, 2, 1), etuple(mul, 2, 1), 1],)),
     ([etuple(add, 1, 1), etuple(log, etuple(exp, 5)),],
      ([etuple(mul, 2, 1), 5],
       [etuple(add, 1, 1), 5],
       [etuple(mul, 2, 1), etuple(log, etuple(exp, 5))]))])
def test_lapply_anyo(test_input, test_output):
    """Test `lapply_anyo` with fully ground terms (i.e. no logic variables)."""
    q_lv = var()
    test_res = run(0, q_lv,
                   (lapply_anyo, math_reduceo, test_input, q_lv))

    assert len(test_res) == len(test_output)

    # Make sure the first result matches.
    # TODO: This is fairly implementation-specific (i.e. dependent on the order
    # in which `condeseq` returns results).
    if len(test_output) > 0:
        assert test_res[0] == test_output[0]

    # Make sure all the results match.
    # TODO: If we want to avoid fixing the output order, convert the lists to
    # tuples and add everything to a set, then compare.
    assert test_res == test_output


def test_lapply_anyo_reverse():
    """Test `lapply_anyo` in "reverse" (i.e. specify the reduced form and generate the un-reduced form)."""
    # Unbounded reverse
    q_lv = var()
    rev_input = [etuple(mul, 2, 1)]
    test_res = run(4, q_lv, (lapply_anyo, reduces, q_lv, rev_input))
    assert test_res == ([etuple(add, 1, 1)],
                        [etuple(log, etuple(exp, etuple(mul, 2, 1)))])

    # Guided reverse
    test_res = run(4, q_lv,
                   (lapply_anyo, reduces,
                    [etuple(add, q_lv, 1)],
                    [etuple(mul, 2, 1)]))

    assert test_res == (1,)


@pytest.mark.parametrize(
    'test_input, test_output',
    [(1, ()),
     (etuple(add, 1, 1),
      (etuple(mul, 2, 1),)),
     (etuple(add, etuple(mul, 2, 1), etuple(add, 1, 1)),
      (etuple(mul, 2, etuple(mul, 2, 1)),
       etuple(add, etuple(mul, 2, 1), etuple(mul, 2, 1)))),
     (etuple(add, etuple(mul, etuple(log, etuple(exp, 2)), 1), etuple(add, 1, 1)),
      (etuple(mul, 2, etuple(mul, 2, 1)),
       etuple(add, etuple(mul, 2, 1), etuple(mul, 2, 1)),
       etuple(add, etuple(mul, etuple(log, etuple(exp, 2)), 1), etuple(mul, 2, 1)),
       etuple(add, etuple(mul, 2, 1), etuple(add, 1, 1))))])
def test_graph_applyo(test_input, test_output):
    """Test `graph_applyo` with fully ground terms (i.e. no logic variables)."""

    q_lv = var()
    test_res = run(len(test_output), q_lv,
                   graph_applyo(math_reduceo, test_input, q_lv,
                                preprocess_graph=None))

    assert len(test_res) == len(test_output)

    # Make sure the first result matches.
    if len(test_output) > 0:
        assert test_res[0] == test_output[0]

    # Make sure all the results match.
    assert set(test_res) == set(test_output)


@pytest.mark.skip('Not ready, yet.')
def test_graph_applyo_reverse(test_input, test_output):
    """Test `graph_applyo` in "reverse" (i.e. specify the reduced form and generate the un-reduced form)."""
    q_lv = var()
    test_res = run(1, q_lv, (graph_applyo, reduces, q_lv, 5))
