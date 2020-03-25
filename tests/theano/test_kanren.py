import pytest

import theano.tensor as tt

from unification import var

from etuples import etuple

from kanren import eq, run
from kanren.graph import applyo
from kanren.term import term, operator, arguments
from kanren.assoccomm import eq_assoc, eq_comm

from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.utils import graph_equal


@pytest.mark.usefixtures("run_with_theano")
def test_terms():
    x, a, b = tt.dvectors("xab")
    test_expr = x + a * b

    assert mt(test_expr.owner.op) == operator(test_expr)
    assert mt(tuple(test_expr.owner.inputs)) == tuple(arguments(test_expr))

    assert tuple(arguments(test_expr)) == mt(tuple(test_expr.owner.inputs))

    # Implicit `etuple` conversion should retain the original object
    # (within the implicitly introduced meta object, of course).
    assert test_expr == arguments(test_expr)._parent._eval_obj.obj

    assert graph_equal(test_expr, term(operator(test_expr), arguments(test_expr)))
    assert mt(test_expr) == term(operator(test_expr), arguments(test_expr))

    # Same here: should retain the original object.
    assert test_expr == term(operator(test_expr), arguments(test_expr)).reify()


@pytest.mark.usefixtures("run_with_theano")
def test_kanren_algebra():
    a, b = mt.dvectors("ab")
    assert b == run(1, var("x"), eq(mt.add(a, b), mt.add(a, var("x"))))[0]
    assert b == run(1, var("x"), eq(mt.mul(a, b), mt.mul(a, var("x"))))[0]


@pytest.mark.usefixtures("run_with_theano")
def test_assoccomm():
    x, a, b, c = tt.dvectors("xabc")
    test_expr = x + 1
    q = var()

    res = run(1, q, applyo(tt.add, etuple(*test_expr.owner.inputs), test_expr))
    assert q == res[0]

    res = run(1, q, applyo(q, etuple(*test_expr.owner.inputs), test_expr))
    assert tt.add == res[0].reify()

    res = run(1, q, applyo(tt.add, q, test_expr))
    assert mt(tuple(test_expr.owner.inputs)) == res[0]

    x = var()
    res = run(0, x, eq_comm(mt.mul(a, b), mt.mul(b, x)))
    assert (mt(a),) == res

    res = run(0, x, eq_comm(mt.add(a, b), mt.add(b, x)))
    assert (mt(a),) == res

    (res,) = run(0, x, eq_assoc(mt.add(a, b, c), mt.add(a, x)))
    assert res == mt(b + c)

    (res,) = run(0, x, eq_assoc(mt.mul(a, b, c), mt.mul(a, x)))
    assert res == mt(b * c)
