import pytest

import theano.tensor as tt

from unification import unify, reify, var, variables

from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.utils import graph_equal
from symbolic_pymc.etuple import (ExpressionTuple, etuple, etuplize)


@pytest.mark.usefixtures("run_with_theano")
def test_unification():
    x, y, a, b = tt.dvectors('xyab')
    x_s = tt.scalar('x_s')
    y_s = tt.scalar('y_s')
    c_tt = tt.constant(1, 'c')
    d_tt = tt.constant(2, 'd')

    x_l = var('x_l')
    y_l = var('y_l')

    assert a == reify(x_l, {x_l: a}).reify()
    test_expr = mt.add(1, mt.mul(2, x_l))
    test_reify_res = reify(test_expr, {x_l: a})
    assert graph_equal(test_reify_res.reify(), 1 + 2*a)

    z = tt.add(b, a)
    assert {x_l: z} == unify(x_l, z)
    assert b == unify(mt.add(x_l, a), mt.add(b, a))[x_l].reify()

    res = unify(mt.inv(mt.add(x_l, a)), mt.inv(mt.add(b, y_l)))
    assert res[x_l].reify() == b
    assert res[y_l].reify() == a

    # TODO: This produces a `DimShuffle` so that the scalar constant `1`
    # will match the dimensions of the vector `b`.  That `DimShuffle` isn't
    # handled by the logic variable form.
    # assert unify(mt.add(x_l, 1), mt.add(b_l, 1))[x] == b

    with variables(x):
        assert unify(x + 1, b + 1)[x].reify() == b

    assert unify(mt.add(x_l, a), mt.add(b, a))[x_l].reify() == b
    with variables(x):
        assert unify(x, b)[x] == b
        assert unify([x], [b])[x] == b
        assert unify((x,), (b,))[x] == b
        assert unify(x + 1, b + 1)[x].reify() == b
        assert unify(x + a, b + a)[x].reify() == b

    with variables(x):
        assert unify(a + b, a + x)[x].reify() == b

    mt_expr_add = mt.add(x_l, y_l)

    # The parameters are vectors
    tt_expr_add_1 = tt.add(x, y)
    assert graph_equal(tt_expr_add_1,
                       reify(mt_expr_add,
                             unify(mt_expr_add, tt_expr_add_1)).reify())

    # The parameters are scalars
    tt_expr_add_2 = tt.add(x_s, y_s)
    assert graph_equal(tt_expr_add_2,
                       reify(mt_expr_add,
                             unify(mt_expr_add, tt_expr_add_2)).reify())

    # The parameters are constants
    tt_expr_add_3 = tt.add(c_tt, d_tt)
    assert graph_equal(tt_expr_add_3,
                       reify(mt_expr_add,
                             unify(mt_expr_add, tt_expr_add_3)).reify())


@pytest.mark.usefixtures("run_with_theano")
def test_etuple_term():
    """Test `etuplize` and `etuple` interaction with `term`
    """
    # Take apart an already constructed/evaluated meta
    # object.
    e2 = mt.add(mt.vector(), mt.vector())

    e2_et = etuplize(e2)

    assert isinstance(e2_et, ExpressionTuple)

    e2_et_expect = etuple(
        mt.add,
        etuple(mt.TensorVariable,
               etuple(mt.TensorType,
                      'float64', (False,), None),
               None, None, None),
        etuple(mt.TensorVariable,
               etuple(mt.TensorType,
                      'float64', (False,), None),
               None, None, None),
    )
    assert e2_et == e2_et_expect
    assert e2_et.eval_obj is e2

    # Make sure expression expansion works from Theano objects, too.
    # First, do it manually.
    tt_expr = tt.vector() + tt.vector()

    mt_expr = mt(tt_expr)
    assert mt_expr.obj is tt_expr
    assert mt_expr.reify() is tt_expr
    e3 = etuplize(mt_expr)
    assert e3 == e2_et
    assert e3.eval_obj is mt_expr
    assert e3.eval_obj.reify() is tt_expr

    # Now, through `etuplize`
    e2_et_2 = etuplize(tt_expr)
    assert e2_et_2 == e3 == e2_et
    assert isinstance(e2_et_2, ExpressionTuple)
    assert e2_et_2.eval_obj == tt_expr
