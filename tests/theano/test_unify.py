import pytest
import theano.tensor as tt

from operator import add

from unification import unify, reify, var, variables

from kanren.term import term, operator, arguments

from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.utils import graph_equal
from symbolic_pymc.unify import (ExpressionTuple, etuple, tuple_expression)


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


def test_etuple():
    """Test basic `etuple` functionality.
    """
    def test_op(*args):
        return tuple(object() for i in range(sum(args)))

    e1 = etuple(test_op, 1, 2)

    assert not hasattr(e1, '_eval_obj')

    with pytest.raises(ValueError):
        e1.eval_obj = 1

    e1_obj = e1.eval_obj
    assert len(e1_obj) == 3
    assert all(type(o) == object for o in e1_obj)

    # Make sure we don't re-create the cached `eval_obj`
    e1_obj_2 = e1.eval_obj
    assert e1_obj == e1_obj_2

    # Confirm that evaluation is recursive
    e2 = etuple(add, (object(),), e1)

    # Make sure we didn't convert this single tuple value to
    # an `etuple`
    assert type(e2[1]) == tuple

    # Slices should be `etuple`s, though.
    assert isinstance(e2[:1], ExpressionTuple)
    assert e2[1] == e2[1:2][0]

    e2_obj = e2.eval_obj

    assert type(e2_obj) == tuple
    assert len(e2_obj) == 4
    assert all(type(o) == object for o in e2_obj)
    # Make sure that it used `e1`'s original `eval_obj`
    assert e2_obj[1:] == e1_obj

    # Confirm that any combination of `tuple`s/`etuple`s in
    # concatenation result in an `etuple`
    e_radd = (1,) + etuple(2, 3)
    assert isinstance(e_radd, ExpressionTuple)
    assert e_radd == (1, 2, 3)

    e_ladd = etuple(1, 2) + (3,)
    assert isinstance(e_ladd, ExpressionTuple)
    assert e_ladd == (1, 2, 3)


def test_etuple_term():
    """Test `tuple_expression` and `etuple` interaction with `term`
    """
    # Make sure that we don't lose underlying `eval_obj`s
    # when taking apart and re-creating expression tuples
    # using `kanren`'s `operator`, `arguments` and `term`
    # functions.
    e1 = etuple(add, (object(),), (object(),))
    e1_obj = e1.eval_obj

    e1_dup = (operator(e1),) + arguments(e1)

    assert isinstance(e1_dup, ExpressionTuple)
    assert e1_dup.eval_obj == e1_obj

    e1_dup_2 = term(operator(e1), arguments(e1))
    assert e1_dup_2 == e1_obj

    # Take apart an already constructed/evaluated meta
    # object.
    e2 = mt.add(mt.vector(), mt.vector())

    e2_et = tuple_expression(e2)

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
    e3 = tuple_expression(mt_expr)
    assert e3 == e2_et
    assert e3.eval_obj is mt_expr
    assert e3.eval_obj.reify() is tt_expr

    # Now, through `tuple_expression`
    e2_et_2 = tuple_expression(tt_expr)
    assert e2_et_2 == e3 == e2_et
    assert isinstance(e2_et_2, ExpressionTuple)
    assert e2_et_2.eval_obj.reify() == tt_expr
