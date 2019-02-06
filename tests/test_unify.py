import theano
import theano.tensor as tt

# import symbolic_pymc3.unify # noqa

from unification import unify, reify, var, variables

from symbolic_pymc.meta import mt
from symbolic_pymc.utils import graph_equal


def test_unification():
    x, y, a, b = tt.dvectors('xyab')
    x_s = tt.scalar('x_s')
    y_s = tt.scalar('y_s')
    c_tt = tt.constant(1, 'c')
    d_tt = tt.constant(2, 'd')
    # x_l = tt.vector('x_l')
    # y_l = tt.vector('y_l')
    # z_l = tt.vector('z_l')

    x_l = var('x_l')
    y_l = var('y_l')
    z_l = var('z_l')

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
