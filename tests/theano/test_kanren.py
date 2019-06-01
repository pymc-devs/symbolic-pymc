import pytest
import theano.tensor as tt

from kanren import run, eq, variables
from kanren.term import term, operator, arguments
from kanren.assoccomm import eq_assoc, eq_comm
from unification import var

from symbolic_pymc.theano.random_variables import MvNormalRV
from symbolic_pymc.theano.meta import mt
from symbolic_pymc.unify import etuple
from symbolic_pymc.theano.utils import graph_equal


@pytest.mark.usefixtures("run_with_theano")
def test_terms():
    x, a, b = tt.dvectors('xab')
    test_expr = x + a * b

    assert mt(test_expr.owner.op) == operator(test_expr)
    assert mt(tuple(test_expr.owner.inputs)) == tuple(arguments(test_expr))

    assert tuple(arguments(test_expr)) == mt(tuple(test_expr.owner.inputs))

    # Implicit `etuple` conversion should retain the original object
    # (within the implicitly introduced meta object, of course).
    assert test_expr == arguments(test_expr).orig_expr._eval_obj.obj

    assert graph_equal(test_expr, term(operator(test_expr),
                                       arguments(test_expr)))
    assert mt(test_expr) == term(operator(test_expr),
                                 arguments(test_expr))

    # Same here: should retain the original object.
    assert test_expr == term(operator(test_expr),
                             arguments(test_expr)).reify()


@pytest.mark.usefixtures("run_with_theano")
def test_kanren():
    # x, a, b = tt.dvectors('xab')
    #
    # with variables(x, a, b):
    #     assert b == run(1, x, eq(a + b, a + x))[0]
    #     assert b == run(1, x, eq(a * b, a * x))[0]

    a, b = mt.dvectors('ab')
    assert b == run(1, var('x'), eq(mt.add(a, b), mt.add(a, var('x'))))[0]
    assert b == run(1, var('x'), eq(mt.mul(a, b), mt.mul(a, var('x'))))[0]

    a_tt = tt.vector('a')
    R_tt = tt.matrix('R')
    F_t_tt = tt.matrix('F')
    V_tt = tt.matrix('V')
    beta_rv = MvNormalRV(a_tt, R_tt, name='\\beta')
    E_y_rv = F_t_tt.dot(beta_rv)
    Y_rv = MvNormalRV(E_y_rv, V_tt, name='y')

    beta_name_lv = var('beta_name')
    beta_size_lv = var('beta_size')
    beta_rng_lv = var('beta_rng')
    a_lv = var('a')
    R_lv = var('R')
    beta_prior_mt = mt.MvNormalRV(a_lv, R_lv,
                                  beta_size_lv, beta_rng_lv,
                                  name=beta_name_lv)

    y_name_lv = var('y_name')
    y_size_lv = var('y_size')
    y_rng_lv = var('y_rng')
    F_t_lv = var('f')
    V_lv = var('V')
    E_y_mt = mt.dot(F_t_lv, beta_prior_mt)

    Y_mt = mt.MvNormalRV(E_y_mt, V_lv,
                         y_size_lv, y_rng_lv,
                         name=y_name_lv)

    with variables(Y_mt):
        res, = run(0, Y_mt, (eq, Y_rv, Y_mt))
    assert res.reify() == Y_rv


@pytest.mark.usefixtures("run_with_theano")
def test_assoccomm():
    from kanren.assoccomm import buildo

    x, a, b, c = tt.dvectors('xabc')
    test_expr = x + 1
    q = var('q')

    assert q == run(1, q, buildo(tt.add, test_expr.owner.inputs, test_expr))[0]
    assert tt.add == run(1, q,
                         buildo(q, test_expr.owner.inputs, test_expr))[0].reify()
    assert graph_equal(tuple(test_expr.owner.inputs),
                       run(1, q, buildo(tt.add, q, test_expr))[0])

    assert (mt(a),) == run(0, var('x'),
                           (eq_comm, mt.mul(a, b), mt.mul(b, var('x'))))
    assert (mt(a),) == run(0, var('x'),
                           (eq_comm, mt.add(a, b), mt.add(b, var('x'))))

    res = run(0, var('x'),
              (eq_assoc,
               mt.add(a, b, c),
               mt.add(a, var('x'))))

    # TODO: `res[0]` should return `etuple`s.  Since `eq_assoc` effectively
    # picks apart the results of `arguments(...)`, I don't know if we can
    # keep the `etuple`s around.  We might be able to convert the results
    # to `etuple`s automatically by wrapping `eq_assoc`, though.
    assert etuple(*res[0]).eval_obj == mt(b + c)

    res = run(0, var('x'),
              (eq_assoc,
               mt.mul(a, b, c),
               mt.mul(a, var('x'))))
    assert etuple(*res[0]).eval_obj == mt(b * c)
