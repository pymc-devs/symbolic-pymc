import pytest

import numpy as np

import theano
import theano.tensor as tt

from unification import reify, unify, var

from cons.core import ConsError

from etuples import etuple, etuplize, rator, rands
from etuples.core import ExpressionTuple

from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.utils import graph_equal
from symbolic_pymc.theano.random_variables import MvNormalRV


@pytest.mark.usefixtures("run_with_theano")
def test_unification():
    x, y, a, b = tt.dvectors("xyab")
    x_s = tt.scalar("x_s")
    y_s = tt.scalar("y_s")
    c_tt = tt.constant(1, "c")
    d_tt = tt.constant(2, "d")

    x_l = var("x_l")
    y_l = var("y_l")

    assert a == reify(x_l, {x_l: a}).reify()
    test_expr = mt.add(1, mt.mul(2, x_l))
    test_reify_res = reify(test_expr, {x_l: a})
    assert graph_equal(test_reify_res.reify(), 1 + 2 * a)

    z = tt.add(b, a)
    assert {x_l: z} == unify(x_l, z)
    assert b == unify(mt.add(x_l, a), mt.add(b, a))[x_l].reify()

    res = unify(mt.inv(mt.add(x_l, a)), mt.inv(mt.add(b, y_l)))
    assert res[x_l].reify() == b
    assert res[y_l].reify() == a

    mt_expr_add = mt.add(x_l, y_l)

    # The parameters are vectors
    tt_expr_add_1 = tt.add(x, y)
    assert graph_equal(tt_expr_add_1, reify(mt_expr_add, unify(mt_expr_add, tt_expr_add_1)).reify())

    # The parameters are scalars
    tt_expr_add_2 = tt.add(x_s, y_s)
    assert graph_equal(tt_expr_add_2, reify(mt_expr_add, unify(mt_expr_add, tt_expr_add_2)).reify())

    # The parameters are constants
    tt_expr_add_3 = tt.add(c_tt, d_tt)
    assert graph_equal(tt_expr_add_3, reify(mt_expr_add, unify(mt_expr_add, tt_expr_add_3)).reify())


@pytest.mark.usefixtures("run_with_theano")
def test_etuple_term():
    """Test `etuplize` and `etuple` interaction with `term`."""
    # Take apart an already constructed/evaluated meta
    # object.
    e2 = mt.add(mt.vector(), mt.vector())

    e2_et = etuplize(e2)

    assert isinstance(e2_et, ExpressionTuple)

    # e2_et_expect = etuple(
    #     mt.add,
    #     etuple(mt.TensorVariable,
    #            etuple(mt.TensorType,
    #                   'float64', (False,), None),
    #            None, None, None),
    #     etuple(mt.TensorVariable,
    #            etuple(mt.TensorType,
    #                   'float64', (False,), None),
    #            None, None, None),
    # )
    e2_et_expect = etuple(mt.add, e2.base_arguments[0], e2.base_arguments[1])
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

    test_expr = mt(tt.vector("z") * 7)
    assert rator(test_expr) == mt.mul
    assert rands(test_expr)[0] == mt(tt.vector("z"))

    dim_shuffle_op = rator(rands(test_expr)[1])

    assert isinstance(dim_shuffle_op, mt.DimShuffle)
    assert rands(rands(test_expr)[1]) == etuple(mt(7))

    with pytest.raises(ConsError):
        rator(dim_shuffle_op)
    # assert rator(dim_shuffle_op) == mt.DimShuffle
    # assert rands(dim_shuffle_op) == etuple((), ("x",), True)

    const_tensor = rands(rands(test_expr)[1])[0]
    with pytest.raises(ConsError):
        rator(const_tensor)
    with pytest.raises(ConsError):
        rands(const_tensor)

    et_expr = etuplize(test_expr)
    exp_res = etuple(
        mt.mul,
        mt(tt.vector("z")),
        etuple(mt.DimShuffle((), ("x",), True), mt(7))
        # etuple(etuple(mt.DimShuffle, (), ("x",), True), mt(7))
    )

    assert et_expr == exp_res
    assert exp_res.eval_obj == test_expr


@pytest.mark.usefixtures("run_with_theano")
def test_unify_rvs():

    a_tt = tt.vector("a")
    R_tt = tt.matrix("R")
    F_t_tt = tt.matrix("F")
    V_tt = tt.matrix("V")
    beta_rv = MvNormalRV(a_tt, R_tt, name="\\beta")
    E_y_rv = F_t_tt.dot(beta_rv)
    Y_rv = MvNormalRV(E_y_rv, V_tt, name="y")

    E_y_lv, V_lv, Y_name_lv = var(), var(), var()
    Y_lv = mt.MvNormalRV(E_y_lv, V_lv, size=var(), rng=var(), name=Y_name_lv)

    s = unify(Y_lv, Y_rv)

    assert s[E_y_lv].reify() == E_y_rv
    assert s[V_lv].reify() == V_tt
    assert s[Y_name_lv] == "y"


@pytest.mark.usefixtures("run_with_theano")
def test_unify_ops():
    def f_pow2(x_tm1):
        return 2 * x_tm1

    state = theano.tensor.scalar("state")
    n_steps = theano.tensor.iscalar("nsteps")
    output, updates = theano.scan(
        f_pow2, [], state, [], n_steps=n_steps, truncate_gradient=-1, go_backwards=False
    )

    assert np.array_equal(output.eval({state: 1.0, n_steps: 4}), np.r_[2.0, 4.0, 8.0, 16.0])

    scan_tt = output.owner.inputs[0].owner.op

    inputs_lv, outputs_lv, info_lv = var(), var(), var()
    scan_lv = mt.Scan(inputs_lv, outputs_lv, info_lv)

    s = unify(scan_lv, scan_tt, {})

    assert s is not False
    assert s[inputs_lv] is scan_tt.inputs

    s_new = s.copy()
    s_new[outputs_lv] = [5 * s_new[inputs_lv][0]]

    new_scan_mt = reify(scan_lv, s_new)

    output_mt = mt(output)
    output_mt.owner.inputs[0].owner.op = new_scan_mt
    output_mt.owner.inputs[0].reset()
    output_mt.owner.outputs[0].reset()
    output_mt.owner.reset()
    output_mt.reset()
    assert output_mt.obj is not output

    output_new = output_mt.reify()

    assert output_new != output

    assert np.array_equal(output_new.eval({state: 1.0, n_steps: 4}), np.r_[5.0, 25.0, 125.0, 625.0])
