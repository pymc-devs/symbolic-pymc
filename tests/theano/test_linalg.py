import pytest

import theano
import theano.tensor as tt

import numpy as np

from theano.gof.graph import inputs as tt_inputs

from symbolic_pymc.theano.random_variables import NormalRV, observed
from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.opt import eval_and_reify_meta, FunctionGraph
from symbolic_pymc.unify import (etuple, tuple_expression)
from symbolic_pymc.relations.theano.linalg import (normal_normal_regression, buildo,
                                                   normal_qr_transform)

from kanren import run, eq, var


@pytest.mark.usefixtures("run_with_theano")
@pytest.mark.xfail(strict=True)
def test_normal_normal_regression():
    tt.config.compute_test_value = 'ignore'
    theano.config.cxx = ''
    np.random.seed(9283)

    N = 10
    M = 3
    a_tt = tt.vector('a')
    R_tt = tt.vector('R')
    X_tt = tt.matrix('X')
    V_tt = tt.vector('V')

    a_tt.tag.test_value = np.random.normal(size=M)
    R_tt.tag.test_value = np.abs(np.random.normal(size=M))
    X = np.random.normal(10, 1, size=N)
    X = np.c_[np.ones(10), X, X * X]
    X_tt.tag.test_value = X
    V_tt.tag.test_value = np.ones(N)

    beta_rv = NormalRV(a_tt, R_tt, name='\\beta')

    E_y_rv = X_tt.dot(beta_rv)
    E_y_rv.name = 'E_y'
    Y_rv = NormalRV(E_y_rv, V_tt, name='Y')

    y_tt = tt.as_tensor_variable(Y_rv.tag.test_value)
    y_tt.name = 'y'
    y_obs_rv = observed(y_tt, Y_rv)
    y_obs_rv.name = 'y_obs'

    fgraph = FunctionGraph(tt_inputs([beta_rv, y_obs_rv]),
                           [y_obs_rv])

    #
    # Use the relation with identify/match `Y`, `X` and `beta`.
    #
    b_lv = var()
    y_args_tail_lv, b_args_tail_lv = var(), var()
    res, = run(1, (b_lv, y_args_tail_lv, b_args_tail_lv),
               (buildo, mt.observed, var('y_args'), y_obs_rv),
               (eq, var('y_args'), (var('y'), var('Y'))),
               normal_normal_regression(
                   var('Y'),
                   var('X'),
                   b_lv,
                   y_args_tail_lv,
                   b_args_tail_lv)
    )

    assert res[0].eval_obj.obj == beta_rv
    assert res[0] == tuple_expression(beta_rv)
    assert res[1] == tuple_expression(Y_rv)[2:]
    assert res[2] == tuple_expression(beta_rv)[1:]

    #
    # Use the relation with to produce `Y` from given `X` and `beta`.
    #
    X_new_mt = mt(tt.eye(N, M))
    beta_new_mt = mt(NormalRV(0, 1, size=M))
    Y_args_cdr_mt = tuple_expression(Y_rv)[2:]
    Y_lv = var()
    res, = run(1, Y_lv,
               normal_normal_regression(
                   Y_lv,
                   X_new_mt,
                   beta_new_mt,
                   Y_args_cdr_mt))
    Y_out_mt = res.eval_obj

    Y_new_mt = (etuple(mt.NormalRV, mt.dot(X_new_mt, beta_new_mt)) +
                Y_args_cdr_mt)
    Y_new_mt = Y_new_mt.eval_obj

    assert Y_out_mt == Y_new_mt


@pytest.mark.usefixtures("run_with_theano")
@pytest.mark.xfail(strict=True)
def test_normal_qr_transform():
    np.random.seed(9283)

    N = 10
    M = 3
    X_tt = tt.matrix('X')
    X = np.random.normal(10, 1, size=N)
    X = np.c_[np.ones(10), X, X * X]
    X_tt.tag.test_value = X

    V_tt = tt.vector('V')
    V_tt.tag.test_value = np.ones(N)

    a_tt = tt.vector('a')
    R_tt = tt.vector('R')
    a_tt.tag.test_value = np.random.normal(size=M)
    R_tt.tag.test_value = np.abs(np.random.normal(size=M))

    beta_rv = NormalRV(a_tt, R_tt, name='\\beta')

    E_y_rv = X_tt.dot(beta_rv)
    E_y_rv.name = 'E_y'
    Y_rv = NormalRV(E_y_rv, V_tt, name='Y')

    y_tt = tt.as_tensor_variable(Y_rv.tag.test_value)
    y_tt.name = 'y'
    y_obs_rv = observed(y_tt, Y_rv)
    y_obs_rv.name = 'y_obs'

    fgraph = FunctionGraph(tt_inputs([beta_rv, y_obs_rv]),
                           [y_obs_rv])

    res, = run(1, var('q'), normal_qr_transform(y_obs_rv, var('q')))

    new_node = {eval_and_reify_meta(k): eval_and_reify_meta(v)
                for k, v in res}

    # Make sure the old-to-new `beta` conversion is correct.
    t_Q, t_R = np.linalg.qr(X)
    Coef_new_value = np.linalg.inv(t_R)
    np.testing.assert_array_almost_equal(
        Coef_new_value,
        new_node[beta_rv].owner.inputs[0].tag.test_value)

    # Make sure the new `beta_tilde` has the right standard normal distribution
    # parameters.
    beta_tilde_node = new_node[beta_rv].owner.inputs[1]
    np.testing.assert_array_almost_equal(
        np.r_[0., 0., 0.],
        beta_tilde_node.owner.inputs[0].tag.test_value)
    np.testing.assert_array_almost_equal(
        np.r_[1., 1., 1.],
        beta_tilde_node.owner.inputs[1].tag.test_value)

    Y_new = new_node[y_obs_rv].owner.inputs[1]
    assert Y_new.owner.inputs[0].owner.inputs[1] == beta_tilde_node

    np.testing.assert_array_almost_equal(
        t_Q,
        Y_new.owner.inputs[0].owner.inputs[0].tag.test_value)
