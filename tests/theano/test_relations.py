import pytest

import numpy as np

import theano
import theano.tensor as tt

from functools import partial

from unification import var

from etuples import etuple, etuplize

from kanren import run, eq
from kanren.graph import reduceo, walko, applyo

from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.opt import eval_and_reify_meta
from symbolic_pymc.theano.random_variables import observed, NormalRV, HalfCauchyRV, MvNormalRV

from symbolic_pymc.relations.theano import non_obs_walko
from symbolic_pymc.relations.theano.conjugates import conjugate
from symbolic_pymc.relations.theano.distributions import scale_loc_transform, constant_neq
from symbolic_pymc.relations.theano.linalg import normal_normal_regression, normal_qr_transform


def test_constant_neq():
    q_lv = var()

    res = run(0, q_lv, eq(q_lv, mt(1)), constant_neq(q_lv, np.array(1.0)))
    assert not res

    # TODO: If `constant_neq` was a true constraint, this would work.
    # res = run(0, q_lv, constant_neq(q_lv, np.array(1.0)), eq(q_lv, mt(1)))
    # assert not res

    # TODO: If `constant_neq` was a true constraint, this would work.
    # res = run(0, q_lv, constant_neq(q_lv, np.array(1.0)), eq(q_lv, mt(2)))
    # assert res == (mt(2),)

    res = run(0, q_lv, eq(q_lv, mt(2)), constant_neq(q_lv, np.array(1.0)))
    assert res == (mt(2),)


def test_scale_loc_transform():
    tt.config.compute_test_value = "ignore"

    rand_state = theano.shared(np.random.RandomState())
    mu_a = NormalRV(0.0, 100 ** 2, name="mu_a", rng=rand_state)
    sigma_a = HalfCauchyRV(5, name="sigma_a", rng=rand_state)
    mu_b = NormalRV(0.0, 100 ** 2, name="mu_b", rng=rand_state)
    sigma_b = HalfCauchyRV(5, name="sigma_b", rng=rand_state)
    county_idx = np.r_[1, 1, 2, 3]
    # We want the following for a, b:
    # N(m, S) -> m + N(0, 1) * S
    a = NormalRV(mu_a, sigma_a, size=(len(county_idx),), name="a", rng=rand_state)
    b = NormalRV(mu_b, sigma_b, size=(len(county_idx),), name="b", rng=rand_state)
    radon_est = a[county_idx] + b[county_idx] * 7
    eps = HalfCauchyRV(5, name="eps", rng=rand_state)
    radon_like = NormalRV(radon_est, eps, name="radon_like", rng=rand_state)
    radon_like_rv = observed(tt.as_tensor_variable(np.r_[1.0, 2.0, 3.0, 4.0]), radon_like)

    q_lv = var()

    (expr_graph,) = run(
        1, q_lv, non_obs_walko(partial(reduceo, scale_loc_transform), radon_like_rv, q_lv)
    )

    radon_like_rv_opt = expr_graph.reify()

    assert radon_like_rv_opt.owner.op == observed

    radon_like_opt = radon_like_rv_opt.owner.inputs[1]
    radon_est_opt = radon_like_opt.owner.inputs[0]

    # These should now be `tt.add(mu_*, ...)` outputs.
    a_opt = radon_est_opt.owner.inputs[0].owner.inputs[0]
    b_opt = radon_est_opt.owner.inputs[1].owner.inputs[0].owner.inputs[0]
    # Make sure NormalRV gets replaced with an addition
    assert a_opt.owner.op == tt.add
    assert b_opt.owner.op == tt.add

    # Make sure the first term in the addition is the old NormalRV mean
    mu_a_opt = a_opt.owner.inputs[0].owner.inputs[0]
    assert "mu_a" == mu_a_opt.name == mu_a.name
    mu_b_opt = b_opt.owner.inputs[0].owner.inputs[0]
    assert "mu_b" == mu_b_opt.name == mu_b.name

    # Make sure the second term in the addition is the standard NormalRV times
    # the old std. dev.
    assert a_opt.owner.inputs[1].owner.op == tt.mul
    assert b_opt.owner.inputs[1].owner.op == tt.mul

    sigma_a_opt = a_opt.owner.inputs[1].owner.inputs[0].owner.inputs[0]
    assert sigma_a_opt.owner.op == sigma_a.owner.op
    sigma_b_opt = b_opt.owner.inputs[1].owner.inputs[0].owner.inputs[0]
    assert sigma_b_opt.owner.op == sigma_b.owner.op

    a_std_norm_opt = a_opt.owner.inputs[1].owner.inputs[1]
    assert a_std_norm_opt.owner.op == NormalRV
    assert a_std_norm_opt.owner.inputs[0].data == 0.0
    assert a_std_norm_opt.owner.inputs[1].data == 1.0
    b_std_norm_opt = b_opt.owner.inputs[1].owner.inputs[1]
    assert b_std_norm_opt.owner.op == NormalRV
    assert b_std_norm_opt.owner.inputs[0].data == 0.0
    assert b_std_norm_opt.owner.inputs[1].data == 1.0


def test_mvnormal_conjugate():
    """Test that we can produce the closed-form distribution for the conjugate
    multivariate normal-regression with normal-prior model.
    """
    # import symbolic_pymc.theano.meta as tm
    #
    # tm.load_dispatcher()

    tt.config.cxx = ""
    tt.config.compute_test_value = "ignore"

    a_tt = tt.vector("a")
    R_tt = tt.matrix("R")
    F_t_tt = tt.matrix("F")
    V_tt = tt.matrix("V")

    a_tt.tag.test_value = np.r_[1.0, 0.0]
    R_tt.tag.test_value = np.diag([10.0, 10.0])
    F_t_tt.tag.test_value = np.c_[-2.0, 1.0]
    V_tt.tag.test_value = np.diag([0.5])

    beta_rv = MvNormalRV(a_tt, R_tt, name="\\beta")

    E_y_rv = F_t_tt.dot(beta_rv)
    Y_rv = MvNormalRV(E_y_rv, V_tt, name="Y")

    y_tt = tt.as_tensor_variable(np.r_[-3.0])
    y_tt.name = "y"
    Y_obs = observed(y_tt, Y_rv)

    q_lv = var()

    (expr_graph,) = run(1, q_lv, walko(conjugate, Y_obs, q_lv))

    fgraph_opt = expr_graph.eval_obj
    fgraph_opt_tt = fgraph_opt.reify()

    # Check that the SSE has decreased from prior to posterior.
    # TODO: Use a better test.
    beta_prior_mean_val = a_tt.tag.test_value
    F_val = F_t_tt.tag.test_value
    beta_post_mean_val = fgraph_opt_tt.owner.inputs[0].tag.test_value
    priorp_err = np.square(y_tt.data - F_val.dot(beta_prior_mean_val)).sum()
    postp_err = np.square(y_tt.data - F_val.dot(beta_post_mean_val)).sum()

    # First, make sure the prior and posterior means are simply not equal.
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(priorp_err, postp_err)

    # Now, make sure there's a decrease (relative to the observed point).
    np.testing.assert_array_less(postp_err, priorp_err)


@pytest.mark.xfail(strict=True)
def test_normal_normal_regression():
    tt.config.compute_test_value = "ignore"
    theano.config.cxx = ""
    np.random.seed(9283)

    N = 10
    M = 3
    a_tt = tt.vector("a")
    R_tt = tt.vector("R")
    X_tt = tt.matrix("X")
    V_tt = tt.vector("V")

    a_tt.tag.test_value = np.random.normal(size=M)
    R_tt.tag.test_value = np.abs(np.random.normal(size=M))
    X = np.random.normal(10, 1, size=N)
    X = np.c_[np.ones(10), X, X * X]
    X_tt.tag.test_value = X
    V_tt.tag.test_value = np.ones(N)

    beta_rv = NormalRV(a_tt, R_tt, name="\\beta")

    E_y_rv = X_tt.dot(beta_rv)
    E_y_rv.name = "E_y"
    Y_rv = NormalRV(E_y_rv, V_tt, name="Y")

    y_tt = tt.as_tensor_variable(Y_rv.tag.test_value)
    y_tt.name = "y"
    y_obs_rv = observed(y_tt, Y_rv)
    y_obs_rv.name = "y_obs"

    #
    # Use the relation with identify/match `Y`, `X` and `beta`.
    #
    y_args_tail_lv, b_args_tail_lv = var(), var()
    beta_lv = var()

    y_args_lv, y_lv, Y_lv, X_lv = var(), var(), var(), var()
    (res,) = run(
        1,
        (beta_lv, y_args_tail_lv, b_args_tail_lv),
        applyo(mt.observed, y_args_lv, y_obs_rv),
        eq(y_args_lv, (y_lv, Y_lv)),
        normal_normal_regression(Y_lv, X_lv, beta_lv, y_args_tail_lv, b_args_tail_lv),
    )

    # TODO FIXME: This would work if non-op parameters (e.g. names) were covered by
    # `operator`/`car`.  See `TheanoMetaOperator`.
    assert res[0].eval_obj.obj == beta_rv
    assert res[0] == etuplize(beta_rv)
    assert res[1] == etuplize(Y_rv)[2:]
    assert res[2] == etuplize(beta_rv)[1:]

    #
    # Use the relation with to produce `Y` from given `X` and `beta`.
    #
    X_new_mt = mt(tt.eye(N, M))
    beta_new_mt = mt(NormalRV(0, 1, size=M))
    Y_args_cdr_mt = etuplize(Y_rv)[2:]
    Y_lv = var()
    (res,) = run(1, Y_lv, normal_normal_regression(Y_lv, X_new_mt, beta_new_mt, Y_args_cdr_mt))
    Y_out_mt = res.eval_obj

    Y_new_mt = etuple(mt.NormalRV, mt.dot(X_new_mt, beta_new_mt)) + Y_args_cdr_mt
    Y_new_mt = Y_new_mt.eval_obj

    assert Y_out_mt == Y_new_mt


@pytest.mark.xfail(strict=True)
def test_normal_qr_transform():
    np.random.seed(9283)

    N = 10
    M = 3
    X_tt = tt.matrix("X")
    X = np.random.normal(10, 1, size=N)
    X = np.c_[np.ones(10), X, X * X]
    X_tt.tag.test_value = X

    V_tt = tt.vector("V")
    V_tt.tag.test_value = np.ones(N)

    a_tt = tt.vector("a")
    R_tt = tt.vector("R")
    a_tt.tag.test_value = np.random.normal(size=M)
    R_tt.tag.test_value = np.abs(np.random.normal(size=M))

    beta_rv = NormalRV(a_tt, R_tt, name="\\beta")

    E_y_rv = X_tt.dot(beta_rv)
    E_y_rv.name = "E_y"
    Y_rv = NormalRV(E_y_rv, V_tt, name="Y")

    y_tt = tt.as_tensor_variable(Y_rv.tag.test_value)
    y_tt.name = "y"
    y_obs_rv = observed(y_tt, Y_rv)
    y_obs_rv.name = "y_obs"

    (res,) = run(1, var("q"), normal_qr_transform(y_obs_rv, var("q")))

    new_node = {eval_and_reify_meta(k): eval_and_reify_meta(v) for k, v in res}

    # Make sure the old-to-new `beta` conversion is correct.
    t_Q, t_R = np.linalg.qr(X)
    Coef_new_value = np.linalg.inv(t_R)
    np.testing.assert_array_almost_equal(
        Coef_new_value, new_node[beta_rv].owner.inputs[0].tag.test_value
    )

    # Make sure the new `beta_tilde` has the right standard normal distribution
    # parameters.
    beta_tilde_node = new_node[beta_rv].owner.inputs[1]
    np.testing.assert_array_almost_equal(
        np.r_[0.0, 0.0, 0.0], beta_tilde_node.owner.inputs[0].tag.test_value
    )
    np.testing.assert_array_almost_equal(
        np.r_[1.0, 1.0, 1.0], beta_tilde_node.owner.inputs[1].tag.test_value
    )

    Y_new = new_node[y_obs_rv].owner.inputs[1]
    assert Y_new.owner.inputs[0].owner.inputs[1] == beta_tilde_node

    np.testing.assert_array_almost_equal(t_Q, Y_new.owner.inputs[0].owner.inputs[0].tag.test_value)
