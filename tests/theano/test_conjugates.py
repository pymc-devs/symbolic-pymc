import pytest
import theano.tensor as tt
import numpy as np

from unification import var

from kanren import run

from symbolic_pymc.theano.random_variables import MvNormalRV, observed
from symbolic_pymc.relations.graph import graph_applyo
from symbolic_pymc.relations.theano.conjugates import conjugate


@pytest.mark.usefixtures("run_with_theano")
def test_mvnormal_mvnormal():
    """Test that we can produce the closed-form distribution for the conjugate
    multivariate normal-regression with normal-prior model.
    """
    tt.config.cxx = ''
    tt.config.compute_test_value = 'ignore'

    a_tt = tt.vector('a')
    R_tt = tt.matrix('R')
    F_t_tt = tt.matrix('F')
    V_tt = tt.matrix('V')

    a_tt.tag.test_value = np.r_[1., 0.]
    R_tt.tag.test_value = np.diag([10., 10.])
    F_t_tt.tag.test_value = np.c_[-2., 1.]
    V_tt.tag.test_value = np.diag([0.5])

    beta_rv = MvNormalRV(a_tt, R_tt, name='\\beta')

    E_y_rv = F_t_tt.dot(beta_rv)
    Y_rv = MvNormalRV(E_y_rv, V_tt, name='Y')

    y_tt = tt.as_tensor_variable(np.r_[-3.])
    y_tt.name = 'y'
    Y_obs = observed(y_tt, Y_rv)

    q_lv = var()

    expr_graph, = run(1, q_lv, (graph_applyo, conjugate, Y_obs, q_lv))

    fgraph_opt = expr_graph.eval_obj
    fgraph_opt_tt = fgraph_opt.reify()

    # Check that the SSE has decreased from prior to posterior.
    # TODO: Use a better test.
    beta_prior_mean_val = a_tt.tag.test_value
    F_val = F_t_tt.tag.test_value
    beta_post_mean_val = fgraph_opt_tt.owner.inputs[0].tag.test_value
    priorp_err = np.square(
        y_tt.data - F_val.dot(beta_prior_mean_val)).sum()
    postp_err = np.square(
        y_tt.data - F_val.dot(beta_post_mean_val)).sum()

    # First, make sure the prior and posterior means are simply not equal.
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal,
        priorp_err, postp_err)
    # Now, make sure there's a decrease (relative to the observed point).
    np.testing.assert_array_less(postp_err, priorp_err)
