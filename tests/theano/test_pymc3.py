import pytest

import numpy as np
import theano
import theano.tensor as tt

import pymc3 as pm

from collections import Counter

from unification.utils import transitive_get as walk

# from theano.configparser import change_flags
from theano.gof.graph import inputs as tt_inputs

from symbolic_pymc.theano.random_variables import MvNormalRV, Observed, observed
from symbolic_pymc.theano.ops import RandomVariable
from symbolic_pymc.theano.opt import FunctionGraph
from symbolic_pymc.theano.pymc3 import model_graph, graph_model
from symbolic_pymc.theano.utils import canonicalize
from symbolic_pymc.theano.meta import mt


@pytest.mark.usefixtures("run_with_theano")
def test_pymc_convert_dists():
    """Just a basic check that all PyMC3 RVs will convert to and from Theano RVs."""
    tt.config.compute_test_value = "ignore"
    theano.config.cxx = ""

    with pm.Model() as model:
        norm_rv = pm.Normal("norm_rv", 0.0, 1.0, observed=1.0)
        mvnorm_rv = pm.MvNormal("mvnorm_rv", np.r_[0.0], np.c_[1.0], shape=1, observed=np.r_[1.0])
        cauchy_rv = pm.Cauchy("cauchy_rv", 0.0, 1.0, observed=1.0)
        halfcauchy_rv = pm.HalfCauchy("halfcauchy_rv", 1.0, observed=1.0)
        uniform_rv = pm.Uniform("uniform_rv", observed=1.0)
        gamma_rv = pm.Gamma("gamma_rv", 1.0, 1.0, observed=1.0)
        invgamma_rv = pm.InverseGamma("invgamma_rv", 1.0, 1.0, observed=1.0)
        exp_rv = pm.Exponential("exp_rv", 1.0, observed=1.0)

    # Convert to a Theano `FunctionGraph`
    fgraph = model_graph(model)

    rvs_by_name = {n.owner.inputs[1].name: n.owner.inputs[1] for n in fgraph.outputs}

    pymc_rv_names = {n.name for n in model.observed_RVs}
    assert all(isinstance(rvs_by_name[n].owner.op, RandomVariable) for n in pymc_rv_names)

    # Now, convert back to a PyMC3 model
    pymc_model = graph_model(fgraph)

    new_pymc_rv_names = {n.name for n in pymc_model.observed_RVs}
    pymc_rv_names == new_pymc_rv_names


@pytest.mark.usefixtures("run_with_theano")
def test_pymc_normal_model():
    """Conduct a more in-depth test of PyMC3/Theano conversions for a specific model."""
    tt.config.compute_test_value = "ignore"

    mu_X = tt.dscalar("mu_X")
    sd_X = tt.dscalar("sd_X")
    mu_Y = tt.dscalar("mu_Y")
    mu_X.tag.test_value = np.array(0.0, dtype=tt.config.floatX)
    sd_X.tag.test_value = np.array(1.0, dtype=tt.config.floatX)
    mu_Y.tag.test_value = np.array(1.0, dtype=tt.config.floatX)

    # We need something that uses transforms...
    with pm.Model() as model:
        X_rv = pm.Normal("X_rv", mu_X, sd=sd_X)
        S_rv = pm.HalfCauchy("S_rv", beta=np.array(0.5, dtype=tt.config.floatX))
        Y_rv = pm.Normal("Y_rv", X_rv * S_rv, sd=S_rv)
        Z_rv = pm.Normal("Z_rv", X_rv + Y_rv, sd=sd_X, observed=10.0)

    fgraph = model_graph(model, output_vars=[Z_rv])

    Z_rv_tt = canonicalize(fgraph, return_graph=False)

    # This will break comparison if we don't reuse it
    rng = Z_rv_tt.owner.inputs[1].owner.inputs[-1]

    mu_X_ = mt.dscalar("mu_X")
    sd_X_ = mt.dscalar("sd_X")
    tt.config.compute_test_value = "ignore"
    X_rv_ = mt.NormalRV(mu_X_, sd_X_, None, rng, name="X_rv")
    S_rv_ = mt.HalfCauchyRV(
        np.array(0.0, dtype=tt.config.floatX),
        np.array(0.5, dtype=tt.config.floatX),
        None,
        rng,
        name="S_rv",
    )
    Y_rv_ = mt.NormalRV(mt.mul(X_rv_, S_rv_), S_rv_, None, rng, name="Y_rv")
    Z_rv_ = mt.NormalRV(mt.add(X_rv_, Y_rv_), sd_X, None, rng, name="Z_rv")
    obs_ = mt(Z_rv.observations)
    Z_rv_obs_ = mt.observed(obs_, Z_rv_)

    Z_rv_meta = mt(canonicalize(Z_rv_obs_.reify(), return_graph=False))

    assert mt(Z_rv_tt) == Z_rv_meta

    # Now, let's try that with multiple outputs.
    fgraph.disown()
    fgraph = model_graph(model, output_vars=[Y_rv, Z_rv])

    assert len(fgraph.variables) == 25

    Y_new_rv = walk(Y_rv, fgraph.memo)
    S_new_rv = walk(S_rv, fgraph.memo)
    X_new_rv = walk(X_rv, fgraph.memo)
    Z_new_rv = walk(Z_rv, fgraph.memo)

    # Make sure our new vars are actually in the graph and where
    # they should be.
    assert Y_new_rv == fgraph.outputs[0]
    assert Z_new_rv == fgraph.outputs[1]
    assert X_new_rv in fgraph.variables
    assert S_new_rv in fgraph.variables
    assert isinstance(Z_new_rv.owner.op, Observed)

    # Let's only look at the variables involved in the `Z_rv` subgraph.
    Z_vars = theano.gof.graph.variables(theano.gof.graph.inputs([Z_new_rv]), [Z_new_rv])

    # Let's filter for only the `RandomVariables` with names.
    Z_vars_count = Counter(
        [n.name for n in Z_vars if n.name and n.owner and isinstance(n.owner.op, RandomVariable)]
    )

    # Each new RV should be present and only occur once.
    assert Y_new_rv.name in Z_vars_count.keys()
    assert X_new_rv.name in Z_vars_count.keys()
    assert Z_new_rv.owner.inputs[1].name in Z_vars_count.keys()
    assert all(v == 1 for v in Z_vars_count.values())


@pytest.mark.usefixtures("run_with_theano")
def test_normals_to_model():
    """Test conversion to a PyMC3 model."""
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

    y_val = np.r_[-3.0]

    def _check_model(model):
        assert len(model.observed_RVs) == 1
        assert model.observed_RVs[0].name == "Y"
        Y_pm = model.observed_RVs[0].distribution
        assert isinstance(Y_pm, pm.MvNormal)
        np.testing.assert_array_equal(model.observed_RVs[0].observations.data, y_val)
        assert Y_pm.mu.owner.op == tt.basic._dot
        assert Y_pm.cov.name == "V"
        assert len(model.unobserved_RVs) == 1
        assert model.unobserved_RVs[0].name == "\\beta"
        beta_pm = model.unobserved_RVs[0].distribution
        assert isinstance(beta_pm, pm.MvNormal)

    y_tt = theano.shared(y_val, name="y")
    Y_obs = observed(y_tt, Y_rv)

    fgraph = FunctionGraph(tt_inputs([beta_rv, Y_obs]), [beta_rv, Y_obs], clone=True)

    model = graph_model(fgraph)

    _check_model(model)

    # Now, let `graph_model` create the `FunctionGraph`
    model = graph_model(Y_obs)

    _check_model(model)

    # Use a different type of observation value
    y_tt = tt.as_tensor_variable(y_val, name="y")
    Y_obs = observed(y_tt, Y_rv)

    model = graph_model(Y_obs)

    _check_model(model)

    # Use an invalid type of observation value
    tt.config.compute_test_value = "ignore"
    y_tt = tt.vector("y")
    Y_obs = observed(y_tt, Y_rv)

    with pytest.raises(TypeError):
        model = graph_model(Y_obs)


@pytest.mark.usefixtures("run_with_theano")
def test_pymc_broadcastable():
    """Test PyMC3 to Theano conversion amid array broadcasting."""
    tt.config.compute_test_value = "ignore"

    mu_X = tt.vector("mu_X")
    sd_X = tt.vector("sd_X")
    mu_Y = tt.vector("mu_Y")
    sd_Y = tt.vector("sd_Y")
    mu_X.tag.test_value = np.array([0.0], dtype=tt.config.floatX)
    sd_X.tag.test_value = np.array([1.0], dtype=tt.config.floatX)
    mu_Y.tag.test_value = np.array([1.0], dtype=tt.config.floatX)
    sd_Y.tag.test_value = np.array([0.5], dtype=tt.config.floatX)

    with pm.Model() as model:
        X_rv = pm.Normal("X_rv", mu_X, sd=sd_X, shape=(1,))
        Y_rv = pm.Normal("Y_rv", mu_Y, sd=sd_Y, shape=(1,))
        Z_rv = pm.Normal("Z_rv", X_rv + Y_rv, sd=sd_X + sd_Y, shape=(1,), observed=[10.0])

    with pytest.warns(UserWarning):
        fgraph = model_graph(model)

    Z_rv_tt = canonicalize(fgraph, return_graph=False)

    # This will break comparison if we don't reuse it
    rng = Z_rv_tt.owner.inputs[1].owner.inputs[-1]

    mu_X_ = mt.vector("mu_X")
    sd_X_ = mt.vector("sd_X")
    mu_Y_ = mt.vector("mu_Y")
    sd_Y_ = mt.vector("sd_Y")
    tt.config.compute_test_value = "ignore"
    X_rv_ = mt.NormalRV(mu_X_, sd_X_, (1,), rng, name="X_rv")
    X_rv_ = mt.addbroadcast(X_rv_, 0)
    Y_rv_ = mt.NormalRV(mu_Y_, sd_Y_, (1,), rng, name="Y_rv")
    Y_rv_ = mt.addbroadcast(Y_rv_, 0)
    Z_rv_ = mt.NormalRV(mt.add(X_rv_, Y_rv_), mt.add(sd_X_, sd_Y_), (1,), rng, name="Z_rv")
    obs_ = mt(Z_rv.observations)
    Z_rv_obs_ = mt.observed(obs_, Z_rv_)
    Z_rv_meta = canonicalize(Z_rv_obs_.reify(), return_graph=False)

    assert mt(Z_rv_tt) == mt(Z_rv_meta)
