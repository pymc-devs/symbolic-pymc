import pytest

import numpy as np
import theano
import theano.tensor as tt

import pymc3 as pm

from collections import Counter

from unification.utils import transitive_get as walk

# from theano.configparser import change_flags
from theano.gof.graph import inputs as tt_inputs

from symbolic_pymc.theano.random_variables import NormalRV, MvNormalRV, Observed, observed
from symbolic_pymc.theano.ops import RandomVariable
from symbolic_pymc.theano.opt import FunctionGraph
from symbolic_pymc.theano.pymc3 import model_graph, graph_model, logp, convert_rv_to_dist
from symbolic_pymc.theano.utils import canonicalize, vars_to_rvs
from symbolic_pymc.theano.meta import mt

from tests.theano.utils import create_test_hmm


@theano.change_flags(compute_test_value="ignore", cxx="")
def test_pymc3_convert_dists():
    """Just a basic check that all PyMC3 RVs will convert to and from Theano RVs."""

    with pm.Model() as model:
        norm_rv = pm.Normal("norm_rv", 0.0, 1.0, observed=1.0)
        mvnorm_rv = pm.MvNormal("mvnorm_rv", np.r_[0.0], np.c_[1.0], shape=1, observed=np.r_[1.0])
        cauchy_rv = pm.Cauchy("cauchy_rv", 0.0, 1.0, observed=1.0)
        halfcauchy_rv = pm.HalfCauchy("halfcauchy_rv", 1.0, observed=1.0)
        uniform_rv = pm.Uniform("uniform_rv", observed=1.0)
        gamma_rv = pm.Gamma("gamma_rv", 1.0, 1.0, observed=1.0)
        invgamma_rv = pm.InverseGamma("invgamma_rv", 1.0, 1.0, observed=1.0)
        exp_rv = pm.Exponential("exp_rv", 1.0, observed=1.0)
        halfnormal_rv = pm.HalfNormal("halfnormal_rv", 1.0, observed=1.0)
        beta_rv = pm.Beta("beta_rv", 2.0, 2.0, observed=1.0)
        binomial_rv = pm.Binomial("binomial_rv", 10, 0.5, observed=5)
        dirichlet_rv = pm.Dirichlet("dirichlet_rv", np.r_[0.1, 0.1], observed=np.r_[0.1, 0.1])
        poisson_rv = pm.Poisson("poisson_rv", 10, observed=5)
        bernoulli_rv = pm.Bernoulli("bernoulli_rv", 0.5, observed=0)
        betabinomial_rv = pm.BetaBinomial("betabinomial_rv", 0.1, 0.1, 10, observed=5)
        categorical_rv = pm.Categorical("categorical_rv", np.r_[0.5, 0.5], observed=1)
        multinomial_rv = pm.Multinomial("multinomial_rv", 5, np.r_[0.5, 0.5], observed=np.r_[2])
        negbinomial_rv = pm.NegativeBinomial("negbinomial_rv", 10.2, 0.5, observed=5)

    # Convert to a Theano `FunctionGraph`
    fgraph = model_graph(model)

    rvs_by_name = {n.owner.inputs[1].name: n.owner.inputs[1] for n in fgraph.outputs}

    pymc_rv_names = {n.name for n in model.observed_RVs}
    assert all(isinstance(rvs_by_name[n].owner.op, RandomVariable) for n in pymc_rv_names)

    # Now, convert back to a PyMC3 model
    pymc_model = graph_model(fgraph)

    new_pymc_rv_names = {n.name for n in pymc_model.observed_RVs}
    pymc_rv_names == new_pymc_rv_names

    with pytest.raises(TypeError):
        graph_model(NormalRV(0, 1), generate_names=False)

    res = graph_model(NormalRV(0, 1), generate_names=True)
    assert res.vars[0].name == "normal_0"


@theano.change_flags(compute_test_value="ignore")
def test_pymc3_normal_model():
    """Conduct a more in-depth test of PyMC3/Theano conversions for a specific model."""

    mu_X = tt.dscalar("mu_X")
    sd_X = tt.dscalar("sd_X")
    mu_Y = tt.dscalar("mu_Y")
    mu_X.tag.test_value = np.array(0.0, dtype=tt.config.floatX)
    sd_X.tag.test_value = np.array(1.0, dtype=tt.config.floatX)
    mu_Y.tag.test_value = np.array(1.0, dtype=tt.config.floatX)

    # We need something that uses transforms...
    with pm.Model() as model:
        X_rv = pm.Normal("X_rv", mu_X, sigma=sd_X)
        S_rv = pm.HalfCauchy("S_rv", beta=np.array(0.5, dtype=tt.config.floatX))
        Y_rv = pm.Normal("Y_rv", X_rv * S_rv, sigma=S_rv)
        Z_rv = pm.Normal("Z_rv", X_rv + Y_rv, sigma=sd_X, observed=10.0)

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


@theano.change_flags(compute_test_value="ignore")
def test_convert_rv_to_dist_shape():

    # Make sure we use the `ShapeFeature` to get the shape info
    X_rv = NormalRV(np.r_[1, 2], 2.0, name="X_rv")
    fgraph = FunctionGraph(tt_inputs([X_rv]), [X_rv], features=[tt.opt.ShapeFeature()])

    with pm.Model():
        res = convert_rv_to_dist(fgraph.outputs[0].owner, None)

    assert isinstance(res.distribution, pm.Normal)
    assert np.array_equal(res.distribution.shape, np.r_[2])


@theano.change_flags(compute_test_value="ignore")
def test_normals_to_model():
    """Test conversion to a PyMC3 model."""

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


@theano.change_flags(compute_test_value="ignore")
def test_pymc3_broadcastable():
    """Test PyMC3 to Theano conversion amid array broadcasting."""

    mu_X = tt.vector("mu_X")
    sd_X = tt.vector("sd_X")
    mu_Y = tt.vector("mu_Y")
    sd_Y = tt.vector("sd_Y")
    mu_X.tag.test_value = np.array([0.0], dtype=tt.config.floatX)
    sd_X.tag.test_value = np.array([1.0], dtype=tt.config.floatX)
    mu_Y.tag.test_value = np.array([1.0], dtype=tt.config.floatX)
    sd_Y.tag.test_value = np.array([0.5], dtype=tt.config.floatX)

    with pm.Model() as model:
        X_rv = pm.Normal("X_rv", mu_X, sigma=sd_X, shape=(1,))
        Y_rv = pm.Normal("Y_rv", mu_Y, sigma=sd_Y, shape=(1,))
        Z_rv = pm.Normal("Z_rv", X_rv + Y_rv, sigma=sd_X + sd_Y, shape=(1,), observed=[10.0])

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


@theano.change_flags(compute_test_value="warn", cxx="")
def test_logp():

    hmm_model_env = create_test_hmm()
    M_tt = hmm_model_env["M_tt"]
    N_tt = hmm_model_env["N_tt"]
    mus_tt = hmm_model_env["mus_tt"]
    sigmas_tt = hmm_model_env["sigmas_tt"]
    Y_rv = hmm_model_env["Y_rv"]
    S_rv = hmm_model_env["S_rv"]
    S_in = hmm_model_env["S_in"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    rng_tt = hmm_model_env["rng_tt"]

    Y_obs = Y_rv.clone()
    Y_obs.name = "Y_obs"
    # `S_in` includes `S_0_rv` (and `pi_0_rv`), unlike `S_rv`
    S_obs = S_in.clone()
    S_obs.name = "S_obs"
    Gamma_obs = Gamma_rv.clone()
    Gamma_obs.name = "Gamma_obs"

    test_point = {
        mus_tt: mus_tt.tag.test_value,
        N_tt: N_tt.tag.test_value,
        Gamma_obs: Gamma_rv.tag.test_value,
        Y_obs: Y_rv.tag.test_value,
        S_obs: S_in.tag.test_value,
    }

    def logp_scan_fn(s_t, s_tm1, y_t, mus_t, sigma_t, Gamma_t):
        gamma_t = Gamma_t[s_tm1]
        log_s_t = pm.Categorical.dist(gamma_t).logp(s_t)
        mu_t = mus_t[s_t]
        log_y_t = pm.Normal.dist(mu_t, sigma_t).logp(y_t)
        gamma_t.name = "gamma_t"
        log_y_t.name = "logp(y_t)"
        log_s_t.name = "logp(s_t)"
        mu_t.name = "mu[S_t]"
        return log_s_t, log_y_t

    (true_S_logp, true_Y_logp), scan_updates = theano.scan(
        fn=logp_scan_fn,
        sequences=[{"input": S_obs, "taps": [0, -1]}, Y_obs, mus_tt, sigmas_tt],
        non_sequences=[Gamma_obs],
        outputs_info=[{}, {}],
        strict=True,
        name="scan_rv",
    )

    # Make sure there are no `RandomVariable` nodes among our
    # expected/true log-likelihood graph.
    assert not vars_to_rvs(true_S_logp)
    assert not vars_to_rvs(true_Y_logp)

    true_S_logp_val = true_S_logp.eval(test_point)
    true_Y_logp_val = true_Y_logp.eval(test_point)

    #
    # Now, compute the log-likelihoods
    #
    logps = logp(Y_rv)

    S_logp = logps[S_in][1]
    Y_logp = logps[Y_rv][1]

    # from theano.printing import debugprint as tt_dprint

    # There shouldn't be any `RandomVariable`s here either
    assert not vars_to_rvs(S_logp[1])
    assert not vars_to_rvs(Y_logp[1])

    assert N_tt in tt_inputs([S_logp])
    assert mus_tt in tt_inputs([S_logp])
    assert logps[S_in][0] in tt_inputs([S_logp])
    assert logps[Y_rv][0] in tt_inputs([S_logp])
    assert logps[Gamma_rv][0] in tt_inputs([S_logp])

    new_test_point = {
        mus_tt: mus_tt.tag.test_value,
        N_tt: N_tt.tag.test_value,
        logps[Gamma_rv][0]: Gamma_rv.tag.test_value,
        logps[Y_rv][0]: Y_rv.tag.test_value,
        logps[S_in][0]: S_in.tag.test_value,
    }

    with theano.change_flags(on_unused_input="warn"):
        S_logp_val = S_logp.eval(new_test_point)
        Y_logp_val = Y_logp.eval(new_test_point)

    assert np.array_equal(true_S_logp_val, S_logp_val)
    assert np.array_equal(Y_logp_val, true_Y_logp_val)
