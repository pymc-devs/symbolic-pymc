import pytest

import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm

from theano.configparser import change_flags
from theano.gof import FunctionGraph, Feature, NodeFinder
from theano.gof.graph import inputs as tt_inputs, clone_get_equiv

from symbolic_pymc import *
from symbolic_pymc.pymc3 import *
from symbolic_pymc.utils import graph_equal, canonicalize
from symbolic_pymc.meta import mt

tt.config.on_opt_error = 'raise'
tt.config.compute_test_value = 'ignore'
theano.config.mode = 'FAST_COMPILE'
theano.config.cxx = ''


# @change_flags
def test_pymc_normals():
    mu_X = tt.scalar('mu_X')
    sd_X = tt.scalar('sd_X')
    mu_Y = tt.scalar('mu_Y')
    sd_Y = tt.scalar('sd_Y')
    mu_X.tag.test_value = np.array(0., dtype=tt.config.floatX)
    sd_X.tag.test_value = np.array(1., dtype=tt.config.floatX)
    mu_Y.tag.test_value = np.array(1., dtype=tt.config.floatX)
    sd_Y.tag.test_value = np.array(0.5, dtype=tt.config.floatX)

    with pm.Model() as model:
        X_rv = pm.Normal('X_rv', mu_X, sd=sd_X)
        Y_rv = pm.Normal('Y_rv', mu_Y, sd=sd_Y)
        Z_rv = pm.Normal('Z_rv',
                         X_rv + Y_rv,
                         sd=sd_X + sd_Y,
                         observed=10.)

    fgraph = model_graph(model, output_vars=[Z_rv])
    Z_rv_tt = canonicalize(fgraph)

    # This will break comparison if we don't reuse it
    rng = Z_rv_tt.owner.inputs[1].owner.inputs[-1]

    mu_X_ = mt.scalar('mu_X')
    sd_X_ = mt.scalar('sd_X')
    mu_Y_ = mt.scalar('mu_Y')
    sd_Y_ = mt.scalar('sd_Y')
    tt.config.compute_test_value = 'ignore'
    X_rv_ = mt.NormalRV(mu_X_, sd_X_, None, rng, name='X_rv')
    Y_rv_ = mt.NormalRV(mu_Y_, sd_Y_, None, rng, name='Y_rv')
    Z_rv_ = mt.NormalRV(mt.add(X_rv_, Y_rv_),
                        mt.add(sd_X_, sd_Y_),
                        None, rng, name='Z_rv')
    Z_rv_ = mt.observed(Z_rv.observations, Z_rv_)

    Z_rv_meta = canonicalize(Z_rv_.reify())

    assert mt(Z_rv_tt) == mt(Z_rv_meta)


def test_normals_to_model():
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

    fgraph = FunctionGraph(tt_inputs([beta_rv, Y_obs]),
                           [beta_rv, Y_obs],
                           clone=True)

    model = graph_model(fgraph)

    assert len(model.observed_RVs) == 1
    assert model.observed_RVs[0].name == 'Y'
    Y_pm = model.observed_RVs[0].distribution
    assert isinstance(Y_pm, pm.MvNormal)
    np.testing.assert_array_equal(
        model.observed_RVs[0].observations.data,
        y_tt.data)
    assert Y_pm.mu.owner.op == tt.basic._dot
    assert Y_pm.cov.name == 'V'
    assert len(model.unobserved_RVs) == 1
    assert model.unobserved_RVs[0].name == '\\beta'
    beta_pm = model.unobserved_RVs[0].distribution
    assert isinstance(beta_pm, pm.MvNormal)


def test_pymc_broadcastable():
    mu_X = tt.vector('mu_X')
    sd_X = tt.vector('sd_X')
    mu_Y = tt.vector('mu_Y')
    sd_Y = tt.vector('sd_Y')
    mu_X.tag.test_value = np.array([0.], dtype=tt.config.floatX)
    sd_X.tag.test_value = np.array([1.], dtype=tt.config.floatX)
    mu_Y.tag.test_value = np.array([1.], dtype=tt.config.floatX)
    sd_Y.tag.test_value = np.array([0.5], dtype=tt.config.floatX)

    with pm.Model() as model:
        X_rv = pm.Normal('X_rv', mu_X, sd=sd_X, shape=(1,))
        Y_rv = pm.Normal('Y_rv', mu_Y, sd=sd_Y, shape=(1,))
        Z_rv = pm.Normal('Z_rv',
                         X_rv + Y_rv,
                         sd=sd_X + sd_Y,
                         shape=(1,),
                         observed=[10.])

    with pytest.warns(UserWarning):
        fgraph = model_graph(model)

    Z_rv_tt = canonicalize(fgraph)

    # This will break comparison if we don't reuse it
    rng = Z_rv_tt.owner.inputs[1].owner.inputs[-1]

    mu_X_ = mt.vector('mu_X')
    sd_X_ = mt.vector('sd_X')
    mu_Y_ = mt.vector('mu_Y')
    sd_Y_ = mt.vector('sd_Y')
    tt.config.compute_test_value = 'ignore'
    X_rv_ = mt.NormalRV(mu_X_, sd_X_, None, rng, name='X_rv')
    X_rv_ = mt.addbroadcast(X_rv_, 0)
    Y_rv_ = mt.NormalRV(mu_Y_, sd_Y_, None, rng, name='Y_rv')
    Y_rv_ = mt.addbroadcast(Y_rv_, 0)
    Z_rv_ = mt.NormalRV(mt.add(X_rv_, Y_rv_),
                        mt.add(sd_X_, sd_Y_),
                        None, rng, name='Z_rv')
    Z_rv_ = mt.observed(Z_rv.observations, Z_rv_)

    Z_rv_meta = canonicalize(Z_rv_.reify())

    assert mt(Z_rv_tt) == mt(Z_rv_meta)
