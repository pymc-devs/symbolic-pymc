import pytest

import numpy as np
import theano
import theano.tensor as tt

from functools import partial
from unification import var

from kanren import run, eq

from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.random_variables import (observed, NormalRV,
                                                   HalfCauchyRV)
from symbolic_pymc.relations.graph import reduceo
from symbolic_pymc.relations.theano import non_obs_graph_applyo
from symbolic_pymc.relations.theano.distributions import scale_loc_transform, constant_neq


def non_obs_fixedp_graph_applyo(r, x, y):
    return reduceo(partial(non_obs_graph_applyo, r), x, y)


@pytest.mark.usefixtures("run_with_theano")
def test_pymc_normals():
    tt.config.compute_test_value = 'ignore'

    rand_state = theano.shared(np.random.RandomState())
    mu_a = NormalRV(0., 100**2, name='mu_a', rng=rand_state)
    sigma_a = HalfCauchyRV(5, name='sigma_a', rng=rand_state)
    mu_b = NormalRV(0., 100**2, name='mu_b', rng=rand_state)
    sigma_b = HalfCauchyRV(5, name='sigma_b', rng=rand_state)
    county_idx = np.r_[1, 1, 2, 3]
    # We want the following for a, b:
    # N(m, S) -> m + N(0, 1) * S
    a = NormalRV(mu_a, sigma_a, size=(len(county_idx),), name='a', rng=rand_state)
    b = NormalRV(mu_b, sigma_b, size=(len(county_idx),), name='b', rng=rand_state)
    radon_est = a[county_idx] + b[county_idx] * 7
    eps = HalfCauchyRV(5, name='eps', rng=rand_state)
    radon_like = NormalRV(radon_est, eps, name='radon_like', rng=rand_state)
    radon_like_rv = observed(tt.as_tensor_variable(np.r_[1., 2., 3., 4.]), radon_like)

    graph_mt = mt(radon_like_rv)
    expr_graph, = run(1, var('q'),
                      non_obs_fixedp_graph_applyo(scale_loc_transform, graph_mt, var('q')))

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
    assert 'mu_a' == mu_a_opt.name == mu_a.name
    mu_b_opt = b_opt.owner.inputs[0].owner.inputs[0]
    assert 'mu_b' == mu_b_opt.name == mu_b.name

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


def test_distributions():
    res = run(0, var('q'), eq(var('q'), mt(1)), constant_neq(var('q'), np.array(1.)))

    assert not res

    res = run(0, var('q'), eq(var('q'), mt(2)), constant_neq(var('q'), np.array(1.)))

    assert res