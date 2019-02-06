import pytest

import theano
import theano.tensor as tt
import numpy as np

from theano.gof.opt import EquilibriumOptimizer

from symbolic_pymc import MvNormalRV
from symbolic_pymc.opt import KanrenRelationSub
from symbolic_pymc.utils import optimize_graph
from symbolic_pymc.relations.conjugates import observed, posterior_transforms

theano.config.mode = 'FAST_COMPILE'
theano.config.cxx = ''


@pytest.mark.xfail
def test_mvnormal_mvnormal():
    theano.config.compute_test_value = 'ignore'
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
    Y_rv = MvNormalRV(E_y_rv, V_tt, name='y')

    y_tt = Y_rv.clone()
    y_tt.name = 'y_obs'

    Y_obs = observed(y_tt, Y_rv)

    posterior_opt = EquilibriumOptimizer(
        [KanrenRelationSub(posterior_transforms)],
        max_use_ratio=10)

    Y_opt = optimize_graph(Y_obs, posterior_opt)

    assert False
