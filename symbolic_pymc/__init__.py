import theano
import scipy

from functools import partial

from .rv import RandomVariable, param_supp_shape_fn

# We need this so that `multipledispatch` initialization occurs
from .unify import *


# Continuous Numpy-generated variates
class UniformRVType(RandomVariable):
    def __init__(self):
        super().__init__('uniform', theano.config.floatX, 0, [0, 0], 'uniform',
                         inplace=True)

    def make_node(self, lower, upper, size=None, rng=None, name=None):
        return super().make_node(lower, upper, size=size, rng=rng, name=name)


UniformRV = UniformRVType()


class NormalRVType(RandomVariable):
    def __init__(self):
        super().__init__('normal', theano.config.floatX, 0, [0, 0], 'normal',
                         inplace=True)

    def make_node(self, mu, sigma, size=None, rng=None, name=None):
        return super().make_node(mu, sigma, size=size, rng=rng, name=name)


NormalRV = NormalRVType()


class GammaRVType(RandomVariable):
    def __init__(self):
        super().__init__('gamma', theano.config.floatX, 0, [0, 0], 'gamma',
                         inplace=True)

    def make_node(self, shape, scale, size=None, rng=None, name=None):
        return super().make_node(shape, scale, size=size, rng=rng, name=name)


GammaRV = GammaRVType()


class ExponentialRVType(RandomVariable):
    def __init__(self):
        super().__init__('exponential', theano.config.floatX, 0, [0],
                         'exponential', inplace=True)

    def make_node(self, scale, size=None, rng=None, name=None):
        return super().make_node(scale, size=size, rng=rng, name=name)


ExponentialRV = ExponentialRVType()


# One with multivariate support
class MvNormalRVType(RandomVariable):
    def __init__(self):
        super().__init__('multivariate_normal', theano.config.floatX, 1,
                         [1, 2], 'multivariate_normal', inplace=True)

    def make_node(self, mean, cov, size=None, rng=None, name=None):
        return super().make_node(mean, cov, size=size, rng=rng, name=name)


MvNormalRV = MvNormalRVType()


class DirichletRVType(RandomVariable):
    def __init__(self):
        super().__init__('dirichlet', theano.config.floatX, 1, [1],
                         'dirichlet', inplace=True)

    def make_node(self, alpha, size=None, rng=None, name=None):
        return super().make_node(alpha, size=size, rng=rng, name=name)


DirichletRV = DirichletRVType()


# A discrete Numpy-generated variate
class PoissonRVType(RandomVariable):
    def __init__(self):
        super().__init__('poisson', 'int64', 0, [0], 'poisson', inplace=True)

    def make_node(self, rate, size=None, rng=None, name=None):
        return super().make_node(rate, size=size, rng=rng, name=name)


PoissonRV = PoissonRVType()


# A SciPy-generated variate
class CauchyRVType(RandomVariable):
    def __init__(self):
        super().__init__('cauchy', theano.config.floatX, 0, [0, 0],
                         lambda rng, *args: scipy.stats.cauchy.rvs(*args,
                                                                   random_state=rng),
                         inplace=True)

    def make_node(self, loc, scale, size=None, rng=None, name=None):
        return super().make_node(loc, scale, size=size, rng=rng, name=name)


CauchyRV = CauchyRVType()


# Support shape is determined by the first dimension in the *second* parameter (i.e.
# the probabilities vector)
class MultinomialRVType(RandomVariable):
    def __init__(self):
        super().__init__('multinomial', 'int64', 1, [0, 1], 'multinomial',
                         supp_shape_fn=partial(param_supp_shape_fn,
                                               rep_param_idx=1),
                         inplace=True)

    def make_node(self, n, pvals, size=None, rng=None, name=None):
        return super().make_node(n, pvals, size=size, rng=rng, name=name)


MultinomialRV = MultinomialRVType()
