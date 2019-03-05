import theano
import scipy
import theano.tensor as tt

from functools import partial

from .ops import RandomVariable, param_supp_shape_fn


# Continuous Numpy-generated variates
class UniformRVType(RandomVariable):
    print_name = ("U", "\\operatorname{U}")

    def __init__(self):
        super().__init__("uniform", theano.config.floatX, 0, [0, 0], "uniform", inplace=True)

    def make_node(self, lower, upper, size=None, rng=None, name=None):
        return super().make_node(lower, upper, size=size, rng=rng, name=name)


UniformRV = UniformRVType()


class NormalRVType(RandomVariable):
    print_name = ("N", "\\operatorname{N}")

    def __init__(self):
        super().__init__("normal", theano.config.floatX, 0, [0, 0], "normal", inplace=True)

    def make_node(self, mu, sigma, size=None, rng=None, name=None):
        return super().make_node(mu, sigma, size=size, rng=rng, name=name)


NormalRV = NormalRVType()


class HalfNormalRVType(RandomVariable):
    print_name = ("N**+", "\\operatorname{N^{+}}")

    def __init__(self):
        super().__init__(
            "halfnormal",
            theano.config.floatX,
            0,
            [0, 0],
            lambda rng, *args: scipy.stats.halfnorm.rvs(*args, random_state=rng),
            inplace=True,
        )

    def make_node(self, mu=0.0, sigma=1.0, size=None, rng=None, name=None):
        return super().make_node(mu, sigma, size=size, rng=rng, name=name)


HalfNormalRV = HalfNormalRVType()


class GammaRVType(RandomVariable):
    print_name = ("Gamma", "\\operatorname{Gamma}")

    def __init__(self):
        super().__init__("gamma", theano.config.floatX, 0, [0, 0], "gamma", inplace=True)

    def make_node(self, shape, scale, size=None, rng=None, name=None):
        return super().make_node(shape, scale, size=size, rng=rng, name=name)


GammaRV = GammaRVType()


class ExponentialRVType(RandomVariable):
    print_name = ("Exp", "\\operatorname{Exp}")

    def __init__(self):
        super().__init__("exponential", theano.config.floatX, 0, [0], "exponential", inplace=True)

    def make_node(self, scale, size=None, rng=None, name=None):
        return super().make_node(scale, size=size, rng=rng, name=name)


ExponentialRV = ExponentialRVType()


# One with multivariate support
class MvNormalRVType(RandomVariable):
    print_name = ("N", "\\operatorname{N}")

    def __init__(self):
        super().__init__(
            "multivariate_normal",
            theano.config.floatX,
            1,
            [1, 2],
            "multivariate_normal",
            inplace=True,
        )

    def make_node(self, mean, cov, size=None, rng=None, name=None):
        return super().make_node(mean, cov, size=size, rng=rng, name=name)


MvNormalRV = MvNormalRVType()


class DirichletRVType(RandomVariable):
    print_name = ("Dir", "\\operatorname{Dir}")

    def __init__(self):
        super().__init__("dirichlet", theano.config.floatX, 1, [1], "dirichlet", inplace=True)

    def make_node(self, alpha, size=None, rng=None, name=None):
        return super().make_node(alpha, size=size, rng=rng, name=name)


DirichletRV = DirichletRVType()


# A discrete Numpy-generated variate
class PoissonRVType(RandomVariable):
    print_name = ("Pois", "\\operatorname{Pois}")

    def __init__(self):
        super().__init__("poisson", "int64", 0, [0], "poisson", inplace=True)

    def make_node(self, rate, size=None, rng=None, name=None):
        return super().make_node(rate, size=size, rng=rng, name=name)


PoissonRV = PoissonRVType()


# A SciPy-generated variate
class CauchyRVType(RandomVariable):
    print_name = ("C", "\\operatorname{C}")

    def __init__(self):
        super().__init__(
            "cauchy",
            theano.config.floatX,
            0,
            [0, 0],
            lambda rng, *args: scipy.stats.cauchy.rvs(*args, random_state=rng),
            inplace=True,
        )

    def make_node(self, loc, scale, size=None, rng=None, name=None):
        return super().make_node(loc, scale, size=size, rng=rng, name=name)


CauchyRV = CauchyRVType()


class HalfCauchyRVType(RandomVariable):
    print_name = ("C**+", "\\operatorname{C^{+}}")

    def __init__(self):
        super().__init__(
            "halfcauchy",
            theano.config.floatX,
            0,
            [0, 0],
            lambda rng, *args: scipy.stats.halfcauchy.rvs(*args, random_state=rng),
            inplace=True,
        )

    def make_node(self, loc=0.0, scale=1.0, size=None, rng=None, name=None):
        return super().make_node(loc, scale, size=size, rng=rng, name=name)


HalfCauchyRV = HalfCauchyRVType()


class InvGammaRVType(RandomVariable):
    print_name = ("InvGamma", "\\operatorname{Gamma^{-1}}")

    def __init__(self):
        super().__init__(
            "invgamma",
            theano.config.floatX,
            0,
            [0, 0],
            lambda rng, *args: scipy.stats.invgamma.rvs(*args, random_state=rng),
            inplace=True,
        )

    def make_node(self, loc, scale, size=None, rng=None, name=None):
        return super().make_node(loc, scale, size=size, rng=rng, name=name)


InvGammaRV = InvGammaRVType()


class TruncExponentialRVType(RandomVariable):
    print_name = ("TruncExp", "\\operatorname{Exp}")

    def __init__(self):
        super().__init__(
            "truncexpon",
            theano.config.floatX,
            0,
            [0, 0, 0],
            lambda rng, *args: scipy.stats.truncexpon.rvs(*args, random_state=rng),
            inplace=True,
        )

    def make_node(self, b, loc, scale, size=None, rng=None, name=None):
        return super().make_node(b, loc, scale, size=size, rng=rng, name=name)


TruncExponentialRV = TruncExponentialRVType()


# Support shape is determined by the first dimension in the *second* parameter
# (i.e.  the probabilities vector)
class MultinomialRVType(RandomVariable):
    print_name = ("MN", "\\operatorname{MN}")

    def __init__(self):
        super().__init__(
            "multinomial",
            "int64",
            1,
            [0, 1],
            "multinomial",
            supp_shape_fn=partial(param_supp_shape_fn, rep_param_idx=1),
            inplace=True,
        )

    def make_node(self, n, pvals, size=None, rng=None, name=None):
        return super().make_node(n, pvals, size=size, rng=rng, name=name)


MultinomialRV = MultinomialRVType()


class Observed(tt.Op):
    """An `Op` that represents an observed random variable.

    This `Op` establishes an observation relationship between a random
    variable and a specific value.
    """

    default_output = 0

    def __init__(self):
        self.view_map = {0: [0]}

    def make_node(self, val, rv=None):
        """Make an `Observed` random variable.

        Parameters
        ----------
        val: Variable
            The observed value.
        rv: RandomVariable
            The distribution from which `val` is assumed to be a sample value.
        """
        val = tt.as_tensor_variable(val)
        if rv:
            if rv.owner and not isinstance(rv.owner.op, RandomVariable):
                raise ValueError(f"`rv` must be a RandomVariable type: {rv}")

            if rv.type.convert_variable(val) is None:
                raise ValueError(
                    ("`rv` and `val` do not have compatible types:" f" rv={rv}, val={val}")
                )
        else:
            rv = tt.NoneConst.clone()

        inputs = [val, rv]

        return tt.Apply(self, inputs, [val.type()])

    def perform(self, node, inputs, out):
        out[0][0] = inputs[0]

    def grad(self, inputs, outputs):
        return outputs


observed = Observed()
