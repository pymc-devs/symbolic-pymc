import numpy as np
import theano
import scipy.stats as stats
import theano.tensor as tt

from functools import partial

try:
    from pypolyagamma import PyPolyaGamma
except ImportError:  # pragma: no cover

    def PyPolyaGamma(*args, **kwargs):
        raise RuntimeError("pypolygamma not installed!")


from .ops import RandomVariable, param_supp_shape_fn


class UniformRVType(RandomVariable):
    print_name = ("U", "\\operatorname{U}")

    def __init__(self):
        super().__init__("uniform", theano.config.floatX, 0, [0, 0], "uniform", inplace=True)

    def make_node(self, lower, upper, size=None, rng=None, name=None):
        return super().make_node(lower, upper, size=size, rng=rng, name=name)


UniformRV = UniformRVType()


class BetaRVType(RandomVariable):
    print_name = ("Beta", "\\operatorname{Beta}")

    def __init__(self):
        super().__init__("beta", theano.config.floatX, 0, [0, 0], "beta", inplace=True)

    def make_node(self, alpha, beta, size=None, rng=None, name=None):
        return super().make_node(alpha, beta, size=size, rng=rng, name=name)


BetaRV = BetaRVType()


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
            lambda rng, *args: stats.halfnorm.rvs(*args, random_state=rng),
            inplace=True,
        )

    def make_node(self, mu=0.0, sigma=1.0, size=None, rng=None, name=None):
        return super().make_node(mu, sigma, size=size, rng=rng, name=name)


HalfNormalRV = HalfNormalRVType()


class GammaRVType(RandomVariable):
    print_name = ("Gamma", "\\operatorname{Gamma}")

    def __init__(self):
        super().__init__(
            "gamma",
            theano.config.floatX,
            0,
            [0, 0],
            lambda rng, shape, rate, size: stats.gamma.rvs(
                shape, scale=1.0 / rate, size=size, random_state=rng
            ),
            inplace=True,
        )

    def make_node(self, shape, rate, size=None, rng=None, name=None):
        return super().make_node(shape, rate, size=size, rng=rng, name=name)


GammaRV = GammaRVType()


class ExponentialRVType(RandomVariable):
    print_name = ("Exp", "\\operatorname{Exp}")

    def __init__(self):
        super().__init__("exponential", theano.config.floatX, 0, [0], "exponential", inplace=True)

    def make_node(self, scale, size=None, rng=None, name=None):
        return super().make_node(scale, size=size, rng=rng, name=name)


ExponentialRV = ExponentialRVType()


class MvNormalRVType(RandomVariable):
    print_name = ("N", "\\operatorname{N}")

    def __init__(self):
        super().__init__(
            "multivariate_normal", theano.config.floatX, 1, [1, 2], self._smpl_fn, inplace=True,
        )

    @classmethod
    def _smpl_fn(cls, rng, mean, cov, size):
        res = np.atleast_1d(
            stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True).rvs(
                size=size, random_state=rng
            )
        )

        if size is not None:
            res = res.reshape(list(size) + [-1])

        return res

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


class PoissonRVType(RandomVariable):
    print_name = ("Pois", "\\operatorname{Pois}")

    def __init__(self):
        super().__init__("poisson", "int64", 0, [0], "poisson", inplace=True)

    def make_node(self, rate, size=None, rng=None, name=None):
        return super().make_node(rate, size=size, rng=rng, name=name)


PoissonRV = PoissonRVType()


class CauchyRVType(RandomVariable):
    print_name = ("C", "\\operatorname{C}")

    def __init__(self):
        super().__init__(
            "cauchy",
            theano.config.floatX,
            0,
            [0, 0],
            lambda rng, *args: stats.cauchy.rvs(*args, random_state=rng),
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
            lambda rng, *args: stats.halfcauchy.rvs(*args, random_state=rng),
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
            lambda rng, shape, rate, size: stats.invgamma.rvs(
                shape, scale=rate, size=size, random_state=rng
            ),
            inplace=True,
        )

    def make_node(self, shape, rate=1.0, size=None, rng=None, name=None):
        return super().make_node(shape, rate, size=size, rng=rng, name=name)


InvGammaRV = InvGammaRVType()


class TruncExponentialRVType(RandomVariable):
    print_name = ("TruncExp", "\\operatorname{Exp}")

    def __init__(self):
        super().__init__(
            "truncexpon",
            theano.config.floatX,
            0,
            [0, 0, 0],
            lambda rng, *args: stats.truncexpon.rvs(*args, random_state=rng),
            inplace=True,
        )

    def make_node(self, b, loc=0.0, scale=1.0, size=None, rng=None, name=None):
        return super().make_node(b, loc, scale, size=size, rng=rng, name=name)


TruncExponentialRV = TruncExponentialRVType()


class BernoulliRVType(RandomVariable):
    print_name = ("Bern", "\\operatorname{Bern}")

    def __init__(self):
        super().__init__(
            "bernoulli",
            "int64",
            0,
            [0],
            lambda rng, *args: stats.bernoulli.rvs(args[0], size=args[1], random_state=rng),
            inplace=True,
        )

    def make_node(self, p, size=None, rng=None, name=None):
        return super().make_node(p, size=size, rng=rng, name=name)


BernoulliRV = BernoulliRVType()


class BinomialRVType(RandomVariable):
    print_name = ("Binom", "\\operatorname{Binom}")

    def __init__(self):
        super().__init__("binomial", "int64", 0, [0, 0], "binomial", inplace=True)

    def make_node(self, n, p, size=None, rng=None, name=None):
        return super().make_node(n, p, size=size, rng=rng, name=name)


BinomialRV = BinomialRVType()


class NegBinomialRVType(RandomVariable):
    print_name = ("NB", "\\operatorname{NB}")

    def __init__(self):
        super().__init__(
            "neg-binomial",
            "int64",
            0,
            [0, 0],
            lambda rng, n, p, size: stats.nbinom.rvs(n, p, size=size, random_state=rng),
            inplace=True,
        )

    def make_node(self, n, p, size=None, rng=None, name=None):
        return super().make_node(n, p, size=size, rng=rng, name=name)


NegBinomialRV = NegBinomialRVType()


class BetaBinomialRVType(RandomVariable):
    print_name = ("BetaBinom", "\\operatorname{BetaBinom}")

    def __init__(self):
        super().__init__(
            "beta_binomial",
            "int64",
            0,
            [0, 0, 0],
            lambda rng, *args: stats.betabinom.rvs(*args[:-1], size=args[-1], random_state=rng),
            inplace=True,
        )

    def make_node(self, n, a, b, size=None, rng=None, name=None):
        return super().make_node(n, a, b, size=size, rng=rng, name=name)


BetaBinomialRV = BetaBinomialRVType()


class MultinomialRVType(RandomVariable):
    """A Multinomial random variable type.

    FYI: Support shape is determined by the first dimension in the *second*
    parameter (i.e.  the probabilities vector).

    """

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


class CategoricalRVType(RandomVariable):
    print_name = ("Cat", "\\operatorname{Cat}")

    def __init__(self):
        super().__init__(
            "categorical",
            "int64",
            0,
            [1],
            lambda rng, *args: stats.rv_discrete(values=(range(len(args[0])), args[0])).rvs(
                size=args[1], random_state=rng
            ),
            inplace=True,
        )

    def make_node(self, pvals, size=None, rng=None, name=None):
        return super().make_node(pvals, size=size, rng=rng, name=name)


CategoricalRV = CategoricalRVType()


class PolyaGammaRVType(RandomVariable):
    """Polya-Gamma random variable.

    XXX: This doesn't really use the given RNG, due to the narrowness of the
    sampler package's implementation.
    """

    print_name = ("PG", "\\operatorname{PG}")

    def __init__(self):
        super().__init__(
            "polya-gamma", theano.config.floatX, 0, [0, 0], self._smpl_fn, inplace=True,
        )

    def make_node(self, b, c, size=None, rng=None, name=None):
        return super().make_node(b, c, size=size, rng=rng, name=name)

    @classmethod
    def _smpl_fn(cls, rng, b, c, size):
        pg = PyPolyaGamma(rng.randint(2 ** 16))

        if not size and b.shape == c.shape == ():
            return pg.pgdraw(b, c)
        else:
            b, c = np.broadcast_arrays(b, c)
            out_shape = b.shape + tuple(size or ())
            smpl_val = np.empty(out_shape, dtype="double")
            b = np.tile(b, tuple(size or ()) + (1,))
            c = np.tile(c, tuple(size or ()) + (1,))
            pg.pgdrawv(
                np.asarray(b.flat).astype("double", copy=True),
                np.asarray(c.flat).astype("double", copy=True),
                np.asarray(smpl_val.flat),
            )
            return smpl_val


PolyaGammaRV = PolyaGammaRVType()


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
