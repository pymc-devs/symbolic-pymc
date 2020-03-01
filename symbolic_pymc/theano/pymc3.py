import logging

import numpy as np
import theano
import theano.tensor as tt

# Don't let pymc3 play with this setting!
_ctv = tt.config.compute_test_value
import pymc3 as pm

tt.config.compute_test_value = _ctv

from warnings import warn

from multipledispatch import dispatch
from unification.utils import transitive_get as walk

from theano.gof.graph import Apply, inputs as tt_inputs

from .random_variables import (
    observed,
    UniformRV,
    UniformRVType,
    NormalRV,
    NormalRVType,
    HalfNormalRV,
    HalfNormalRVType,
    MvNormalRV,
    MvNormalRVType,
    GammaRV,
    GammaRVType,
    InvGammaRV,
    InvGammaRVType,
    ExponentialRV,
    ExponentialRVType,
    CauchyRV,
    CauchyRVType,
    HalfCauchyRV,
    HalfCauchyRVType,
    BetaRV,
    BetaRVType,
    BinomialRV,
    BinomialRVType,
    PoissonRV,
    PoissonRVType,
    DirichletRV,
    DirichletRVType,
    BernoulliRV,
    BernoulliRVType,
    BetaBinomialRV,
    BetaBinomialRVType,
    CategoricalRV,
    CategoricalRVType,
    MultinomialRV,
    MultinomialRVType,
)
from .opt import FunctionGraph
from .ops import RandomVariable
from .utils import replace_input_nodes, get_rv_observation

logger = logging.getLogger("symbolic_pymc")


def tt_get_values(obj):
    """Get the value of a Theano constant or shared variable."""
    if isinstance(obj, tt.Constant):
        return obj.data
    elif isinstance(obj, theano.compile.sharedvalue.SharedVariable):
        return obj.get_value()
    else:
        raise TypeError(f"Unhandled observation type: {type(obj)}")


@dispatch(Apply, object)
def convert_rv_to_dist(node, obs):
    if not isinstance(node.op, RandomVariable):
        raise TypeError(f"{node} is not of type `RandomVariable`")

    rv = node.default_output()

    if hasattr(node, "fgraph") and hasattr(node.fgraph, "shape_feature"):
        shape = list(node.fgraph.shape_feature.shape_tuple(rv))
    else:
        shape = list(rv.shape)

    for i, s in enumerate(shape):
        try:
            shape[i] = tt.get_scalar_constant_value(s)
        except tt.NotScalarConstantError:
            shape[i] = s.tag.test_value

    dist_type, dist_params = _convert_rv_to_dist(node.op, node)
    return dist_type(rv.name, shape=shape, observed=obs, **dist_params)


@dispatch(pm.Uniform, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[UniformRV.ndim_supp :]
    res = UniformRV(dist.lower, dist.upper, size=size, rng=rng)
    return res


@dispatch(UniformRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {"lower": rv.inputs[0], "upper": rv.inputs[1]}
    return pm.Uniform, params


@convert_dist_to_rv.register(pm.Normal, object)
def convert_dist_to_rv_Normal(dist, rng):
    size = dist.shape.astype(int)[NormalRV.ndim_supp :]
    res = NormalRV(dist.mu, dist.sd, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(NormalRVType, Apply)
def _convert_rv_to_dist_Normal(op, rv):
    params = {"mu": rv.inputs[0], "sd": rv.inputs[1]}
    return pm.Normal, params


@convert_dist_to_rv.register(pm.HalfNormal, object)
def convert_dist_to_rv_HalfNormal(dist, rng):
    size = dist.shape.astype(int)[HalfNormalRV.ndim_supp :]
    res = HalfNormalRV(np.array(0.0, dtype=dist.dtype), dist.sd, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(HalfNormalRVType, Apply)
def _convert_rv_to_dist_HalfNormal(op, rv):
    assert not np.any(tt_get_values(rv.inputs[0]))
    params = {"sd": rv.inputs[1]}
    return pm.HalfNormal, params


@convert_dist_to_rv.register(pm.MvNormal, object)
def convert_dist_to_rv_MvNormal(dist, rng):
    size = dist.shape.astype(int)[MvNormalRV.ndim_supp :]
    res = MvNormalRV(dist.mu, dist.cov, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(MvNormalRVType, Apply)
def _convert_rv_to_dist_MvNormal(op, rv):
    params = {"mu": rv.inputs[0], "cov": rv.inputs[1]}
    return pm.MvNormal, params


@convert_dist_to_rv.register(pm.Gamma, object)
def convert_dist_to_rv_Gamma(dist, rng):
    size = dist.shape.astype(int)[GammaRV.ndim_supp :]
    res = GammaRV(dist.alpha, tt.inv(dist.beta), size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(GammaRVType, Apply)
def _convert_rv_to_dist_Gamma(op, rv):
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[1]}
    return pm.Gamma, params


@convert_dist_to_rv.register(pm.InverseGamma, object)
def convert_dist_to_rv_InverseGamma(dist, rng):
    size = dist.shape.astype(int)[InvGammaRV.ndim_supp :]
    res = InvGammaRV(dist.alpha, scale=dist.beta, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(InvGammaRVType, Apply)
def _convert_rv_to_dist_InvGamma(op, rv):
    assert not np.any(tt_get_values(rv.inputs[1]))
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[2]}
    return pm.InverseGamma, params


@convert_dist_to_rv.register(pm.Exponential, object)
def convert_dist_to_rv_Exponential(dist, rng):
    size = dist.shape.astype(int)[ExponentialRV.ndim_supp :]
    res = ExponentialRV(tt.inv(dist.lam), size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(ExponentialRVType, Apply)
def _convert_rv_to_dist_Exponential(op, rv):
    params = {"lam": tt.inv(rv.inputs[0])}
    return pm.Exponential, params


@convert_dist_to_rv.register(pm.Cauchy, object)
def convert_dist_to_rv_Cauchy(dist, rng):
    size = dist.shape.astype(int)[CauchyRV.ndim_supp :]
    res = CauchyRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(CauchyRVType, Apply)
def _convert_rv_to_dist_Cauchy(op, rv):
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[1]}
    return pm.Cauchy, params


@convert_dist_to_rv.register(pm.HalfCauchy, object)
def convert_dist_to_rv_HalfCauchy(dist, rng):
    size = dist.shape.astype(int)[HalfCauchyRV.ndim_supp :]
    res = HalfCauchyRV(np.array(0.0, dtype=dist.dtype), dist.beta, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(HalfCauchyRVType, Apply)
def _convert_rv_to_dist_HalfCauchy(op, rv):
    # TODO: Assert that `rv.inputs[0]` must be all zeros!
    params = {"beta": rv.inputs[1]}
    return pm.HalfCauchy, params


@convert_dist_to_rv.register(pm.Beta, object)
def convert_dist_to_rv_Beta(dist, rng):
    size = dist.shape.astype(int)[BetaRV.ndim_supp :]
    res = BetaRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(BetaRVType, Apply)
def _convert_rv_to_dist_Beta(op, rv):
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[1]}
    return pm.Beta, params


@convert_dist_to_rv.register(pm.Binomial, object)
def convert_dist_to_rv_Binomial(dist, rng):
    size = dist.shape.astype(int)[BinomialRV.ndim_supp :]
    res = BinomialRV(dist.n, dist.p, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(BinomialRVType, Apply)
def _convert_rv_to_dist_Binomial(op, rv):
    params = {"n": rv.inputs[0], "p": rv.inputs[1]}
    return pm.Binomial, params


@convert_dist_to_rv.register(pm.Poisson, object)
def convert_dist_to_rv_Poisson(dist, rng):
    size = dist.shape.astype(int)[PoissonRV.ndim_supp :]
    res = PoissonRV(dist.mu, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(PoissonRVType, Apply)
def _convert_rv_to_dist_Poisson(op, rv):
    params = {"mu": rv.inputs[0]}
    return pm.Poisson, params


@convert_dist_to_rv.register(pm.Dirichlet, object)
def convert_dist_to_rv_Dirichlet(dist, rng):
    size = dist.shape.astype(int)[DirichletRV.ndim_supp :]
    res = DirichletRV(dist.a, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(DirichletRVType, Apply)
def _convert_rv_to_dist_Dirichlet(op, rv):
    params = {"a": rv.inputs[0]}
    return pm.Dirichlet, params


@convert_dist_to_rv.register(pm.Bernoulli, object)
def convert_dist_to_rv_Bernoulli(dist, rng):
    size = dist.shape.astype(int)[BernoulliRV.ndim_supp :]
    res = BernoulliRV(dist.p, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(BernoulliRVType, Apply)
def _convert_rv_to_dist_Bernoulli(op, rv):
    params = {"p": rv.inputs[0]}
    return pm.Bernoulli, params


@convert_dist_to_rv.register(pm.BetaBinomial, object)
def convert_dist_to_rv_BetaBinomial(dist, rng):
    size = dist.shape.astype(int)[BetaBinomialRV.ndim_supp :]
    res = BetaBinomialRV(dist.n, dist.alpha, dist.beta, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(BetaBinomialRVType, Apply)
def _convert_rv_to_dist_BetaBinomial(op, rv):
    params = {"n": rv.inputs[0], "alpha": rv.inputs[1], "beta": rv.inputs[2]}
    return pm.BetaBinomial, params


@convert_dist_to_rv.register(pm.Categorical, object)
def convert_dist_to_rv_Categorical(dist, rng):
    size = dist.shape.astype(int)[CategoricalRV.ndim_supp :]
    res = CategoricalRV(dist.p, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(CategoricalRVType, Apply)
def _convert_rv_to_dist_Categorical(op, rv):
    params = {"p": rv.inputs[0]}
    return pm.Categorical, params


@convert_dist_to_rv.register(pm.Multinomial, object)
def convert_dist_to_rv_Multinomial(dist, rng):
    size = dist.shape.astype(int)[MultinomialRV.ndim_supp :]
    res = MultinomialRV(dist.n, dist.p, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(MultinomialRVType, Apply)
def _convert_rv_to_dist_Multinomial(op, rv):
    params = {"n": rv.inputs[0], "p": rv.inputs[1]}
    return pm.Multinomial, params


# TODO: More RV conversions!


def pymc3_var_to_rv(pm_var, rand_state=None):
    """Convert a PyMC3 random variable into a `RandomVariable`."""
    dist = pm_var.distribution
    new_rv = convert_dist_to_rv(dist, rand_state)
    new_rv.name = pm_var.name

    if isinstance(pm_var, pm.model.ObservedRV):
        obs = pm_var.observations
        # For some reason, the observations can be float when the RV's dtype is
        # not.
        if obs.dtype != pm_var.dtype:
            obs = obs.astype(pm_var.dtype)

        obs = tt.as_tensor_variable(obs)

        if getattr(obs, "cached", False):
            obs = obs.clone()

        new_rv = observed(obs, new_rv)

    # Let's attempt to fix the PyMC3 broadcastable dims "oracle" issue,
    # if present.  We'll basically find the dimensions PyMC3 says
    # are broadcastable--but don't need to be--and restrict our
    # `RandomVariable`s to be broadcastable there, too.
    diff_bcasts = tuple(
        i
        for i, (a, b) in enumerate(zip(pm_var.type.broadcastable, new_rv.type.broadcastable))
        if a > b
    )

    if len(diff_bcasts) > 0:
        warn(
            f"The tensor type for {pm_var} has an overly restrictive"
            " broadcast dimension.  Try re-creating the model without"
            " specifying a shape with a dimension value of 1"
            " (e.g. `(1,)`)."
        )
        new_rv = tt.addbroadcast(new_rv, *diff_bcasts)

    return new_rv


def rec_conv_to_rv(v, replacements, model, rand_state=None):
    """Recursively convert a PyMC3 random variable to a Theano graph."""
    if v in replacements:
        return walk(v, replacements)
    elif v.name and pm.util.is_transformed_name(v.name):
        untrans_name = pm.util.get_untransformed_name(v.name)
        v_untrans = getattr(model, untrans_name)

        rv_new = rec_conv_to_rv(v_untrans, replacements, model, rand_state=rand_state)
        replacements[v] = rv_new
        return rv_new
    elif hasattr(v, "distribution"):
        rv = pymc3_var_to_rv(v, rand_state=rand_state)

        rv_ins = []
        for i in tt_inputs([rv]):
            i_rv = rec_conv_to_rv(i, replacements, model, rand_state=rand_state)

            if i_rv is not None:
                replacements[i] = i_rv
                rv_ins.append(i_rv)
            else:
                rv_ins.append(i)

        _ = replace_input_nodes(rv_ins, [rv], memo=replacements, clone_inputs=False)

        rv_new = walk(rv, replacements)

        replacements[v] = rv_new

        return rv_new
    else:
        return None


def model_graph(pymc_model, output_vars=None, rand_state=None, attach_memo=True):
    """Convert a PyMC3 model into a Theano `FunctionGraph`.

    Parameters
    ----------
    pymc_model: `Model`
        A PyMC3 model object.
    output_vars: list (optional)
        Variables to use as `FunctionGraph` outputs.  If not specified,
        the model's observed random variables are used.
    rand_state: Numpy rng (optional)
        When converting to `RandomVariable`s, use this random state object.
    attach_memo: boolean (optional)
        Add a property to the returned `FunctionGraph` name `memo` that
        contains the mappings between PyMC and `RandomVariable` terms.

    Results
    -------
    out: `FunctionGraph`

    """
    model = pm.modelcontext(pymc_model)
    replacements = {}

    if output_vars is None:
        output_vars = list(model.observed_RVs)
    if rand_state is None:
        rand_state = theano.shared(np.random.RandomState())

    replacements = {}
    # First pass...
    for i, o in enumerate(output_vars):
        _ = rec_conv_to_rv(o, replacements, model, rand_state=rand_state)
        output_vars[i] = walk(o, replacements)

    output_vars = [walk(o, replacements) for o in output_vars]

    fg_features = [tt.opt.ShapeFeature()]
    model_fg = FunctionGraph(
        [i for i in tt_inputs(output_vars) if not isinstance(i, tt.Constant)],
        output_vars,
        clone=True,
        memo=replacements,
        features=fg_features,
    )
    if attach_memo:
        model_fg.memo = replacements

    return model_fg


def graph_model(graph, *model_args, **model_kwargs):
    """Create a PyMC3 model from a Theano graph with `RandomVariable` nodes."""
    model = pm.Model(*model_args, **model_kwargs)

    fgraph = graph
    if not isinstance(fgraph, FunctionGraph):
        fgraph = FunctionGraph(tt.gof.graph.inputs([fgraph]), [fgraph])

    nodes = [n for n in fgraph.toposort() if isinstance(n.op, RandomVariable)]
    rv_replacements = {}

    for node in nodes:

        obs = get_rv_observation(node)

        if obs is not None:
            obs = obs.inputs[0]

            obs = tt_get_values(obs)

        old_rv_var = node.default_output()

        rv_var = theano.scan_module.scan_utils.clone(old_rv_var, replace=rv_replacements)

        node = rv_var.owner

        # Make sure there are only PyMC3 vars in the result.
        assert not any(
            isinstance(op.op, RandomVariable)
            for op in theano.gof.graph.ops(tt_inputs([rv_var]), [rv_var])
            if op != node
        )

        with model:
            rv = convert_rv_to_dist(node, obs)

        rv_replacements[old_rv_var] = rv

    model.rv_replacements = rv_replacements

    return model
