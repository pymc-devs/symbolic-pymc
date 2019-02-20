import logging

import numpy as np
import theano
import theano.tensor as tt

# Don't let pymc3 play with this setting!
_ctv = tt.config.compute_test_value
import pymc3 as pm
tt.config.compute_test_value = _ctv

from warnings import warn
from itertools import chain

from multipledispatch import dispatch

from theano.gof.graph import (Variable, Apply, inputs as tt_inputs,
                              clone_get_equiv)

from . import (Observed, observed,
               UniformRV, UniformRVType,
               NormalRV, NormalRVType,
               MvNormalRV, MvNormalRVType,
               GammaRV, GammaRVType,
               InvGammaRV, InvGammaRVType,
               ExponentialRV, ExponentialRVType,
               CauchyRV, CauchyRVType,
               HalfCauchyRV, HalfCauchyRVType,
               # MultinomialRV, MultinomialRVType,
               # DirichletRV, DirichletRVType,
               # PoissonRV, PoissonRVType,
)
from .opt import FunctionGraph
from .rv import RandomVariable
from .utils import replace_input_nodes, get_rv_observation

logger = logging.getLogger("symbolic_pymc")


@dispatch(Apply, object)
def convert_rv_to_dist(node, obs):
    if not isinstance(node.op, RandomVariable):
        raise TypeError(f'{node} is not of type `RandomVariable`')

    rv = node.default_output()

    if hasattr(node, 'fgraph') and hasattr(node.fgraph, 'shape_feature'):
        shape = list(node.fgraph.shape_feature.shape_tuple(rv))
    else:
        shape = list(rv.shape)

    for i, s in enumerate(shape):
        try:
            shape[i] = tt.get_scalar_constant_value(s)
        except tt.NotScalarConstantError:
            shape[i] = s.tag.test_value

    dist_type, dist_params = _convert_rv_to_dist(node.op, node)
    return dist_type(rv.name,
                     shape=shape,
                     observed=obs,
                     **dist_params)


# @dispatch(pm.distributions.transforms.TransformedDistribution, object)
# def convert_dist_to_rv(dist, rng):
#     # TODO: Anything more to do with the transform information?
#     return convert_dist_to_rv(dist.dist, rng)


@dispatch(pm.Uniform, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[UniformRV.ndim_supp:]
    res = UniformRV(dist.lower, dist.upper, size=size, rng=rng)
    return res


@dispatch(UniformRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {'lower': rv.inputs[0],
              'upper': rv.inputs[1]}
    return pm.Uniform, params


@dispatch(pm.Normal, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[NormalRV.ndim_supp:]
    res = NormalRV(dist.mu, dist.sd, size=size, rng=rng)
    return res


@dispatch(NormalRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {'mu': rv.inputs[0],
              'sd': rv.inputs[1]}
    return pm.Normal, params


@dispatch(pm.MvNormal, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[MvNormalRV.ndim_supp:]
    res = MvNormalRV(dist.mu, dist.cov, size=size, rng=rng)
    return res


@dispatch(MvNormalRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {'mu': rv.inputs[0],
              'cov': rv.inputs[1]}
    return pm.MvNormal, params


@dispatch(pm.Gamma, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[GammaRV.ndim_supp:]
    res = GammaRV(dist.alpha, tt.inv(dist.beta), size=size, rng=rng)
    return res


@dispatch(GammaRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {'alpha': rv.inputs[0],
              'beta': rv.inputs[1]}
    return pm.Gamma, params


@dispatch(pm.InverseGamma, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[InvGammaRV.ndim_supp:]
    res = InvGammaRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@dispatch(InvGammaRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {'alpha': rv.inputs[0],
              'beta': rv.inputs[1]}
    return pm.InverseGamma, params


@dispatch(pm.Exponential, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[ExponentialRV.ndim_supp:]
    res = ExponentialRV(tt.inv(dist.lam), size=size, rng=rng)
    return res


@dispatch(ExponentialRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {'lam': tt.inv(rv.inputs[0])}
    return pm.Exponential, params


@dispatch(pm.Cauchy, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[CauchyRV.ndim_supp:]
    res = CauchyRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@dispatch(CauchyRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {'alpha': rv.inputs[0],
              'beta': rv.inputs[1]}
    return pm.Cauchy, params


@dispatch(pm.HalfCauchy, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[HalfCauchyRV.ndim_supp:]
    res = HalfCauchyRV(0.0, dist.beta, size=size, rng=rng)
    return res


@dispatch(HalfCauchyRVType, Apply)
def _convert_rv_to_dist(op, rv):
    # TODO: Assert that `rv.inputs[0]` must be all zeros!
    params = {'beta': rv.inputs[1]}
    return pm.HalfCauchy, params

# TODO: More RV conversions!


def pymc3_var_to_rv(pm_var, rand_state=None):
    """Convert a PyMC3 random variable into a `RandomVariable`
    """
    dist = pm_var.distribution
    new_rv = convert_dist_to_rv(dist, rand_state)
    new_rv.name = pm_var.name

    if isinstance(pm_var, pm.model.ObservedRV):
        obs = tt.as_tensor_variable(pm_var.observations)
        new_rv = observed(obs, new_rv)

    # Let's attempt to fix the PyMC3 broadcastable dims "oracle" issue,
    # if present.  We'll basically find the dimensions PyMC3 says
    # are broadcastable--but don't need to be--and restrict our
    # `RandomVariable`s to be broadcastable there, too.
    diff_bcasts = tuple(
        i for i, (a, b) in enumerate(
            zip(pm_var.type.broadcastable,
                new_rv.type.broadcastable))
        if a > b)

    if len(diff_bcasts) > 0:
        warn(f'The tensor type for {pm_var} has an overly restrictive'
             ' broadcast dimension.  Try re-creating the model without'
             ' specifying a shape with a dimension value of 1'
             ' (e.g. `(1,)`).')
        new_rv = tt.addbroadcast(new_rv, *diff_bcasts)

    return new_rv


def model_graph(pymc_model, output_vars=None, convert_rvs=True,
                rand_state=None):
    """Convert a PyMC3 model into a Theano `FunctionGraph`.

    Parameters
    ==========
    pymc_model: `Model`
        A PyMC3 model object.
    output_vars: list (optional)
        Variables to use as `FunctionGraph` outputs.  If not specified,
        the model's observed random variables are used.
    rand_state: Numpy rng (optional)
        When converting to `RandomVariable`s, use this random state object.

    Results
    =======
    out: `FunctionGraph`
    """
    model = pm.modelcontext(pymc_model)
    replacements = {}
    topo_sorted_rvs = []

    if rand_state is None:
        rand_state = theano.shared(np.random.RandomState())

    for v in theano.gof.graph.io_toposort(
            theano.gof.graph.inputs([model.varlogpt] + model.observed_RVs),
            [model.varlogpt] + model.observed_RVs):
        for i in v.inputs + v.outputs:
            if isinstance(i, pm.Factor):
                if i.name and pm.util.is_transformed_name(i.name):
                    untrans_name = pm.util.get_untransformed_name(i.name)
                    i_untrans = getattr(model, untrans_name)
                    replacements[i] = i_untrans
                    i = i_untrans

                if i not in topo_sorted_rvs:
                    topo_sorted_rvs.append(i)
                    old_rv_var = pymc3_var_to_rv(i, rand_state=rand_state)
                    rv_var = theano.scan_module.scan_utils.clone(
                        old_rv_var, replace=replacements)
                    replacements[i] = rv_var

    obs_rvs = [replacements[o] for o in model.observed_RVs]

    fg_features = [tt.opt.ShapeFeature()]
    model_fg = FunctionGraph([i for i in tt_inputs(obs_rvs)
                              if not isinstance(i, tt.Constant)],
                             obs_rvs,
                             clone=True,
                             features=fg_features)
    return model_fg


def graph_model(fgraph, *model_args, **model_kwargs):
    """Create a PyMC3 model from a Theano graph with `RandomVariable`
    nodes.
    """
    model = pm.Model(*model_args, **model_kwargs)
    nodes = [n for n in fgraph.toposort()
             if isinstance(n.op, RandomVariable)]
    rv_replacements = {}

    for node in nodes:

        obs = get_rv_observation(node)

        if obs is not None:
            obs = obs.inputs[0]

            if isinstance(obs, tt.Constant):
                obs = obs.data
            elif isinstance(
                    obs, theano.compile.sharedvalue.SharedVariable):
                obs = obs.get_value()
            else:
                raise TypeError(
                    f'Unhandled observation type: {type(obs)}')

        old_rv_var = node.default_output()

        rv_var = theano.scan_module.scan_utils.clone(
            old_rv_var, replace=rv_replacements)

        node = rv_var.owner

        # Make sure there are only PyMC3 vars in the result.
        assert not any(isinstance(op.op, RandomVariable)
                       for op in theano.gof.graph.ops(
                               tt_inputs([rv_var]), [rv_var])
                       if op != node)

        with model:
            rv = convert_rv_to_dist(node, obs)

        rv_replacements[old_rv_var] = rv

    model.rv_replacements = rv_replacements

    return model
