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
from .utils import replace_nodes

logger = logging.getLogger("symbolic_pymc")


@dispatch(Apply, object)
def convert_rv_to_pymc(node, obs):
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

    dist_type, dist_params = _convert_rv_to_pymc(node.op, node)
    return dist_type(rv.name,
                     shape=shape,
                     observed=obs,
                     **dist_params)


@dispatch(pm.distributions.transforms.TransformedDistribution, object)
def convert_pymc_to_rv(dist, rng):
    # TODO: Anything more to do with the transform information?
    return convert_pymc_to_rv(dist.dist, rng)


@dispatch(pm.Uniform, object)
def convert_pymc_to_rv(dist, rng):
    size = dist.shape.astype(int)[UniformRV.ndim_supp:]
    res = UniformRV(dist.lower, dist.upper, size=size, rng=rng)
    return res


@dispatch(UniformRVType, Apply)
def _convert_rv_to_pymc(op, rv):
    params = {'lower': rv.inputs[0],
              'upper': rv.inputs[1]}
    return pm.Uniform, params


@dispatch(pm.Normal, object)
def convert_pymc_to_rv(dist, rng):
    size = dist.shape.astype(int)[NormalRV.ndim_supp:]
    res = NormalRV(dist.mu, dist.sd, size=size, rng=rng)
    return res


@dispatch(NormalRVType, Apply)
def _convert_rv_to_pymc(op, rv):
    params = {'mu': rv.inputs[0],
              'sd': rv.inputs[1]}
    return pm.Normal, params


@dispatch(pm.MvNormal, object)
def convert_pymc_to_rv(dist, rng):
    size = dist.shape.astype(int)[MvNormalRV.ndim_supp:]
    res = MvNormalRV(dist.mu, dist.cov, size=size, rng=rng)
    return res


@dispatch(MvNormalRVType, Apply)
def _convert_rv_to_pymc(op, rv):
    params = {'mu': rv.inputs[0],
              'cov': rv.inputs[1]}
    return pm.MvNormal, params


@dispatch(pm.Gamma, object)
def convert_pymc_to_rv(dist, rng):
    size = dist.shape.astype(int)[GammaRV.ndim_supp:]
    res = GammaRV(dist.alpha, tt.inv(dist.beta), size=size, rng=rng)
    return res


@dispatch(GammaRVType, Apply)
def _convert_rv_to_pymc(op, rv):
    params = {'alpha': rv.inputs[0],
              'beta': rv.inputs[1]}
    return pm.Gamma, params


@dispatch(pm.InverseGamma, object)
def convert_pymc_to_rv(dist, rng):
    size = dist.shape.astype(int)[InvGammaRV.ndim_supp:]
    res = InvGammaRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@dispatch(InvGammaRVType, Apply)
def _convert_rv_to_pymc(op, rv):
    params = {'alpha': rv.inputs[0],
              'beta': rv.inputs[1]}
    return pm.InverseGamma, params


@dispatch(pm.Exponential, object)
def convert_pymc_to_rv(dist, rng):
    size = dist.shape.astype(int)[ExponentialRV.ndim_supp:]
    res = ExponentialRV(tt.inv(dist.lam), size=size, rng=rng)
    return res


@dispatch(ExponentialRVType, Apply)
def _convert_rv_to_pymc(op, rv):
    params = {'lam': tt.inv(rv.inputs[0])}
    return pm.Exponential, params


@dispatch(pm.Cauchy, object)
def convert_pymc_to_rv(dist, rng):
    size = dist.shape.astype(int)[CauchyRV.ndim_supp:]
    res = CauchyRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@dispatch(CauchyRVType, Apply)
def _convert_rv_to_pymc(op, rv):
    params = {'alpha': rv.inputs[0],
              'beta': rv.inputs[1]}
    return pm.Cauchy, params


@dispatch(pm.HalfCauchy, object)
def convert_pymc_to_rv(dist, rng):
    size = dist.shape.astype(int)[HalfCauchyRV.ndim_supp:]
    res = HalfCauchyRV(0.0, dist.beta, size=size, rng=rng)
    return res


@dispatch(HalfCauchyRVType, Apply)
def _convert_rv_to_pymc(op, rv):
    # TODO: Assert that `rv.inputs[0]` must be all zeros!
    params = {'beta': rv.inputs[1]}
    return pm.HalfCauchy, params

# TODO: More RV conversions!


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

    if output_vars is None:
        output_vars = [o for o in model.observed_RVs]

    # Make sure the distribution and observation info sticks around through all
    # the object cloning.
    # Also, we don't want to use the transformed variables, and, since some
    # distribution objects use them (instead of their non-transformed
    # counterparts), we need to replace them at some point.
    model_vars = model.vars + model.observed_RVs + model.deterministics
    model_vars = {v.name: v for v in model_vars}
    for name, v in model_vars.items():
        if pm.util.is_transformed_name(name):
            untrans_name = pm.util.get_untransformed_name(name)
            v.tag.untransformed = model_vars[untrans_name]
        if hasattr(v, 'distribution'):
            v.tag.distribution = v.distribution
        if hasattr(v, 'observations'):
            # isinstance(k, pm.model.ObservedRV)

            # If the observation variable we obtained is *not* identical/equal
            # to the corresponding observed RV's owner's input, then we'll
            # have duplicates (and a bad graph).  Generally, `k.observations`
            # is not identical to the `ViewOp` argument, so we use the former
            # to avoid dups.
            if isinstance(v.owner.op, theano.compile.ops.ViewOp):
                v.tag.observations = v.owner.inputs[0]
            else:
                # TODO: Is it ever not a `ViewOp`?
                v.tag.observations = v.observations

    if not output_vars:
        raise ValueError('No derived or observable variables specified')

    model_inputs = [
        inp for inp in tt_inputs(output_vars)
    ]

    model_memo = clone_get_equiv(model_inputs, output_vars,
                                 copy_orphans=False)

    fg_features = [tt.opt.ShapeFeature()]
    model_fg = FunctionGraph([model_memo[i] for i in model_inputs],
                             [model_memo[i] for i in output_vars],
                             clone=False, features=fg_features)
    model_fg.memo = model_memo
    model_fg.rev_memo = {v: k for k, v in model_memo.items()}

    convert_pymc3_rvs(model_fg, clone=False, rand_state=rand_state)

    return model_fg


def convert_pymc3_rvs(fgraph, clone=True, rand_state=None):
    """Replace PyMC3 random variables with `RandomFunction` Ops.

    TODO: Could use a Theano graph `Feature` to trace--or even
    replace--random variables.

    Parameters
    ----------
    fgraph: FunctionGraph
        A graph containing PyMC3 random variables.

    clone: bool, optional
        Clone the original graph.

    rand_state: RandomStateType, optional
        The Theano random state.

    Returns
    -------
    out: A cloned graph with random variables replaced and a `memo` attribute.
    """
    if clone:
        fgraph_, fgraph_memo_ = fgraph.clone_get_equiv(attach_feature=False)
        fgraph_.memo = fgraph_memo_
        fgraph_.rev_memo = {v: k for k, v in fgraph_memo_.items()}
    else:
        fgraph_ = fgraph
        if not isinstance(fgraph, FunctionGraph):
            warn(
                "Use symbolic_pymc's FunctionGraph implementation;"
                "otherwise, MergeOptimizer will remove RandomVariables.")
        assert hasattr(fgraph_, 'memo')
        assert hasattr(fgraph_, 'rev_memo')

    if rand_state is None:
        rand_state = theano.shared(np.random.RandomState())

    # fgraph_replacements = {}
    nodes = set(o for o in fgraph_.outputs
                if hasattr(o.tag, 'distribution'))
    while nodes:
        pm_var = nodes.pop()

        logger.debug(f'creating a RandomVariable for {pm_var}')

        if getattr(pm_var.tag, 'untransformed', None):

            _pm_var = pm_var.tag.untransformed

            logger.debug(f'{pm_var} is a transform of {_pm_var}')

            # Make a copy of this var's sub-graph.
            nodes_updates = replace_nodes(
                tt_inputs([_pm_var]), [_pm_var], memo=fgraph_.memo)
            _pm_var = nodes_updates.get(_pm_var, _pm_var)

            fgraph_.replace_validate(pm_var, _pm_var)

            if pm_var in fgraph_.inputs:
                fgraph_.inputs.remove(pm_var)

            # From here on, always replace the transformed with the
            # untransformed.
            fgraph_.memo[pm_var] = _pm_var
            pm_var = _pm_var

        dist = pm_var.tag.distribution

        new_rv = convert_pymc_to_rv(dist, rand_state)

        new_rv.name = pm_var.name

        if isinstance(pm_var, pm.model.ObservedRV):
            logger.debug(f'{pm_var} is an observed variable.')
            new_rv = observed(pm_var.tag.observations, new_rv)

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

        logger.debug(f'{pm_var} converted to {new_rv.owner}.')

        # The variables in these distribution objects--and the new
        # `RandomVariable`--are *not* the same as the ones in the fgraph
        # object (those ones are clones)!
        # We need to use the memo mappings to replace those old variables
        # *and* we need to clone any new variables introduced by the
        # `distribution` object (e.g. variables in distribution parameters).
        nodes_updates = replace_nodes(
            tt_inputs([new_rv]), [new_rv], memo=fgraph_.memo)
        # It's possible that our new `RandomVariable` itself was cloned,
        # in which case we need to use the new one.
        new_rv = nodes_updates.get(new_rv, new_rv)

        # Update the memo so that next time around we don't
        # needlessly clone objects already in the fgraph.
        # fgraph_.memo.update(nodes_updates)

        for i in tt_inputs([new_rv]):
            # We've probably introduced new inputs via the distribution
            # parameters.
            if i not in fgraph_.inputs:
                logger.debug(f'{new_rv} introduces new input {i}')
                fgraph_.add_input(i)

            # Add any new PyMC3 RVs that need to be converted.
            if hasattr(i.tag, 'distribution'):
                logger.debug(f'{new_rv} introduces new random variable {i}')
                nodes.add(i)

        # Finally, replace the old PyMC3 RV with the new one.
        assert pm_var in fgraph_.variables
        fgraph_.replace_validate(pm_var, new_rv)

        # Finally, remove the unused inputs.  For instance, if the original
        # inputs were PyMC3 RVs, then they've been replaced; however,
        # `FunctionGraph.replace` won't remove them for some reason.
        # TODO: Create a Theano issue/PR for this?
        if pm_var in fgraph_.inputs:
            fgraph_.inputs.remove(pm_var)

    fgraph_.check_integrity()

    return fgraph_


def graph_model(fgraph, *model_args, **model_kwargs):
    """Create a PyMC3 model from a Theano graph with `RandomVariable`
    nodes.
    """
    with pm.Model(*model_args, **model_kwargs) as model:
        for node in filter(lambda x: isinstance(x.op, RandomVariable),
                           fgraph.apply_nodes):
            rv = node.default_output()
            obs = None
            for cl_node, i in fgraph.clients(rv):
                if cl_node == 'output':
                    cl_node = fgraph.outputs[i].owner
                if isinstance(cl_node.op, Observed):
                    obs = cl_node.inputs[0]
                    break

            new_pm_rv = convert_rv_to_pymc(node, obs)

    return model
