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
)
from .opt import FunctionGraph
from .ops import RandomVariable
from .utils import replace_input_nodes, get_rv_observation

logger = logging.getLogger("symbolic_pymc")


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


@dispatch(pm.Normal, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[NormalRV.ndim_supp :]
    res = NormalRV(dist.mu, dist.sd, size=size, rng=rng)
    return res


@dispatch(NormalRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {"mu": rv.inputs[0], "sd": rv.inputs[1]}
    return pm.Normal, params


@dispatch(pm.HalfNormal, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[HalfNormalRV.ndim_supp :]
    res = HalfNormalRV(np.array(0.0, dtype=dist.dtype), dist.sd, size=size, rng=rng)
    return res


@dispatch(HalfNormalRVType, Apply)
def _convert_rv_to_dist(op, rv):
    # TODO: Assert that `rv.inputs[0]` must be all zeros!
    params = {"sd": rv.inputs[1]}
    return pm.HalfNormal, params


@dispatch(pm.MvNormal, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[MvNormalRV.ndim_supp :]
    res = MvNormalRV(dist.mu, dist.cov, size=size, rng=rng)
    return res


@dispatch(MvNormalRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {"mu": rv.inputs[0], "cov": rv.inputs[1]}
    return pm.MvNormal, params


@dispatch(pm.Gamma, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[GammaRV.ndim_supp :]
    res = GammaRV(dist.alpha, tt.inv(dist.beta), size=size, rng=rng)
    return res


@dispatch(GammaRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[1]}
    return pm.Gamma, params


@dispatch(pm.InverseGamma, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[InvGammaRV.ndim_supp :]
    res = InvGammaRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@dispatch(InvGammaRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[1]}
    return pm.InverseGamma, params


@dispatch(pm.Exponential, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[ExponentialRV.ndim_supp :]
    res = ExponentialRV(tt.inv(dist.lam), size=size, rng=rng)
    return res


@dispatch(ExponentialRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {"lam": tt.inv(rv.inputs[0])}
    return pm.Exponential, params


@dispatch(pm.Cauchy, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[CauchyRV.ndim_supp :]
    res = CauchyRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@dispatch(CauchyRVType, Apply)
def _convert_rv_to_dist(op, rv):
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[1]}
    return pm.Cauchy, params


@dispatch(pm.HalfCauchy, object)
def convert_dist_to_rv(dist, rng):
    size = dist.shape.astype(int)[HalfCauchyRV.ndim_supp :]
    res = HalfCauchyRV(np.array(0.0, dtype=dist.dtype), dist.beta, size=size, rng=rng)
    return res


@dispatch(HalfCauchyRVType, Apply)
def _convert_rv_to_dist(op, rv):
    # TODO: Assert that `rv.inputs[0]` must be all zeros!
    params = {"beta": rv.inputs[1]}
    return pm.HalfCauchy, params


# TODO: More RV conversions!


def pymc3_var_to_rv(pm_var, rand_state=None):
    """Convert a PyMC3 random variable into a `RandomVariable`."""
    dist = pm_var.distribution
    new_rv = convert_dist_to_rv(dist, rand_state)
    new_rv.name = pm_var.name

    if isinstance(pm_var, pm.model.ObservedRV):
        obs = tt.as_tensor_variable(pm_var.observations)
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


def graph_model(fgraph, *model_args, **model_kwargs):
    """Create a PyMC3 model from a Theano graph with `RandomVariable` nodes."""
    model = pm.Model(*model_args, **model_kwargs)
    nodes = [n for n in fgraph.toposort() if isinstance(n.op, RandomVariable)]
    rv_replacements = {}

    for node in nodes:

        obs = get_rv_observation(node)

        if obs is not None:
            obs = obs.inputs[0]

            if isinstance(obs, tt.Constant):
                obs = obs.data
            elif isinstance(obs, theano.compile.sharedvalue.SharedVariable):
                obs = obs.get_value()
            else:
                raise TypeError(f"Unhandled observation type: {type(obs)}")

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
