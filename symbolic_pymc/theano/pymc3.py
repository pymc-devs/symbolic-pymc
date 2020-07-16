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

from theano.gof.op import get_test_value
from theano.gof.graph import Apply, inputs as tt_inputs
from theano.scan_module.scan_op import Scan
from theano.scan_module.scan_utils import clone

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
    NegBinomialRV,
    NegBinomialRVType,
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


@dispatch(tt.TensorVariable, object)
def logp(var, obs):

    node = var.owner

    if hasattr(node, "fgraph") and hasattr(node.fgraph, "shape_feature"):
        shape = list(node.fgraph.shape_feature.shape_tuple(var))
    else:
        shape = list(var.shape)

    for i, s in enumerate(shape):
        try:
            shape[i] = tt.get_scalar_constant_value(s)
        except tt.NotScalarConstantError:
            shape[i] = s.tag.test_value

    logp_fn = _logp_fn(node.op, node, shape)
    return logp_fn(obs)


@dispatch(RandomVariable, Apply, object)
def _logp_fn(op, node, shape=None):
    dist_type, dist_params = _convert_rv_to_dist(op, node)
    if shape is not None:
        dist_params["shape"] = shape
    res = dist_type.dist(**dist_params)
    # Add extra information to the PyMC3 `Distribution` object
    res.dist_params = dist_params
    res.ndim_supp = op.ndim_supp
    # TODO: Need to maintain the order of these so that they correspond with
    # the `Distribution`'s parameters
    res.ndims_params = op.ndims_params
    return res.logp


@_logp_fn.register(Scan, Apply, object)
def _logp_fn_Scan(op, scan_node, shape=None):

    scan_inner_inputs = scan_node.op.inputs
    scan_inner_outputs = scan_node.op.outputs

    def create_obs_var(i, x):
        obs = x.type()
        obs.name = f"{x.name or x.owner.op.name}_obs_{i}"
        if hasattr(x.tag, "test_value"):
            obs.tag.test_value = x.tag.test_value
        return obs

    rv_outs = [
        (i, x, create_obs_var(i, x))
        for i, x in enumerate(scan_inner_outputs)
        if x.owner and isinstance(x.owner.op, RandomVariable)
    ]
    rv_inner_out_idx, rv_out_vars, rv_out_obs = zip(*rv_outs)
    # rv_outer_out_idx = [scan_node.op.get_oinp_iinp_iout_oout_mappings()['outer_out_from_inner_out'][i] for i in rv_inner_out_idx]
    # rv_outer_outputs = [scan_node.outputs[i] for i in rv_outer_out_idx]

    logp_inner_outputs = [clone(logp(rv, obs)) for i, rv, obs in rv_outs]
    assert all(o in tt.gof.graph.inputs(logp_inner_outputs) for o in rv_out_obs)

    logp_inner_outputs_inputs = tt.gof.graph.inputs(logp_inner_outputs)
    rv_relevant_inner_input_idx, rv_relevant_inner_inputs = zip(
        *[(n, i) for n, i in enumerate(scan_inner_inputs) if i in logp_inner_outputs_inputs]
    )
    logp_inner_inputs = list(rv_out_obs) + list(rv_relevant_inner_inputs)

    # We need to create outer-inputs that represent arrays of observations
    # for each random variable.
    # To do that, we're going to use each random variable's outer-output term,
    # since they necessarily have the same shape and type as the observations
    # arrays.

    # Just like we did for the inner-inputs, we need to get only the outer-inputs
    # that are relevant to the new logp graphs.
    # We can do that by removing the irrelevant outer-inputs using the known relevant inner-inputs
    removed_inner_inputs = set(range(len(scan_inner_inputs))) - set(rv_relevant_inner_input_idx)
    old_in_out_mappings = scan_node.op.get_oinp_iinp_iout_oout_mappings()
    rv_removed_outer_input_idx = [
        old_in_out_mappings["outer_inp_from_inner_inp"][i] for i in removed_inner_inputs
    ]
    rv_removed_outer_inputs = [scan_node.inputs[i] for i in rv_removed_outer_input_idx]

    rv_relevant_outer_inputs = [r for r in scan_node.inputs if r not in rv_removed_outer_inputs]

    # Now, we can create a new op with our new inner-graph inputs and outputs.
    # Also, since our inner graph has new placeholder terms representing
    # an observed value for each random variable, we need to update the
    # "info" `dict`.
    logp_info = scan_node.op.info.copy()
    logp_info["tap_array"] = []
    logp_info["n_seqs"] += len(rv_out_obs)
    logp_info["n_mit_mot"] = 0
    logp_info["n_mit_mot_outs"] = 0
    logp_info["mit_mot_out_slices"] = []
    logp_info["n_mit_sot"] = 0
    logp_info["n_sit_sot"] = 0
    logp_info["n_shared_outs"] = 0
    logp_info["n_nit_sot"] += len(rv_out_obs) - 1
    logp_info["name"] = None
    logp_info["strict"] = True

    # These are the tensor variables corresponding to each random variable's
    # array of observations.
    def logp_fn(*obs):
        logp_obs_outer_inputs = list(obs)  # [r.clone() for r in rv_outer_outputs]
        logp_outer_inputs = (
            [rv_relevant_outer_inputs[0]] + logp_obs_outer_inputs + rv_relevant_outer_inputs[1:]
        )
        logp_op = Scan(logp_inner_inputs, logp_inner_outputs, logp_info)
        scan_logp = logp_op(*logp_outer_inputs)
        return scan_logp

    # logp_fn = OpFromGraph(logp_obs_outer_inputs, [scan_logp])
    return logp_fn


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
    res = NormalRV(dist.mu, dist.sigma, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(NormalRVType, Apply)
def _convert_rv_to_dist_Normal(op, rv):
    params = {"mu": rv.inputs[0], "sigma": rv.inputs[1]}
    return pm.Normal, params


@convert_dist_to_rv.register(pm.HalfNormal, object)
def convert_dist_to_rv_HalfNormal(dist, rng):
    size = dist.shape.astype(int)[HalfNormalRV.ndim_supp :]
    res = HalfNormalRV(np.array(0.0, dtype=dist.dtype), dist.sigma, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(HalfNormalRVType, Apply)
def _convert_rv_to_dist_HalfNormal(op, rv):
    assert not np.any(tt_get_values(rv.inputs[0]))
    params = {"sigma": rv.inputs[1]}
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
    res = GammaRV(dist.alpha, dist.beta, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(GammaRVType, Apply)
def _convert_rv_to_dist_Gamma(op, rv):
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[1]}
    return pm.Gamma, params


@convert_dist_to_rv.register(pm.InverseGamma, object)
def convert_dist_to_rv_InverseGamma(dist, rng):
    size = dist.shape.astype(int)[InvGammaRV.ndim_supp :]
    res = InvGammaRV(dist.alpha, rate=dist.beta, size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(InvGammaRVType, Apply)
def _convert_rv_to_dist_InvGamma(op, rv):
    params = {"alpha": rv.inputs[0], "beta": rv.inputs[1]}
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


@convert_dist_to_rv.register(pm.NegativeBinomial, object)
def convert_dist_to_rv_NegBinomial(dist, rng):
    size = dist.shape.astype(int)[BinomialRV.ndim_supp :]
    res = NegBinomialRV(dist.alpha, dist.mu / (dist.mu + dist.alpha), size=size, rng=rng)
    return res


@_convert_rv_to_dist.register(NegBinomialRVType, Apply)
def _convert_rv_to_dist_NegBinomial(op, rv):
    params = {"alpha": rv.inputs[0], "mu": rv.inputs[1] * rv.inputs[0] / (1.0 - rv.inputs[1])}
    return pm.NegativeBinomial, params


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
    a_tt = rv.inputs[0]
    # TODO FIXME: This is a work-around; remove when/if
    # https://github.com/pymc-devs/pymc3/pull/4000 is merged.
    a_pm = get_test_value(a_tt)
    params = {"a": a_pm}
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


def graph_model(graph, *model_args, generate_names=False, **model_kwargs):
    """Create a PyMC3 model from a Theano graph with `RandomVariable` nodes."""
    model = pm.Model(*model_args, **model_kwargs)

    fgraph = graph
    if not isinstance(fgraph, FunctionGraph):
        fgraph = FunctionGraph(tt.gof.graph.inputs([fgraph]), [fgraph])

    nodes = [n for n in fgraph.toposort() if isinstance(n.op, RandomVariable)]
    rv_replacements = {}

    node_id = 0

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

        if generate_names and rv_var.name is None:
            node_name = "{}_{}".format(node.op.name, node_id)
            # warn("Name {} generated for node {}.".format(node, node_name))
            node_id += 1
            rv_var.name = node_name

        with model:
            rv = convert_rv_to_dist(node, obs)

        rv_replacements[old_rv_var] = rv

    model.rv_replacements = rv_replacements

    return model
