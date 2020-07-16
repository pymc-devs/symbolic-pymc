import pytest
import numpy as np
import theano
import theano.tensor as tt

from copy import copy
from unification import var

from kanren import eq
from kanren.core import lall

from etuples import etuple, etuplize

from theano.gof.opt import EquilibriumOptimizer
from theano.gof.graph import inputs as tt_inputs
from theano.scan_module.scan_op import Scan

from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.opt import (
    KanrenRelationSub,
    FunctionGraph,
    push_out_rvs_from_scan,
    ScanArgs,
    convert_outer_out_to_in,
)
from symbolic_pymc.theano.ops import RandomVariable
from symbolic_pymc.theano.utils import optimize_graph
from symbolic_pymc.theano.random_variables import CategoricalRV, DirichletRV, NormalRV


@theano.change_flags(compute_test_value="ignore", cxx="", mode="FAST_COMPILE")
def test_kanren_opt():
    """Make sure we can run miniKanren "optimizations" over a graph until a fixed-point/normal-form is reached.
    """
    x_tt = tt.vector("x")
    c_tt = tt.vector("c")
    d_tt = tt.vector("c")
    A_tt = tt.matrix("A")
    B_tt = tt.matrix("B")

    Z_tt = A_tt.dot(x_tt + B_tt.dot(c_tt + d_tt))

    fgraph = FunctionGraph(tt_inputs([Z_tt]), [Z_tt], clone=True)

    assert isinstance(fgraph.outputs[0].owner.op, tt.Dot)

    def distributes(in_lv, out_lv):
        return lall(
            # lhs == A * (x + b)
            eq(etuple(mt.dot, var("A"), etuple(mt.add, var("x"), var("b"))), etuplize(in_lv)),
            # rhs == A * x + A * b
            eq(
                etuple(
                    mt.add, etuple(mt.dot, var("A"), var("x")), etuple(mt.dot, var("A"), var("b"))
                ),
                out_lv,
            ),
        )

    distribute_opt = EquilibriumOptimizer([KanrenRelationSub(distributes)], max_use_ratio=10)

    fgraph_opt = optimize_graph(fgraph, distribute_opt, return_graph=False)

    assert fgraph_opt.owner.op == tt.add
    assert isinstance(fgraph_opt.owner.inputs[0].owner.op, tt.Dot)
    # TODO: Something wrong with `etuple` caching?
    # assert fgraph_opt.owner.inputs[0].owner.inputs[0] == A_tt
    assert fgraph_opt.owner.inputs[0].owner.inputs[0].name == "A"
    assert fgraph_opt.owner.inputs[1].owner.op == tt.add
    assert isinstance(fgraph_opt.owner.inputs[1].owner.inputs[0].owner.op, tt.Dot)
    assert isinstance(fgraph_opt.owner.inputs[1].owner.inputs[1].owner.op, tt.Dot)


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_push_out_rvs():

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
    rng_tt = theano.shared(rng_state, name="rng", borrow=True)
    rng_tt.tag.is_rng = True
    rng_tt.default_update = rng_tt

    N_tt = tt.iscalar("N")
    N_tt.tag.test_value = 10
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 2

    mus_tt = tt.matrix("mus_t")
    mus_tt.tag.test_value = np.stack([np.arange(0.0, 10), np.arange(0.0, -10, -1)], axis=-1).astype(
        theano.config.floatX
    )

    sigmas_tt = tt.ones((N_tt,))
    Gamma_rv = DirichletRV(tt.ones((M_tt, M_tt)), rng=rng_tt, name="Gamma")

    # The optimizer should do nothing to this term, because it's not a `Scan`
    fgraph = FunctionGraph(tt_inputs([Gamma_rv]), [Gamma_rv])
    pushoutrvs_opt = EquilibriumOptimizer([push_out_rvs_from_scan], max_use_ratio=10)
    Gamma_opt_rv = optimize_graph(fgraph, pushoutrvs_opt, return_graph=False)
    # The `FunctionGraph` will, however, clone the graph objects, so we can't
    # simply check that `gamma_opt_rv == Gamma_rv`
    assert all(type(a) == type(b) for a, b in zip(tt_inputs([Gamma_rv]), tt_inputs([Gamma_opt_rv])))
    assert theano.scan_module.scan_utils.equal_computations(
        [Gamma_opt_rv], [Gamma_rv], tt_inputs([Gamma_opt_rv]), tt_inputs([Gamma_rv])
    )

    # In this case, `Y_t` depends on `S_t` and `S_t` is not output.  Our
    # push-out optimization should create a new `Scan` that also outputs each
    # `S_t`.
    def scan_fn(mus_t, sigma_t, Gamma_t, rng):
        S_t = CategoricalRV(Gamma_t[0], rng=rng, name="S_t")
        Y_t = NormalRV(mus_t[S_t], sigma_t, rng=rng, name="Y_t")
        return Y_t

    Y_rv, _ = theano.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv, rng_tt],
        outputs_info=[{}],
        strict=True,
        name="scan_rv",
    )
    Y_rv.name = "Y_rv"

    orig_scan_op = Y_rv.owner.op
    assert len(Y_rv.owner.outputs) == 2
    assert isinstance(orig_scan_op, Scan)
    assert len(orig_scan_op.outputs) == 2
    assert orig_scan_op.outputs[0].owner.op == NormalRV
    assert isinstance(orig_scan_op.outputs[1].type, tt.raw_random.RandomStateType)

    fgraph = FunctionGraph(tt_inputs([Y_rv]), [Y_rv], clone=True)
    fgraph_opt = optimize_graph(fgraph, pushoutrvs_opt, return_graph=True)

    # There should now be a new output for all the `S_t`
    new_scan = fgraph_opt.outputs[0].owner
    assert len(new_scan.outputs) == 3
    assert isinstance(new_scan.op, Scan)
    assert new_scan.op.outputs[0].owner.op == NormalRV
    assert new_scan.op.outputs[1].owner.op == CategoricalRV
    assert isinstance(new_scan.op.outputs[2].type, tt.raw_random.RandomStateType)


def create_test_hmm():
    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
    rng_init_state = rng_state.get_state()
    rng_tt = theano.shared(rng_state, name="rng", borrow=True)
    rng_tt.tag.is_rng = True
    rng_tt.default_update = rng_tt

    N_tt = tt.iscalar("N")
    N_tt.tag.test_value = 10
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 2

    mus_tt = tt.matrix("mus")
    mus_tt.tag.test_value = np.stack([np.arange(0.0, 10), np.arange(0.0, -10, -1)], axis=-1).astype(
        theano.config.floatX
    )

    sigmas_tt = tt.ones((N_tt,))
    sigmas_tt.name = "sigmas"

    pi_0_rv = DirichletRV(tt.ones((M_tt,)), rng=rng_tt, name="pi_0")
    Gamma_rv = DirichletRV(tt.ones((M_tt, M_tt)), rng=rng_tt, name="Gamma")

    S_0_rv = CategoricalRV(pi_0_rv, rng=rng_tt, name="S_0")

    def scan_fn(mus_t, sigma_t, S_tm1, Gamma_t, rng):
        S_t = CategoricalRV(Gamma_t[S_tm1], rng=rng, name="S_t")
        Y_t = NormalRV(mus_t[S_t], sigma_t, rng=rng, name="Y_t")
        return S_t, Y_t

    (S_rv, Y_rv), scan_updates = theano.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv, rng_tt],
        outputs_info=[{"initial": S_0_rv, "taps": [-1]}, {}],
        strict=True,
        name="scan_rv",
    )
    Y_rv.name = "Y_rv"

    scan_op = Y_rv.owner.op
    scan_args = ScanArgs.from_node(Y_rv.owner)

    Gamma_in = scan_args.inner_in_non_seqs[0]
    Y_t = scan_args.inner_out_nit_sot[0]
    mus_t = scan_args.inner_in_seqs[0]
    sigmas_t = scan_args.inner_in_seqs[1]
    S_t = scan_args.inner_out_sit_sot[0]
    rng_in = scan_args.inner_out_shared[0]

    rng_updates = scan_updates[rng_tt]
    rng_updates.name = "rng_updates"
    mus_in = Y_rv.owner.inputs[1]
    mus_in.name = "mus_in"
    sigmas_in = Y_rv.owner.inputs[2]
    sigmas_in.name = "sigmas_in"

    # The output `S_rv` is really `S_rv[1:]`, so we have to extract the actual
    # `Scan` output: `S_rv`.
    S_rv = S_rv.owner.inputs[0]
    S_rv.name = "S_in"

    return locals()


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_basics():

    # Make sure we can create an empty `ScanArgs`
    scan_args = ScanArgs.create_empty()
    assert scan_args.n_steps is None
    for name in scan_args.field_names:
        if name == "n_steps":
            continue
        assert len(getattr(scan_args, name)) == 0

    with pytest.raises(TypeError):
        ScanArgs.from_node(tt.ones(2).owner)

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]

    # Make sure we can get alternate variables
    test_v = scan_args.outer_out_sit_sot[0]
    alt_test_v = scan_args.get_alt_field(test_v, "inner_out")
    assert alt_test_v == scan_args.inner_out_sit_sot[0]

    # Check the `__repr__` and `__str__`
    scan_args_repr = repr(scan_args)
    # Just make sure it doesn't err-out
    assert scan_args_repr.startswith("ScanArgs")

    # Check the properties that allow us to use
    # `Scan.get_oinp_iinp_iout_oout_mappings` as-is to implement
    # `ScanArgs.var_mappings`
    assert scan_args.n_nit_sot == scan_op.n_nit_sot
    assert scan_args.n_mit_mot == scan_op.n_mit_mot
    # The `scan_args` base class always clones the inner-graph;
    # here we make sure it doesn't (and that all the inputs are the same)
    assert scan_args.inputs == scan_op.inputs
    scan_op_info = dict(scan_op.info)
    # The `ScanInfo` dictionary has the wrong order and an extra entry
    del scan_op_info["strict"]
    assert dict(scan_args.info) == scan_op_info
    assert scan_args.var_mappings == scan_op.var_mappings

    # Check that `ScanArgs.find_among_fields` works
    test_v = scan_op.inner_seqs(scan_op.inputs)[1]
    field_info = scan_args.find_among_fields(test_v)
    assert field_info.name == "inner_in_seqs"
    assert field_info.index == 1
    assert field_info.inner_index is None
    assert scan_args.inner_inputs[field_info.agg_index] == test_v

    test_l = scan_op.inner_non_seqs(scan_op.inputs)
    # We didn't index this argument, so it's a `list` (i.e. bad input)
    field_info = scan_args.find_among_fields(test_l)
    assert field_info is None

    test_v = test_l[0]
    field_info = scan_args.find_among_fields(test_v)
    assert field_info.name == "inner_in_non_seqs"
    assert field_info.index == 0
    assert field_info.inner_index is None
    assert scan_args.inner_inputs[field_info.agg_index] == test_v

    scan_args_copy = copy(scan_args)
    assert scan_args_copy is not scan_args
    assert scan_args_copy == scan_args

    assert scan_args_copy != test_v
    scan_args_copy.outer_in_seqs.pop()
    assert scan_args_copy != scan_args


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_basics_mit_sot():

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
    rng_tt = theano.shared(rng_state, name="rng", borrow=True)
    rng_tt.tag.is_rng = True
    rng_tt.default_update = rng_tt

    N_tt = tt.iscalar("N")
    N_tt.tag.test_value = 10
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 2

    mus_tt = tt.matrix("mus")
    mus_tt.tag.test_value = np.stack([np.arange(0.0, 10), np.arange(0.0, -10, -1)], axis=-1).astype(
        theano.config.floatX
    )

    sigmas_tt = tt.ones((N_tt,))
    sigmas_tt.name = "sigmas"

    pi_0_rv = DirichletRV(tt.ones((M_tt,)), rng=rng_tt, name="pi_0")
    Gamma_rv = DirichletRV(tt.ones((M_tt, M_tt)), rng=rng_tt, name="Gamma")

    S_0_rv = CategoricalRV(pi_0_rv, rng=rng_tt, name="S_0")

    def scan_fn(mus_t, sigma_t, S_tm2, S_tm1, Gamma_t, rng):
        S_t = CategoricalRV(Gamma_t[S_tm2], rng=rng, name="S_t")
        Y_t = NormalRV(mus_t[S_tm1], sigma_t, rng=rng, name="Y_t")
        return S_t, Y_t

    (S_rv, Y_rv), scan_updates = theano.scan(
        fn=scan_fn,
        sequences=[mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv, rng_tt],
        outputs_info=[{"initial": tt.stack([S_0_rv, S_0_rv]), "taps": [-2, -1]}, {}],
        strict=True,
        name="scan_rv",
    )
    # Adding names should make output easier to read
    Y_rv.name = "Y_rv"
    # This `S_rv` outer-output is actually a `Subtensor` of the "real" output
    S_rv = S_rv.owner.inputs[0]
    S_rv.name = "S_rv"
    rng_updates = scan_updates[rng_tt]
    rng_updates.name = "rng_updates"
    mus_in = Y_rv.owner.inputs[1]
    mus_in.name = "mus_in"
    sigmas_in = Y_rv.owner.inputs[2]
    sigmas_in.name = "sigmas_in"

    scan_args = ScanArgs.from_node(Y_rv.owner)

    test_v = scan_args.inner_in_mit_sot[0][1]
    field_info = scan_args.find_among_fields(test_v)

    assert field_info.name == "inner_in_mit_sot"
    assert field_info.index == 0
    assert field_info.inner_index == 1
    assert field_info.agg_index == 3

    rm_info = scan_args._remove_from_fields(tt.ones(2))
    assert rm_info is None

    rm_info = scan_args._remove_from_fields(test_v)

    assert rm_info.name == "inner_in_mit_sot"
    assert rm_info.index == 0
    assert rm_info.inner_index == 1
    assert rm_info.agg_index == 3


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_remove_inner_input():

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_rv = hmm_model_env["S_rv"]
    S_t = hmm_model_env["S_t"]
    rng_tt = hmm_model_env["rng_tt"]
    rng_in = hmm_model_env["rng_in"]
    rng_updates = hmm_model_env["rng_updates"]

    # Check `ScanArgs.remove_from_fields` by removing `sigmas[t]` (i.e. the
    # inner-graph input)
    scan_args_copy = copy(scan_args)
    test_v = sigmas_t

    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=False)
    removed_nodes, _ = zip(*rm_info)

    assert sigmas_t in removed_nodes
    assert sigmas_t not in scan_args_copy.inner_in_seqs
    assert Y_t not in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 1

    scan_args_copy = copy(scan_args)
    test_v = sigmas_t

    # This removal includes dependents
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    # `sigmas[t]` (i.e. inner-graph input) should be gone
    assert sigmas_t in removed_nodes
    assert sigmas_t not in scan_args_copy.inner_in_seqs
    # `Y_t` (i.e. inner-graph output) should be gone
    assert Y_t in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 0
    # `Y_rv` (i.e. outer-graph output) should be gone
    assert Y_rv in removed_nodes
    assert Y_rv not in scan_args_copy.outer_outputs
    assert len(scan_args_copy.outer_out_nit_sot) == 0
    # `sigmas_in` (i.e. outer-graph input) should be gone
    assert sigmas_in in removed_nodes
    assert test_v not in scan_args_copy.inner_in_seqs

    # These shouldn't have been removed
    assert S_t in scan_args_copy.inner_out_sit_sot
    assert S_rv in scan_args_copy.outer_out_sit_sot
    assert Gamma_in in scan_args_copy.inner_in_non_seqs
    assert Gamma_rv in scan_args_copy.outer_in_non_seqs
    assert rng_tt in scan_args_copy.outer_in_shared
    assert rng_in in scan_args_copy.inner_out_shared
    assert rng_updates in scan_args.outer_out_shared

    # The other `Y_rv`-related inputs currently aren't removed, even though
    # they're no longer needed.
    # TODO: Would be nice if we did this, too
    # assert len(scan_args_copy.outer_in_seqs) == 0
    # TODO: Would be nice if we did this, too
    # assert len(scan_args_copy.inner_in_seqs) == 0

    # We shouldn't be able to remove the removed node
    with pytest.raises(ValueError):
        rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_remove_outer_input():

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_rv = hmm_model_env["S_rv"]
    S_t = hmm_model_env["S_t"]
    rng_tt = hmm_model_env["rng_tt"]
    rng_in = hmm_model_env["rng_in"]
    rng_updates = hmm_model_env["rng_updates"]

    # Remove `sigmas` (i.e. the outer-input)
    scan_args_copy = copy(scan_args)
    test_v = sigmas_in
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    # `sigmas_in` (i.e. outer-graph input) should be gone
    assert scan_args.outer_in_seqs[-1] in removed_nodes
    assert test_v not in scan_args_copy.inner_in_seqs

    # `sigmas[t]` should be gone
    assert sigmas_t in removed_nodes
    assert sigmas_t not in scan_args_copy.inner_in_seqs

    # `Y_t` (i.e. inner-graph output) should be gone
    assert Y_t in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 0

    # `Y_rv` (i.e. outer-graph output) should be gone
    assert Y_rv not in scan_args_copy.outer_outputs
    assert len(scan_args_copy.outer_out_nit_sot) == 0

    assert S_t in scan_args_copy.inner_out_sit_sot
    assert S_rv in scan_args_copy.outer_out_sit_sot
    assert Gamma_in in scan_args_copy.inner_in_non_seqs
    assert Gamma_rv in scan_args_copy.outer_in_non_seqs
    assert rng_tt in scan_args_copy.outer_in_shared
    assert rng_in in scan_args_copy.inner_out_shared
    assert rng_updates in scan_args.outer_out_shared


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_remove_inner_output():

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_rv = hmm_model_env["S_rv"]
    S_t = hmm_model_env["S_t"]
    rng_tt = hmm_model_env["rng_tt"]
    rng_in = hmm_model_env["rng_in"]
    rng_updates = hmm_model_env["rng_updates"]

    # Remove `Y_t` (i.e. the inner-output)
    scan_args_copy = copy(scan_args)
    test_v = Y_t
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    # `Y_t` (i.e. inner-graph output) should be gone
    assert Y_t in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 0

    # `Y_rv` (i.e. outer-graph output) should be gone
    assert Y_rv not in scan_args_copy.outer_outputs
    assert len(scan_args_copy.outer_out_nit_sot) == 0

    assert S_t in scan_args_copy.inner_out_sit_sot
    assert S_rv in scan_args_copy.outer_out_sit_sot
    assert Gamma_in in scan_args_copy.inner_in_non_seqs
    assert Gamma_rv in scan_args_copy.outer_in_non_seqs
    assert rng_tt in scan_args_copy.outer_in_shared
    assert rng_in in scan_args_copy.inner_out_shared
    assert rng_updates in scan_args.outer_out_shared


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_remove_outer_output():

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_rv = hmm_model_env["S_rv"]
    S_t = hmm_model_env["S_t"]
    rng_tt = hmm_model_env["rng_tt"]
    rng_in = hmm_model_env["rng_in"]
    rng_updates = hmm_model_env["rng_updates"]

    # Remove `Y_rv` (i.e. a nit-sot outer-output)
    scan_args_copy = copy(scan_args)
    test_v = Y_rv
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    # `Y_t` (i.e. inner-graph output) should be gone
    assert Y_t in removed_nodes
    assert len(scan_args_copy.inner_out_nit_sot) == 0

    # `Y_rv` (i.e. outer-graph output) should be gone
    assert Y_rv not in scan_args_copy.outer_outputs
    assert len(scan_args_copy.outer_out_nit_sot) == 0

    assert S_t in scan_args_copy.inner_out_sit_sot
    assert S_rv in scan_args_copy.outer_out_sit_sot
    assert Gamma_in in scan_args_copy.inner_in_non_seqs
    assert Gamma_rv in scan_args_copy.outer_in_non_seqs
    assert rng_tt in scan_args_copy.outer_in_shared
    assert rng_in in scan_args_copy.inner_out_shared
    assert rng_updates in scan_args.outer_out_shared


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_remove_nonseq_outer_input():

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    mus_in = hmm_model_env["mus_in"]
    mus_t = hmm_model_env["mus_t"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_rv = hmm_model_env["S_rv"]
    S_t = hmm_model_env["S_t"]
    rng_tt = hmm_model_env["rng_tt"]
    rng_in = hmm_model_env["rng_in"]
    rng_updates = hmm_model_env["rng_updates"]

    # Remove `Gamma` (i.e. a non-sequence outer-input)
    scan_args_copy = copy(scan_args)
    test_v = Gamma_rv
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    assert Gamma_rv in removed_nodes
    assert Gamma_in in removed_nodes
    assert S_rv in removed_nodes
    assert S_t in removed_nodes
    assert Y_t in removed_nodes
    assert Y_rv in removed_nodes

    assert mus_in in scan_args_copy.outer_in_seqs
    assert sigmas_in in scan_args_copy.outer_in_seqs
    assert mus_t in scan_args_copy.inner_in_seqs
    assert sigmas_t in scan_args_copy.inner_in_seqs
    assert rng_tt in scan_args_copy.outer_in_shared
    assert rng_in in scan_args_copy.inner_out_shared
    assert rng_updates in scan_args.outer_out_shared


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_remove_nonseq_inner_input():

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    mus_in = hmm_model_env["mus_in"]
    mus_t = hmm_model_env["mus_t"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_rv = hmm_model_env["S_rv"]
    S_t = hmm_model_env["S_t"]
    rng_tt = hmm_model_env["rng_tt"]
    rng_in = hmm_model_env["rng_in"]
    rng_updates = hmm_model_env["rng_updates"]

    # Remove `Gamma` (i.e. a non-sequence inner-input)
    scan_args_copy = copy(scan_args)
    test_v = Gamma_in
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    assert Gamma_in in removed_nodes
    assert Gamma_rv in removed_nodes
    assert S_rv in removed_nodes
    assert S_t in removed_nodes

    # import pdb; pdb.set_trace()

    assert mus_in in scan_args_copy.outer_in_seqs
    assert sigmas_in in scan_args_copy.outer_in_seqs
    assert mus_t in scan_args_copy.inner_in_seqs
    assert sigmas_t in scan_args_copy.inner_in_seqs
    assert rng_tt in scan_args_copy.outer_in_shared
    assert rng_in in scan_args_copy.inner_out_shared
    assert rng_updates in scan_args.outer_out_shared


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_ScanArgs_remove_shared_inner_output():

    hmm_model_env = create_test_hmm()
    scan_args = hmm_model_env["scan_args"]
    scan_op = hmm_model_env["scan_op"]
    Y_t = hmm_model_env["Y_t"]
    Y_rv = hmm_model_env["Y_rv"]
    mus_in = hmm_model_env["mus_in"]
    mus_t = hmm_model_env["mus_t"]
    sigmas_in = hmm_model_env["sigmas_in"]
    sigmas_t = hmm_model_env["sigmas_t"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    Gamma_in = hmm_model_env["Gamma_in"]
    S_rv = hmm_model_env["S_rv"]
    S_t = hmm_model_env["S_t"]
    rng_tt = hmm_model_env["rng_tt"]
    rng_in = hmm_model_env["rng_in"]
    rng_updates = hmm_model_env["rng_updates"]

    # Remove `rng` (i.e. a shared inner-output)
    scan_args_copy = copy(scan_args)
    test_v = rng_updates
    rm_info = scan_args_copy.remove_from_fields(test_v, rm_dependents=True)
    removed_nodes, _ = zip(*rm_info)

    assert rng_tt in removed_nodes
    assert rng_in in removed_nodes
    assert rng_updates in removed_nodes
    assert Y_rv in removed_nodes
    assert S_rv in removed_nodes

    assert sigmas_in in scan_args_copy.outer_in_seqs
    assert sigmas_t in scan_args_copy.inner_in_seqs
    assert mus_in in scan_args_copy.outer_in_seqs
    assert mus_t in scan_args_copy.inner_in_seqs


def get_random_outer_outputs(scan_args):
    """Get the `RandomVariable` outputs of a `Scan` (well, it's `ScanArgs`)."""
    rv_vars = []
    for n, oo in enumerate(scan_args.outer_outputs):
        oo_info = scan_args.find_among_fields(oo)
        io_type = oo_info.name[(oo_info.name.index("_", 6) + 1) :]
        inner_out_type = "inner_out_{}".format(io_type)
        io_var = getattr(scan_args, inner_out_type)[oo_info.index]
        if io_var.owner and isinstance(io_var.owner.op, RandomVariable):
            rv_vars.append((n, oo))
    return rv_vars


def create_inner_out_logp(input_scan_args, old_inner_out_var, new_inner_in_var, output_scan_args):
    """Create a log-likelihood inner-output.

    This is intended to be use with `get_random_outer_outputs`.

    """
    from symbolic_pymc.theano.pymc3 import _logp_fn

    logp_fn = _logp_fn(old_inner_out_var.owner.op, old_inner_out_var.owner, None)
    logp = logp_fn(new_inner_in_var)
    if new_inner_in_var.name:
        logp.name = "logp({})".format(new_inner_in_var.name)
    return logp


def construct_scan(scan_args):
    scan_op = Scan(scan_args.inner_inputs, scan_args.inner_outputs, scan_args.info)
    scan_out = scan_op(*scan_args.outer_inputs)

    if not isinstance(scan_out, list):
        scan_out = [scan_out]

    return scan_out


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_convert_outer_out_to_in():

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
    rng_tt = theano.shared(rng_state, name="rng", borrow=True)
    rng_tt.tag.is_rng = True
    rng_tt.default_update = rng_tt

    #
    # We create a `Scan` representing a time-series model with normally
    # distributed responses that are dependent on lagged values of both the
    # response `RandomVariable` and a lagged "deterministic" that also depends
    # on the lagged response values.
    #
    def input_step_fn(mu_tm1, y_tm1, rng):
        mu_tm1.name = "mu_tm1"
        y_tm1.name = "y_tm1"
        mu = mu_tm1 + y_tm1 + 1
        mu.name = "mu_t"
        return mu, NormalRV(mu, 1.0, rng=rng, name="Y_t")

    (mu_tt, Y_rv), _ = theano.scan(
        fn=input_step_fn,
        outputs_info=[
            {"initial": tt.as_tensor_variable(np.r_[0.0]), "taps": [-1]},
            {"initial": tt.as_tensor_variable(np.r_[0.0]), "taps": [-1]},
        ],
        non_sequences=[rng_tt],
        n_steps=10,
    )

    mu_tt.name = "mu_tt"
    mu_tt.owner.inputs[0].name = "mu_all"
    Y_rv.name = "Y_rv"
    Y_rv.owner.inputs[0].name = "Y_all"

    input_scan_args = ScanArgs.from_node(Y_rv.owner.inputs[0].owner)

    #
    # Sample from the model and create another `Scan` that computes the
    # log-likelihood of the model at the sampled point.
    #
    Y_obs = tt.as_tensor_variable(Y_rv.eval())
    Y_obs.name = "Y_obs"

    def output_step_fn(y_t, y_tm1, mu_tm1):
        import pymc3 as pm

        mu_tm1.name = "mu_tm1"
        y_tm1.name = "y_tm1"
        mu = mu_tm1 + y_tm1 + 1
        mu.name = "mu_t"
        logp = pm.Normal.dist(mu, 1.0).logp(y_t)
        logp.name = "logp"
        return mu, logp

    (mu_tt, Y_logp), _ = theano.scan(
        fn=output_step_fn,
        sequences=[{"input": Y_obs, "taps": [0, -1]}],
        outputs_info=[{"initial": tt.as_tensor_variable(np.r_[0.0]), "taps": [-1]}, {}],
    )

    Y_logp.name = "Y_logp"
    mu_tt.name = "mu_tt"

    # output_scan_args = ScanArgs.from_node(Y_logp.owner)

    #
    # Get the model output variable that corresponds to the response
    # `RandomVariable`
    #
    var_idx, var = get_random_outer_outputs(input_scan_args)[0]

    #
    # Convert the original model `Scan` into another `Scan` that's equivalent
    # to the log-likelihood `Scan` given above.
    # In other words, automatically construct the log-likelihood `Scan` based
    # on the model `Scan`.
    #
    test_scan_args = convert_outer_out_to_in(
        input_scan_args, var, inner_out_fn=create_inner_out_logp, output_scan_args=input_scan_args
    )

    scan_out = construct_scan(test_scan_args)

    #
    # Evaluate the manually and automatically constructed log-likelihoods and
    # compare.
    #
    res = scan_out[var_idx].eval({var: Y_obs.value})
    exp_res = Y_logp.eval()

    assert np.array_equal(res, exp_res)


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_convert_outer_out_to_in_mit_sot():

    rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
    rng_tt = theano.shared(rng_state, name="rng", borrow=True)
    rng_tt.tag.is_rng = True
    rng_tt.default_update = rng_tt

    #
    # This is a very simple model with only one output, but multiple
    # taps/lags.
    #
    def input_step_fn(y_tm1, y_tm2, rng):
        y_tm1.name = "y_tm1"
        y_tm2.name = "y_tm2"
        return NormalRV(y_tm1 + y_tm2, 1.0, rng=rng, name="Y_t")

    Y_rv, _ = theano.scan(
        fn=input_step_fn,
        outputs_info=[{"initial": tt.as_tensor_variable(np.r_[-1.0, 0.0]), "taps": [-1, -2]},],
        non_sequences=[rng_tt],
        n_steps=10,
    )

    Y_rv.name = "Y_rv"
    Y_rv.owner.inputs[0].name = "Y_all"

    Y_obs = tt.as_tensor_variable(Y_rv.eval())
    Y_obs.name = "Y_obs"

    input_scan_args = ScanArgs.from_node(Y_rv.owner.inputs[0].owner)

    #
    # The corresponding log-likelihood
    #
    def output_step_fn(y_t, y_tm1, y_tm2):
        import pymc3 as pm

        y_t.name = "y_t"
        y_tm1.name = "y_tm1"
        y_tm2.name = "y_tm2"
        logp = pm.Normal.dist(y_tm1 + y_tm2, 1.0).logp(y_t)
        logp.name = "logp(y_t)"
        return logp

    Y_logp, _ = theano.scan(
        fn=output_step_fn, sequences=[{"input": Y_obs, "taps": [0, -1, -2]}], outputs_info=[{}]
    )

    # output_scan_args = ScanArgs.from_node(Y_logp.owner)

    #
    # Get the model output variable that corresponds to the response
    # `RandomVariable`
    #
    var_idx, var = get_random_outer_outputs(input_scan_args)[0]

    #
    # Convert the original model `Scan` into another `Scan` that's equivalent
    # to the log-likelihood `Scan` given above.
    # In other words, automatically construct the log-likelihood `Scan` based
    # on the model `Scan`.
    #
    # In this case, we perform the conversion on a "blank" `ScanArgs`.
    #
    test_scan_args = convert_outer_out_to_in(
        input_scan_args, var, inner_out_fn=create_inner_out_logp, output_scan_args=None
    )

    scan_out = construct_scan(test_scan_args)

    #
    # Evaluate the manually and automatically constructed log-likelihoods and
    # compare.
    #
    res = scan_out[var_idx].eval({var: Y_obs.value})
    exp_res = Y_logp.eval()

    assert np.array_equal(res, exp_res)

    #
    # Now, we rerun the test, but use the "replace" features of
    # `convert_outer_out_to_in`
    #
    test_scan_args = convert_outer_out_to_in(
        input_scan_args, var, inner_out_fn=create_inner_out_logp, output_scan_args=input_scan_args
    )

    scan_out = construct_scan(test_scan_args)

    #
    # Evaluate the manually and automatically constructed log-likelihoods and
    # compare.
    #
    res = scan_out[var_idx].eval({var: Y_obs.value})
    exp_res = Y_logp.eval()

    assert np.array_equal(res, exp_res)


@theano.change_flags(compute_test_value="warn", cxx="", mode="FAST_COMPILE")
def test_convert_outer_out_to_in_hmm():
    hmm_model_env = create_test_hmm()
    input_scan_args = hmm_model_env["scan_args"]
    M_tt = hmm_model_env["M_tt"]
    N_tt = hmm_model_env["N_tt"]
    mus_tt = hmm_model_env["mus_tt"]
    sigmas_tt = hmm_model_env["sigmas_tt"]
    Y_rv = hmm_model_env["Y_rv"]
    S_0_rv = hmm_model_env["S_0_rv"]
    Gamma_rv = hmm_model_env["Gamma_rv"]
    rng_tt = hmm_model_env["rng_tt"]
    rng_init_state = hmm_model_env["rng_init_state"]

    test_point = {
        M_tt: 2,
        N_tt: 10,
        mus_tt: mus_tt.tag.test_value,
    }
    Y_obs = tt.as_tensor_variable(Y_rv.eval(test_point))
    Y_obs.name = "Y_obs"

    def logp_scan_fn(y_t, mus_t, sigma_t, S_tm1, Gamma_t, rng):
        import pymc3 as pm

        gamma_t = Gamma_t[S_tm1]
        gamma_t.name = "gamma_t"
        S_t = CategoricalRV(gamma_t, rng=rng, name="S_t")
        mu_t = mus_t[S_t]
        mu_t.name = "mu[S_t]"
        Y_logp_t = pm.Normal.dist(mu_t, sigma_t).logp(y_t)
        Y_logp_t.name = "logp(y_t)"
        return S_t, Y_logp_t

    (S_rv, Y_logp), scan_updates = theano.scan(
        fn=logp_scan_fn,
        sequences=[Y_obs, mus_tt, sigmas_tt],
        non_sequences=[Gamma_rv, rng_tt],
        outputs_info=[{"initial": S_0_rv, "taps": [-1]}, {}],
        strict=True,
        name="scan_rv",
    )
    Y_logp.name = "Y_logp"

    var_idx, var = get_random_outer_outputs(input_scan_args)[1]

    test_scan_args = convert_outer_out_to_in(
        input_scan_args, var, inner_out_fn=create_inner_out_logp, output_scan_args=input_scan_args
    )

    scan_out = construct_scan(test_scan_args)
    test_Y_logp = scan_out[var_idx]

    #
    # Evaluate the manually and automatically constructed log-likelihoods and
    # compare.
    #
    new_test_point = dict(test_point)
    new_test_point[var] = Y_obs.value

    # We need to reset the RNG each time, because `S_t` is still a
    # `RandomVariable`
    rng_tt.get_value(borrow=True).set_state(rng_init_state)
    res = test_Y_logp.eval(new_test_point)

    rng_tt.get_value(borrow=True).set_state(rng_init_state)
    exp_res = Y_logp.eval(test_point)

    assert np.array_equal(res, exp_res)
