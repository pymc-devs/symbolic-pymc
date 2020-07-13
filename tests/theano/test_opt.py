import numpy as np
import theano
import theano.tensor as tt

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
)
from symbolic_pymc.theano.utils import optimize_graph
from symbolic_pymc.theano.random_variables import CategoricalRV, DirichletRV, NormalRV


def test_kanren_opt():
    """Make sure we can run miniKanren "optimizations" over a graph until a fixed-point/normal-form is reached.
    """
    tt.config.cxx = ""
    tt.config.compute_test_value = "ignore"

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


def test_push_out_rvs():
    theano.config.cxx = ""
    theano.config.mode = "FAST_COMPILE"
    tt.config.compute_test_value = "warn"

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
