import pytest
import theano.tensor as tt

from unification import var

from kanren import eq
from kanren.core import lall

from etuples import etuple, etuplize

from theano.gof.opt import EquilibriumOptimizer
from theano.gof.graph import inputs as tt_inputs

from symbolic_pymc.theano.meta import mt
from symbolic_pymc.theano.opt import KanrenRelationSub, FunctionGraph
from symbolic_pymc.theano.utils import optimize_graph


@pytest.mark.usefixtures("run_with_theano")
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
