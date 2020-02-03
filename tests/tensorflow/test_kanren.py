from unification import var, unify

from kanren import run, eq, lall
from kanren.graph import walko
from kanren.assoccomm import eq_comm, commutative

from symbolic_pymc.meta import enable_lvar_defaults
from symbolic_pymc.tensorflow.meta import mt

from tests.tensorflow import run_in_graph_mode


@run_in_graph_mode
def test_walko():
    with enable_lvar_defaults("names"):
        add_1_mt = mt(1) + mt(2)

    def walk_rel(x, y):
        return lall(eq(x, mt(1)), eq(y, mt(3)))

    q = var()
    (res,) = run(1, q, walko(walk_rel, add_1_mt, q))

    # The easiest way to check whether or not two arbitrary TF meta graphs are
    # (structurally) equivalent is to confirm that they unify.  This avoids
    # uninteresting differences in node names, uninferred type information,
    # etc.
    with enable_lvar_defaults("names", "node_attrs"):
        assert unify(res.eval_obj, mt(3) + mt(2)) is not False


@run_in_graph_mode
def test_commutativity():
    with enable_lvar_defaults("names"):
        add_1_mt = mt(1) + mt(2)
        add_2_mt = mt(2) + mt(1)

    q = var()
    res = run(0, q, commutative(add_1_mt.base_operator))
    assert res is not False

    res = run(0, q, eq_comm(add_1_mt, add_2_mt))
    assert res is not False

    with enable_lvar_defaults("names"):
        add_pattern_mt = mt(2) + q

    res = run(0, q, eq_comm(add_1_mt, add_pattern_mt))
    assert res[0] == add_1_mt.base_arguments[0]
