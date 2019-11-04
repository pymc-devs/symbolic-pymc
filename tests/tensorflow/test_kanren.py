from unification import var

from kanren import run
from kanren.assoccomm import eq_comm, commutative

from symbolic_pymc.meta import enable_lvar_defaults
from symbolic_pymc.tensorflow.meta import mt

from tests.tensorflow import run_in_graph_mode


@run_in_graph_mode
def test_commutativity():
    with enable_lvar_defaults('names'):
        add_1_mt = mt(1) + mt(2)
        add_2_mt = mt(2) + mt(1)

    res = run(0, var('q'), commutative(add_1_mt.base_operator))
    assert res is not False

    res = run(0, var('q'), eq_comm(add_1_mt, add_2_mt))
    assert res is not False

    with enable_lvar_defaults('names'):
        add_pattern_mt = mt(2) + var('q')

    res = run(0, var('q'), eq_comm(add_1_mt, add_pattern_mt))
    assert res[0] == add_1_mt.base_arguments[0]
