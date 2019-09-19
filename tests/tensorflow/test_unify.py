import pytest

import tensorflow as tf

from unification import unify, reify, var

from symbolic_pymc.tensorflow.meta import (TFlowOpName, mt, TFlowMetaTensorShape)
from symbolic_pymc.etuple import (ExpressionTuple, etuple, etuplize)

from tests.tensorflow import run_in_graph_mode
from tests.tensorflow.utils import assert_ops_equal


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_etuple_term():
    a = tf.compat.v1.placeholder(tf.float64, name='a')
    b = tf.compat.v1.placeholder(tf.float64, name='b')

    a_mt = mt(a)
    a_mt._obj = None
    a_reified = a_mt.reify()
    assert isinstance(a_reified, tf.Tensor)
    assert a_reified.shape.dims is None

    test_e = etuplize(a_mt, shallow=False)
    assert test_e[0] == mt.placeholder
    assert test_e[1] == tf.float64
    assert isinstance(test_e[2], ExpressionTuple)
    assert test_e[2][0].base == tf.TensorShape
    assert test_e[2][1] is None

    test_e = etuplize(a_mt, shallow=True)
    assert test_e[0] == mt.placeholder
    assert test_e[1] == tf.float64
    assert isinstance(test_e[2], TFlowMetaTensorShape)
    assert test_e[2] is a_mt.op.node_def.attr['shape']

    test_e._eval_obj = ExpressionTuple.null
    a_evaled = test_e.eval_obj
    assert all([a == b for a, b in zip(a_evaled.rands(), a_mt.rands())])

    a_reified = a_evaled.reify()
    assert isinstance(a_reified, tf.Tensor)
    assert a_reified.shape.dims is None
    assert TFlowOpName(a_reified.name) == TFlowOpName(a.name)

    e2 = mt.add(a, b)
    e2_et = etuplize(e2)
    assert isinstance(e2_et, ExpressionTuple)
    assert e2_et[0] == mt.add


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_basic_unify_reify():
    # Test reification with manually constructed replacements
    a = tf.compat.v1.placeholder(tf.float64, name='a')
    x_l = var('x_l')
    a_reif = reify(x_l, {x_l: mt(a)})
    assert a_reif.obj is not None
    # Confirm that identity is preserved (i.e. that the underlying object
    # was properly tracked and not unnecessarily reconstructed)
    assert a == a_reif.reify()

    test_expr = mt.add(tf.constant(1, dtype=tf.float64),
                       mt.mul(tf.constant(2, dtype=tf.float64),
                              x_l))
    test_reify_res = reify(test_expr, {x_l: a})
    test_base_res = test_reify_res.reify()
    assert isinstance(test_base_res, tf.Tensor)

    expected_res = tf.add(tf.constant(1, dtype=tf.float64),
                          tf.constant(2, dtype=tf.float64) * a)
    assert_ops_equal(test_base_res, expected_res)

    # Simply make sure that unification succeeds
    meta_expected_res = mt(expected_res)
    s_test = unify(test_expr, meta_expected_res, {})
    assert len(s_test) == 5

    assert reify(test_expr, s_test) == meta_expected_res


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_sexp_unify_reify():
    """Make sure we can unify and reify etuples/S-exps."""
    # Unify `A . (x + y)`, for `x`, `y` logic variables
    A = tf.compat.v1.placeholder(tf.float64, name='A',
                                 shape=tf.TensorShape([None, None]))
    x = tf.compat.v1.placeholder(tf.float64, name='x',
                                 shape=tf.TensorShape([None, 1]))
    y = tf.compat.v1.placeholder(tf.float64, name='y',
                                 shape=tf.TensorShape([None, 1]))

    z = tf.matmul(A, tf.add(x, y))

    z_sexp = etuplize(z)

    # Let's just be sure that the original TF objects are preserved
    assert z_sexp[1].eval_obj.reify() == A
    assert z_sexp[2][1].eval_obj.reify() == x
    assert z_sexp[2][2].eval_obj.reify() == y

    dis_pat = etuple(mt.matmul, var('A'),
                     etuple(mt.add, var('b'), var('c'), var()),
                     # Some op parameters we can ignore...
                     var(), var(), var())

    s = unify(dis_pat, z_sexp, {})

    assert s[var('A')] == z_sexp[1]
    assert s[var('b')] == z_sexp[2][1]
    assert s[var('c')] == z_sexp[2][2]

    # Now, we construct a graph that reflects the distributive property and
    # reify with the substitutions from the un-distributed form
    out_pat = etuple(mt.add,
                     etuple(mt.matmul, var('A'), var('b')),
                     etuple(mt.matmul, var('A'), var('c')))
    z_dist = reify(out_pat, s)

    # Evaluate the tuple-expression and get a meta object/graph
    z_dist_mt = z_dist.eval_obj

    # If all the logic variables were reified, we should be able to
    # further reify the meta graph and get a concrete TF graph
    z_dist_tf = z_dist_mt.reify()

    # Check the first part of `A . x + A . y` (i.e. `A . x`)
    assert z_dist_tf.op.inputs[0].op.inputs[0] == A
    assert z_dist_tf.op.inputs[0].op.inputs[1] == x
    # Now, the second, `A . y`
    assert z_dist_tf.op.inputs[1].op.inputs[0] == A
    assert z_dist_tf.op.inputs[1].op.inputs[1] == y
