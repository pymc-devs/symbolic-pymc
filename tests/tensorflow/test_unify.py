import pytest

import tensorflow as tf

from unification import unify, reify, var

from kanren.term import term, operator, arguments

from etuples import etuple, etuplize
from etuples.core import ExpressionTuple

from cons.core import ConsError

from symbolic_pymc.tensorflow.meta import TFlowMetaOperator, TFlowMetaTensor, mt

from tests.tensorflow import run_in_graph_mode
from tests.tensorflow.utils import assert_ops_equal


@run_in_graph_mode
def test_operator():
    s = unify(TFlowMetaOperator(var("a"), var("b")), mt.add)

    assert s[var("a")] == mt.add.op_def
    assert s[var("b")] == mt.add.node_def

    add_mt = reify(TFlowMetaOperator(var("a"), var("b")), s)

    assert add_mt == mt.add

    assert unify(mt.mul, mt.matmul) is False
    assert unify(mt.mul.op_def, mt.matmul.op_def) is False


@run_in_graph_mode
def test_etuple_term():

    assert etuplize("blah", return_bad_args=True) == "blah"

    a = tf.compat.v1.placeholder(tf.float64, name="a")
    b = tf.compat.v1.placeholder(tf.float64, name="b")

    a_mt = mt(a)
    a_mt._obj = None
    a_reified = a_mt.reify()
    assert isinstance(a_reified, tf.Tensor)
    assert a_reified.shape.dims is None

    with pytest.raises(TypeError):
        etuplize(a_mt.op.op_def)

    with pytest.raises(TypeError):
        etuplize(a_mt.op.node_def, shallow=False)

    with pytest.raises(TypeError):
        etuplize(a_mt, shallow=False)

    # Now, consider a meta graph with operator arguments
    add_mt = mt.AddV2(a, b)
    add_et = etuplize(add_mt, shallow=True)
    assert isinstance(add_et, ExpressionTuple)
    assert add_et[0].op_def == mt.AddV2.op_def

    # Check `kanren`'s term framework
    assert isinstance(operator(add_mt), TFlowMetaOperator)
    assert arguments(add_mt) == add_mt.op.inputs

    assert operator(add_mt)(*arguments(add_mt)) == add_mt

    assert isinstance(add_et[0], TFlowMetaOperator)
    assert add_et[1:] == add_mt.op.inputs
    assert operator(add_mt)(*arguments(add_mt)) == add_mt

    assert term(operator(add_mt), arguments(add_mt)) == add_mt

    add_mt = mt.AddV2(a, add_mt)
    add_et = etuplize(add_mt, shallow=False)

    assert isinstance(add_et, ExpressionTuple)
    assert len(add_et) == 3
    assert add_et[0].op_def == mt.AddV2.op_def
    assert len(add_et[2]) == 3
    assert add_et[2][0].op_def == mt.AddV2.op_def
    assert add_et.eval_obj is add_mt

    add_et._eval_obj = ExpressionTuple.null
    with tf.Graph().as_default():
        assert add_et.eval_obj == add_mt

    # Make sure things work with logic variables
    add_lvar_mt = TFlowMetaTensor(var(), var(), [1, 2])

    with pytest.raises(ConsError):
        assert operator(add_lvar_mt) is None

    with pytest.raises(ConsError):
        assert arguments(add_lvar_mt) is None


@run_in_graph_mode
def test_basic_unify_reify():
    # Test reification with manually constructed replacements
    a = tf.compat.v1.placeholder(tf.float64, name="a")
    x_l = var("x_l")
    a_reif = reify(x_l, {x_l: mt(a)})
    assert a_reif.obj is not None
    # Confirm that identity is preserved (i.e. that the underlying object
    # was properly tracked and not unnecessarily reconstructed)
    assert a == a_reif.reify()

    test_expr = mt.add(
        tf.constant(1, dtype=tf.float64), mt.mul(tf.constant(2, dtype=tf.float64), x_l)
    )
    test_reify_res = reify(test_expr, {x_l: a})
    test_base_res = test_reify_res.reify()
    assert isinstance(test_base_res, tf.Tensor)

    with tf.Graph().as_default():
        a = tf.compat.v1.placeholder(tf.float64, name="a")
        expected_res = tf.add(
            tf.constant(1, dtype=tf.float64), tf.multiply(tf.constant(2, dtype=tf.float64), a)
        )
    assert_ops_equal(test_base_res, expected_res)

    # Simply make sure that unification succeeds
    meta_expected_res = mt(expected_res)
    s_test = unify(test_expr, meta_expected_res, {})
    assert len(s_test) == 3

    assert reify(test_expr, s_test) == meta_expected_res


@run_in_graph_mode
def test_sexp_unify_reify():
    """Make sure we can unify and reify etuples/S-exps."""
    # Unify `A . (x + y)`, for `x`, `y` logic variables
    A = tf.compat.v1.placeholder(tf.float64, name="A", shape=tf.TensorShape([None, None]))
    x = tf.compat.v1.placeholder(tf.float64, name="x", shape=tf.TensorShape([None, 1]))
    y = tf.compat.v1.placeholder(tf.float64, name="y", shape=tf.TensorShape([None, 1]))

    z = tf.matmul(A, tf.add(x, y))

    z_sexp = etuplize(z, shallow=False)

    # Let's just be sure that the original TF objects are preserved
    assert z_sexp[1].reify() == A
    assert z_sexp[2][1].reify() == x
    assert z_sexp[2][2].reify() == y

    A_lv, x_lv, y_lv = var(), var(), var()
    dis_pat = etuple(
        TFlowMetaOperator(mt.matmul.op_def, var()),
        A_lv,
        etuple(TFlowMetaOperator(mt.add.op_def, var()), x_lv, y_lv),
    )

    s = unify(dis_pat, z_sexp, {})

    assert s[A_lv] == mt(A)
    assert s[x_lv] == mt(x)
    assert s[y_lv] == mt(y)

    # Now, we construct a graph that reflects the distributive property and
    # reify with the substitutions from the un-distributed form
    out_pat = etuple(mt.add, etuple(mt.matmul, A_lv, x_lv), etuple(mt.matmul, A_lv, y_lv))
    z_dist = reify(out_pat, s)

    # Evaluate the tuple-expression and get a meta object/graph
    z_dist_mt = z_dist.eval_obj

    # If all the logic variables were reified, we should be able to
    # further reify the meta graph and get a concrete TF graph
    z_dist_tf = z_dist_mt.reify()

    assert isinstance(z_dist_tf, tf.Tensor)

    # Check the first part of `A . x + A . y` (i.e. `A . x`)
    assert z_dist_tf.op.inputs[0].op.inputs[0] == A
    assert z_dist_tf.op.inputs[0].op.inputs[1] == x
    # Now, the second, `A . y`
    assert z_dist_tf.op.inputs[1].op.inputs[0] == A
    assert z_dist_tf.op.inputs[1].op.inputs[1] == y
