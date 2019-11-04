import pytest

import tensorflow as tf

from unification import unify, reify, var
from kanren.term import term, operator, arguments

from symbolic_pymc.tensorflow.meta import (mt, TFlowMetaOperator, TFlowMetaTensor, TFlowMetaNodeDef)
from symbolic_pymc.etuple import (ExpressionTuple, etuple, etuplize)

from tests.tensorflow import run_in_graph_mode
from tests.tensorflow.utils import assert_ops_equal


@run_in_graph_mode
def test_operator():
    s = unify(TFlowMetaOperator(var('a'), var('b')), mt.add)

    assert s[var('a')] == mt.add.op_def
    assert s[var('b')] == mt.add.node_def

    add_mt = reify(TFlowMetaOperator(var('a'), var('b')), s)

    assert add_mt == mt.add


@run_in_graph_mode
def test_etuple_term():

    assert etuplize("blah", return_bad_args=True) == "blah"

    a = tf.compat.v1.placeholder(tf.float64, name='a')
    b = tf.compat.v1.placeholder(tf.float64, name='b')

    a_mt = mt(a)
    a_mt._obj = None
    a_reified = a_mt.reify()
    assert isinstance(a_reified, tf.Tensor)
    assert a_reified.shape.dims is None

    with pytest.raises(TypeError):
        etuplize(a_mt.op.op_def)

    a_nd_e = etuplize(a_mt.op.node_def, shallow=False)
    assert a_nd_e[0] is TFlowMetaNodeDef
    assert a_nd_e[1] == a_mt.op.node_def.op
    assert a_nd_e[2] == a_mt.op.node_def.name
    assert a_nd_e[3] == a_mt.op.node_def.attr

    # A deep etuplization
    test_e = etuplize(a_mt, shallow=False)
    assert len(test_e) == 1
    assert len(test_e[0]) == 3
    assert test_e[0][0] is TFlowMetaOperator
    assert test_e[0][1] is a_mt.op.op_def
    assert test_e[0][2] == a_nd_e

    assert test_e.eval_obj is a_mt

    test_e._eval_obj = ExpressionTuple.null
    with tf.Graph().as_default():
        a_evaled = test_e.eval_obj
    assert a_evaled == a_mt

    # A shallow etuplization
    test_e = etuplize(a_mt, shallow=True)
    assert len(test_e) == 1
    assert isinstance(test_e[0], TFlowMetaOperator)
    assert test_e[0].op_def is a_mt.op.op_def
    assert test_e[0].node_def is a_mt.op.node_def

    assert test_e.eval_obj is a_mt

    test_e._eval_obj = ExpressionTuple.null
    with tf.Graph().as_default():
        a_evaled = test_e.eval_obj
    assert a_evaled == a_mt

    a_reified = a_evaled.reify()
    assert isinstance(a_reified, tf.Tensor)
    assert a_reified.shape.dims is None

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

    # Make sure things work with logic variables
    add_lvar_mt = TFlowMetaTensor(var(), var(), [1, 2])

    # TODO FIXME: This is bad
    assert operator(add_lvar_mt) is None
    # assert operator(add_lvar_mt) == add_lvar_mt.op
    # TODO FIXME: Same here
    assert arguments(add_lvar_mt) is None
    # assert arguments(add_lvar_mt) == add_lvar_mt.inputs

    # TODO FIXME: Because of the above two, this errs
    # add_lvar_et = etuplize(add_lvar_mt)

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

    with tf.Graph().as_default():
        a = tf.compat.v1.placeholder(tf.float64, name='a')
        expected_res = tf.add(tf.constant(1, dtype=tf.float64),
                              tf.multiply(tf.constant(2, dtype=tf.float64), a))
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
    A = tf.compat.v1.placeholder(tf.float64, name='A',
                                 shape=tf.TensorShape([None, None]))
    x = tf.compat.v1.placeholder(tf.float64, name='x',
                                 shape=tf.TensorShape([None, 1]))
    y = tf.compat.v1.placeholder(tf.float64, name='y',
                                 shape=tf.TensorShape([None, 1]))

    z = tf.matmul(A, tf.add(x, y))

    z_sexp = etuplize(z, shallow=False)

    # Let's just be sure that the original TF objects are preserved
    assert z_sexp[1].eval_obj.reify() == A
    assert z_sexp[2][1].eval_obj.reify() == x
    assert z_sexp[2][2].eval_obj.reify() == y

    dis_pat = etuple(etuple(TFlowMetaOperator, mt.matmul.op_def, var()),
                     var('A'),
                     etuple(etuple(TFlowMetaOperator, mt.add.op_def, var()),
                            var('x'), var('y')))

    s = unify(dis_pat, z_sexp, {})

    assert s[var('A')].eval_obj == mt(A)
    assert s[var('x')].eval_obj == mt(x)
    assert s[var('y')].eval_obj == mt(y)

    # Now, we construct a graph that reflects the distributive property and
    # reify with the substitutions from the un-distributed form
    out_pat = etuple(mt.add,
                     etuple(mt.matmul, var('A'), var('x')),
                     etuple(mt.matmul, var('A'), var('y')))
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
