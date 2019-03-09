import pytest

import numpy as np

import tensorflow as tf

from unification import unify, reify, var, variables

from kanren.term import term, operator, arguments

from symbolic_pymc.tensorflow.meta import (TFlowName, mt)
# from symbolic_pymc.tensorflow.utils import graph_equal
from symbolic_pymc.unify import (ExpressionTuple, etuple, tuple_expression)

from tests.utils import assert_ops_equal


@pytest.mark.usefixtures("run_with_tensorflow")
def test_unification():
    a = tf.compat.v1.placeholder(tf.float64, name='a')
    x_l = var('x_l')
    a_reif = reify(x_l, {x_l: a})
    assert a_reif.obj is not None
    assert a == a_reif.reify()

    test_expr = mt.add(tf.constant(1, dtype=tf.float64),
                       mt.mul(tf.constant(2, dtype=tf.float64),
                              x_l))
    test_reify_res = reify(test_expr, {x_l: a})
    test_base_res = test_reify_res.reify()
    assert isinstance(test_base_res, tf.Tensor)

    expected_res = (tf.constant(1, dtype=tf.float64) +
                    tf.constant(2, dtype=tf.float64) * a)
    assert_ops_equal(test_base_res, expected_res)

    # from symbolic_pymc.unify import debug_unify; debug_unify()

    meta_expected_res = mt(expected_res)
    s_test = unify(test_expr, meta_expected_res, {})
    assert len(s_test) == 5

    assert reify(test_expr, s_test) == meta_expected_res


@pytest.mark.usefixtures("run_with_tensorflow")
def test_etuple_term():
    a = tf.compat.v1.placeholder(tf.float64, name='a')
    b = tf.compat.v1.placeholder(tf.float64, name='b')

    a_mt = mt(a)
    a_mt.obj = None
    a_reified = a_mt.reify()
    assert isinstance(a_reified, tf.Tensor)
    assert a_reified.shape.dims is None

    test_e = tuple_expression(a_mt)
    assert test_e[0] == mt.placeholder
    assert test_e[1] == tf.float64
    assert test_e[2][0].base == tf.TensorShape
    assert test_e[2][1] is None

    del test_e._eval_obj
    a_evaled = test_e.eval_obj
    assert all([a == b for a, b in zip(a_evaled.rands(), a_mt.rands())])

    a_reified = a_evaled.reify()
    assert isinstance(a_reified, tf.Tensor)
    assert a_reified.shape.dims is None
    assert TFlowName(a_reified.name) == TFlowName(a.name)

    e2 = mt.add(a, b)
    e2_et = tuple_expression(e2)
    assert isinstance(e2_et, ExpressionTuple)
    assert e2_et[0] == mt.add
