import io
import textwrap

import pytest
import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout

from unification import var, Var

from symbolic_pymc.tensorflow.meta import mt
from symbolic_pymc.tensorflow.printing import tf_dprint

from tests.tensorflow import run_in_graph_mode


def test_eager_mode():

    assert tf.executing_eagerly()

    N = 100
    X = np.vstack([np.random.randn(N), np.ones(N)]).T
    X_tf = tf.convert_to_tensor(X)

    with pytest.raises(ValueError):
        _ = tf_dprint(X_tf)


@run_in_graph_mode
def test_ascii_printing():
    """Make sure we can ascii/text print a TF graph."""

    A = tf.compat.v1.placeholder("float", name="A", shape=tf.TensorShape([None, None]))
    x = tf.compat.v1.placeholder("float", name="x", shape=tf.TensorShape([None, 1]))
    y = tf.multiply(1.0, x, name="y")

    z = tf.matmul(A, tf.add(y, y, name="x_p_y"), name="A_dot")

    std_out = io.StringIO()
    with redirect_stdout(std_out):
        tf_dprint(z)

    expected_out = textwrap.dedent(
        """
    Tensor(MatMul):0,\tdtype=float32,\tshape=[None, 1],\t"A_dot:0"
    |  Tensor(Placeholder):0,\tdtype=float32,\tshape=[None, None],\t"A:0"
    |  Tensor(Add):0,\tdtype=float32,\tshape=[None, 1],\t"x_p_y:0"
    |  |  Tensor(Mul):0,\tdtype=float32,\tshape=[None, 1],\t"y:0"
    |  |  |  Tensor(Const):0,\tdtype=float32,\tshape=[],\t"y/x:0"
    |  |  |  |  1.
    |  |  |  Tensor(Placeholder):0,\tdtype=float32,\tshape=[None, 1],\t"x:0"
    |  |  Tensor(Mul):0,\tdtype=float32,\tshape=[None, 1],\t"y:0"
    |  |  |  ...
    """
    )

    assert std_out.getvalue() == expected_out.lstrip()

    std_out = io.StringIO()
    with tf.Graph().as_default(), redirect_stdout(std_out):
        Var._id = 0
        tt_lv_inputs_mt = mt.Tensor(mt.Operation(var(), var(), var()), 0, var())
        tt_const_lv_nodedef_mt = mt.Tensor(mt.Operation(mt.Const.op_def, var(), ()), 0, var())
        tt_lv_op_mt = mt.Tensor(var(), 0, var())
        test_mt = mt(1) + tt_lv_inputs_mt + tt_const_lv_nodedef_mt + tt_lv_op_mt
        tf_dprint(test_mt)

    expected_out = textwrap.dedent(
        """
    Tensor(AddV2):0,\tdtype=int32,\tshape=~_11,\t"add:0"
    |  Tensor(AddV2):0,\tdtype=int32,\tshape=~_12,\t"add:0"
    |  |  Tensor(AddV2):0,\tdtype=int32,\tshape=~_13,\t"add:0"
    |  |  |  Tensor(Const):0,\tdtype=int32,\tshape=[],\t"Const:0"
    |  |  |  |  1
    |  |  |  Tensor(~_15):0,\tdtype=~_3,\tshape=~_14,\t"~_17"
    |  |  |  |  ~_2
    |  |  Tensor(Const):0,\tdtype=~_5,\tshape=~_18,\t"~_20"
    |  |  |  ~_4
    |  Tensor(~_6):0,\tdtype=~_7,\tshape=~_21,\t"~_22"
    """
    )

    assert std_out.getvalue() == expected_out.lstrip()


@run_in_graph_mode
def test_unknown_shape():
    """Make sure we can ascii/text print a TF graph with unknown shapes."""

    A = tf.compat.v1.placeholder(tf.float64, name="A")

    std_out = io.StringIO()
    with redirect_stdout(std_out):
        tf_dprint(A)

    expected_out = 'Tensor(Placeholder):0,\tdtype=float64,\tshape=Unknown,\t"A:0"\n'

    assert std_out.getvalue() == expected_out.lstrip()


@run_in_graph_mode
def test_numpy():
    """Make sure we can ascii/text print constant tensors with large Numpy arrays."""

    with tf.Graph().as_default():
        A = tf.convert_to_tensor(np.arange(100))

    std_out = io.StringIO()
    with redirect_stdout(std_out):
        tf_dprint(A)

    expected_out = textwrap.dedent(
        """
    Tensor(Const):0,\tdtype=int64,\tshape=[100],\t"Const:0"
    |  [ 0  1  2 ... 97 98 99]
    """
    )

    assert std_out.getvalue() == expected_out.lstrip()

    N = 100
    np.random.seed(12345)
    X = np.vstack([np.random.randn(N), np.ones(N)]).T

    with tf.Graph().as_default():
        X_tf = tf.convert_to_tensor(X)

    std_out = io.StringIO()
    with redirect_stdout(std_out):
        tf_dprint(X_tf)

    expected_out = textwrap.dedent(
        """
    Tensor(Const):0,\tdtype=float64,\tshape=[100, 2],\t"Const:0"
    |  [[-0.20470766  1.        ]
        [ 0.47894334  1.        ]
        [-0.51943872  1.        ]
        ...
        [-0.74853155  1.        ]
        [ 0.58496974  1.        ]
        [ 0.15267657  1.        ]]
    """
    )

    assert std_out.getvalue() == expected_out.lstrip()
