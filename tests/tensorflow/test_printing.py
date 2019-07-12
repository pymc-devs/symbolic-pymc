import io
import textwrap

import pytest
import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout

from tensorflow.python.eager.context import graph_mode

from symbolic_pymc.tensorflow.printing import tf_dprint

from tests.tensorflow import run_in_graph_mode


@pytest.mark.usefixtures("run_with_tensorflow")
def test_eager_mode():

    assert tf.executing_eagerly()

    N = 100
    X = np.vstack([np.random.randn(N), np.ones(N)]).T
    X_tf = tf.convert_to_tensor(X)

    with pytest.raises(ValueError):
        tf_dprint(X_tf)

    with graph_mode():
        X_tf = tf.convert_to_tensor(X)
        tf_dprint(X_tf)


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_ascii_printing():
    """Make sure we can ascii/text print a TF graph."""

    A = tf.compat.v1.placeholder('float', name='A',
                                 shape=tf.TensorShape([None, None]))
    x = tf.compat.v1.placeholder('float', name='x',
                                 shape=tf.TensorShape([None, 1]))
    y = tf.multiply(1.0, x, name='y')

    z = tf.matmul(A, tf.add(y, y, name='x_p_y'), name='A_dot')

    std_out = io.StringIO()
    with redirect_stdout(std_out):
        tf_dprint(z)

    expected_out = textwrap.dedent('''
    Tensor(MatMul):0,\tshape=[None, 1]\t"A_dot:0"
    |  Op(MatMul)\t"A_dot"
    |  |  Tensor(Placeholder):0,\tshape=[None, None]\t"A:0"
    |  |  Tensor(Add):0,\tshape=[None, 1]\t"x_p_y:0"
    |  |  |  Op(Add)\t"x_p_y"
    |  |  |  |  Tensor(Mul):0,	shape=[None, 1]	"y:0"
    |  |  |  |  |  Op(Mul)\t"y"
    |  |  |  |  |  |  Tensor(Const):0,\tshape=[]\t"y/x:0"
    |  |  |  |  |  |  Tensor(Placeholder):0,\tshape=[None, 1]\t"x:0"
    |  |  |  |  Tensor(Mul):0,\tshape=[None, 1]\t"y:0"
    |  |  |  |  |  ...
    ''')

    assert std_out.getvalue() == expected_out.lstrip()


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_unknown_shape():
    """Make sure we can ascii/text print a TF graph with unknown shapes."""

    A = tf.compat.v1.placeholder(tf.float64, name='A')

    std_out = io.StringIO()
    with redirect_stdout(std_out):
        tf_dprint(A)

    expected_out = 'Tensor(Placeholder):0,\tshape=Unknown\t"A:0"\n'

    assert std_out.getvalue() == expected_out.lstrip()
