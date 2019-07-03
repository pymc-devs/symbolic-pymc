import io
import textwrap

import pytest

import tensorflow as tf

from contextlib import redirect_stdout

from symbolic_pymc.tensorflow.printing import tf_dprint


@pytest.mark.usefixtures("run_with_tensorflow")
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
def test_unknown_shape():
    """Make sure we can ascii/text print a TF graph with unknown shapes."""

    A = tf.compat.v1.placeholder(tf.float64, name='A')

    std_out = io.StringIO()
    with redirect_stdout(std_out):
        tf_dprint(A)

    expected_out = 'Tensor(Placeholder):0,\tshape=Unknown\t"A:0"\n'

    assert std_out.getvalue() == expected_out.lstrip()
