import pytest
import numpy as np

import tensorflow as tf

from tensorflow.python.eager.context import graph_mode
from tensorflow_probability import distributions as tfd

from unification import var, isvar

from symbolic_pymc.tensorflow.meta import (TFlowMetaTensor,
                                           TFlowMetaTensorShape,
                                           TFlowMetaConstant,
                                           _TFlowConstant,
                                           TFlowMetaOp,
                                           TFlowMetaOpDef,
                                           TFlowMetaNodeDef,
                                           TFlowOpName,
                                           mt)

from tests.tensorflow import run_in_graph_mode
from tests.tensorflow.utils import assert_ops_equal


@pytest.mark.usefixtures("run_with_tensorflow")
def test_op_names():
    """Make sure equality is flexible for `Operation`/`OpDef` names."""
    # Against a string, only the distinct operator-name part matters
    assert TFlowOpName('add_1') == 'add'
    assert TFlowOpName('blah/add_1:0') == 'add'
    assert TFlowOpName('blah/add_1:0') != 'add_1'
    assert TFlowOpName('blah/add_1:0') != 'add:0'
    # Unless it's the whole thing
    assert TFlowOpName('blah/add_1:0') == 'blah/add_1:0'

    # Ignore namespaces
    assert TFlowOpName('blah/add_1:0') == TFlowOpName('add:0')
    assert TFlowOpName('blah/add_1:0') == TFlowOpName('agh/add_1:0')
    # and "unique" operator names (for the same operator "type")
    assert TFlowOpName('blah/add_1:0') == TFlowOpName('add_2:0')
    # but not output numbers
    assert TFlowOpName('blah/add_1:0') != TFlowOpName('blah/add:1')


@pytest.mark.usefixtures("run_with_tensorflow")
def test_meta_helper():
    """Make sure the helper/namespace emulator can find `OpDef`s and create their meta objects."""
    assert isinstance(mt.add, TFlowMetaOpDef)
    assert mt.add.obj.name == 'Add'
    assert isinstance(mt.matmul, TFlowMetaOpDef)
    assert mt.matmul.obj.name == 'MatMul'
    assert isinstance(mt.RandomStandardNormal, TFlowMetaOpDef)
    assert mt.RandomStandardNormal.obj.name == 'RandomStandardNormal'


@pytest.mark.usefixtures("run_with_tensorflow")
def test_meta_eager():

    assert tf.executing_eagerly()

    N = 100
    X = np.vstack([np.random.randn(N), np.ones(N)]).T
    X_tf = tf.convert_to_tensor(X)

    with pytest.raises(ValueError):
        _ = mt(X_tf)

    with pytest.raises(ValueError):
        _ = mt(X)

    with graph_mode():
        N = 100
        X = np.vstack([np.random.randn(N), np.ones(N)]).T
        X_tf = tf.convert_to_tensor(X)
        _ = mt(X_tf)


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_create():
    N = 100
    X = np.vstack([np.random.randn(N), np.ones(N)]).T
    X_tf = tf.convert_to_tensor(X)
    X_mt = mt(X)

    assert isinstance(X_mt, TFlowMetaTensor)
    assert X_mt.op.obj.name.startswith('Const')
    # Make sure `reify` returns the cached base object.
    assert X_mt.reify() is X_mt.obj
    assert isinstance(X_mt.reify(), tf.Tensor)

    assert X_mt == mt(X_tf)

    # from google.protobuf import json_format
    # [i for i in X_tf.op.inputs]
    # print(json_format.MessageToJson(X_tf.op.op_def))

    # Create a (constant) tensor meta object manually.
    X_raw_mt = TFlowMetaConstant(obj=X)

    assert X_raw_mt.data is X

    # These are *not* equivalent, since they're constants without matching
    # constant values (well, our manually-created meta constant has no constant
    # value).
    assert X_mt == X_raw_mt
    # TODO: Should this be true?
    # assert X_mt.name == X_raw_mt.name

    add_mt = mt.add(1, 2)

    assert isinstance(add_mt, TFlowMetaTensor)
    assert isinstance(add_mt.obj, tf.Tensor)
    assert isinstance(add_mt.op.obj, tf.Operation)
    assert add_mt.op.obj.type == 'Add'

    assert len(add_mt.op.inputs) == 2
    assert all(isinstance(i, TFlowMetaTensor)
               for i in add_mt.op.inputs)

    one_mt, two_mt = mt(1), mt(2)

    assert one_mt != two_mt

    add_mt_2 = mt.add(one_mt, two_mt)

    assert isinstance(add_mt_2, TFlowMetaTensor)
    assert isinstance(add_mt_2.obj, tf.Tensor)
    assert isinstance(add_mt_2.op.obj, tf.Operation)
    assert add_mt_2.op.obj.type == 'Add'

    assert add_mt.obj is not None
    add_mt.name = None
    assert add_mt.obj is None
    add_mt_2.name = None

    # These aren't technically equal because of the TF auto-generated names,
    # but, since we're using special string wrappers for the names, it should
    # work in most cases.
    # However, the node_def input names will break equality, since even the TF
    # names aren't the same between these two different constructions:
    # tf.add(1, 2).op.node_def
    # tf.add(tf.convert_to_tensor(1), tf.convert_to_tensor(2)).op.node_def

    add_mt.op.node_def.input = [None, None]
    add_mt_2.op.node_def.input = [None, None]
    assert add_mt == add_mt_2

    a_mt = mt(tf.compat.v1.placeholder('float64', name='a', shape=[1, 2]))
    b_mt = mt(tf.compat.v1.placeholder('float64', name='b'))
    assert a_mt != b_mt

    assert a_mt.shape.ndims == 2
    assert a_mt.shape == TFlowMetaTensorShape([1, 2])

    # TODO: Create a placeholder using the string `Operator` name.
    z_mt = TFlowMetaTensor('float64', 'Placeholder', name='z__')

    assert z_mt.op.type == 'Placeholder'
    assert z_mt.name.startswith('z__')
    assert z_mt.obj.name.startswith('z__')

    with pytest.raises(TypeError):
        TFlowMetaTensor('float64', 'Add', name='q__')


@pytest.mark.usefixtures("run_with_tensorflow")
def test_meta_lvars():
    """Make sure we can use lvars as values."""

    nd_mt = TFlowMetaNodeDef(var(), var(), var())
    assert all(isvar(getattr(nd_mt, s)) for s in nd_mt.__slots__)

    mo_mt = TFlowMetaOp(var(), var(), var(), var())
    assert all(isvar(getattr(mo_mt, s)) for s in mo_mt.__slots__)

    ts_mt = TFlowMetaTensorShape(var())
    assert all(isvar(getattr(ts_mt, s)) for s in ts_mt.__slots__)

    assert isvar(ts_mt.as_list())

    tn_mt = TFlowMetaTensor(var(), var(), var(), var(), var())
    assert all(isvar(getattr(tn_mt, s)) for s in tn_mt.__slots__)


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_hashing():
    """Make sure we can hash meta graphs."""
    N = 100
    X = np.vstack([np.random.randn(N), np.ones(N)]).T
    X_mt = mt(X)

    assert isinstance(hash(X_mt), int)

    a_mt = mt(tf.compat.v1.placeholder('float32', name='a', shape=[1, 2]))
    add_mt = mt.add(tf.convert_to_tensor([1.0, 2.0]), mt.add(a_mt, a_mt))

    assert isinstance(hash(add_mt), int)


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_types():
    """Make sure our custom types/classes check out."""

    const_tf = tf.convert_to_tensor([1.0, 2.0])

    assert isinstance(const_tf, _TFlowConstant)


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_compare():
    """Make objects compare correctly."""

    a_tf = tf.compat.v1.placeholder('float', name='a', shape=[None, 1])
    z_tf = tf.multiply(2.0, a_tf)

    assert mt(z_tf) == mt(z_tf)

    const_tf = tf.convert_to_tensor([1.0, 2.0])
    const_mt = mt(const_tf)

    assert const_mt == const_mt
    assert const_mt == mt(const_tf)
    assert const_mt != const_tf
    assert const_mt != a_tf


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_multi_output():
    """Make sure we can handle TF `Operation`s that output more than on tensor."""
    d, U, V = mt.linalg.svd(var())

    assert d.op == U.op == V.op
    assert d.value_index == 0
    assert U.value_index == 1
    assert V.value_index == 2

    assert d.op.outputs == (d, U, V)
    assert d.op.default_output is d.op.outputs


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_reify():
    a_mt = mt(tf.compat.v1.placeholder('float64', name='a', shape=[1, 2]))
    b_mt = mt(tf.compat.v1.placeholder('float64', name='b', shape=[]))
    add_mt = mt.add(a_mt, b_mt)

    assert add_mt.shape.as_list() == [1, 2]

    add_tf = add_mt.reify()

    assert isinstance(add_tf, tf.Tensor)
    assert add_tf.op.type == 'Add'
    assert add_tf.shape.as_list() == [1, 2]

    # Remove cached base object and force manual reification.
    add_mt.obj = None
    add_tf = add_mt.reify()

    assert isinstance(add_tf, tf.Tensor)
    assert add_tf.op.type == 'Add'
    assert add_tf.shape.as_list() == [1, 2]


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_distributions():
    N = 100
    sigma_tf = tfd.Gamma(np.asarray(1.), np.asarray(1.)).sample()
    epsilon_tf = tfd.Normal(np.zeros((N, 1)), sigma_tf).sample()
    beta_tf = tfd.Normal(np.zeros((2, 1)), 1).sample()
    X = np.vstack([np.random.randn(N), np.ones(N)]).T
    X_tf = tf.convert_to_tensor(X)

    Y_tf = tf.linalg.matmul(X_tf, beta_tf) + epsilon_tf

    Y_mt = mt(Y_tf)

    # Confirm that all `Operation`s are the same.
    assert_ops_equal(Y_mt, Y_tf)

    # Now, let's see if we can reconstruct it entirely from the
    # meta objects.
    def _remove_obj(meta_obj):
        if (hasattr(meta_obj, 'obj') and
                not isinstance(meta_obj, TFlowMetaOpDef)):
            meta_obj.obj = None

        if hasattr(meta_obj, 'ancestors'):
            for a in meta_obj.ancestors or []:
                _remove_obj(a)

    _remove_obj(Y_mt)

    Y_mt_tf = Y_mt.reify()

    assert_ops_equal(Y_mt, Y_mt_tf)


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_inputs_remapping():
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    z = tf.concat([t1, t2], 0)

    z_mt = mt(z)

    assert isinstance(z_mt.inputs[0], list)
    assert z_mt.inputs[0][0].obj == z.op.inputs[0]
    assert z_mt.inputs[0][1].obj == z.op.inputs[1]
    assert z_mt.inputs[1].obj == z.op.inputs[2]


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_nodedef():
    with graph_mode():
        X = np.random.normal(0, 1, (10, 10))
        S = tf.matmul(X, X, transpose_a=True)
        d, U, V = tf.linalg.svd(S)
        node_def_mt = mt(d.op.node_def)

    assert 'compute_uv' in node_def_mt.attr
    assert 'full_matrices' in node_def_mt.attr
    assert 'T' not in node_def_mt.attr
