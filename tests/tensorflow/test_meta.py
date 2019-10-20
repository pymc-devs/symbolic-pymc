"""
If you're debugging/running tests manually, it might help to simply
disable eager execution entirely:

    tf.compat.v1.disable_eager_execution()
"""
import pytest
import numpy as np

import tensorflow as tf

from tensorflow.python.eager.context import graph_mode
from tensorflow_probability import distributions as tfd

from unification import var, isvar

from symbolic_pymc.meta import MetaSymbol
from symbolic_pymc.tensorflow.meta import (TFlowMetaTensor,
                                           TFlowMetaTensorShape,
                                           TFlowMetaOp,
                                           TFlowMetaOpDef,
                                           TFlowMetaNodeDef,
                                           MetaOpDefLibrary,
                                           mt)

from tests.tensorflow import run_in_graph_mode
from tests.tensorflow.utils import assert_ops_equal


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

    with pytest.raises(AttributeError):
        _ = mt(X_tf)

    with pytest.raises(AttributeError):
        _ = mt(X)

    with graph_mode():
        N = 100
        X = np.vstack([np.random.randn(N), np.ones(N)]).T
        X_tf = tf.convert_to_tensor(X)
        _ = mt(X_tf)


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_basic():

    var_mt = TFlowMetaTensor(var(), var(), var())
    # It should generate a logic variable for the name and use from here on.
    var_name = var_mt.name
    assert isvar(var_name)
    assert var_mt.name is var_name
    # Same for a tensor shape
    var_shape = var_mt.shape
    assert isinstance(var_shape, TFlowMetaTensorShape)
    assert isvar(var_shape.dims)

    # This essentially logic-variabled tensor should not reify; it should
    # create a distinct/new meta object that's either equal to the original
    # meta object or partially reified.
    assert var_mt.reify() == var_mt

    # This operator is reifiable
    # NOTE: Const objects are automatically created for the constant inputs, so
    # we need to do this in a new graph to make sure that their auto-generated
    # names are consistent throughout runs.
    with tf.Graph().as_default() as test_graph:
        test_op = TFlowMetaOp(mt.Add, TFlowMetaNodeDef('Add', 'Add', {}), [1, 0])

        # This tensor has an "unknown"/logic variable output index and dtype, but,
        # since the operator fully specifies it, reification should still work.
        var_mt = TFlowMetaTensor(test_op, var(), var())

        # This should be partially reified
        var_tf = var_mt.reify()

        assert isinstance(var_tf, tf.Tensor)

        # These shouldn't be equal, since `var_mt` has logic variables for
        # output index and dtype.  (They should be unifiable, though.)
        assert mt(var_tf) != var_mt

        # NOTE: The operator name specified by the meta NodeDef *can* be
        # different from the reified TF tensor (e.g. when meta objects are
        # created/reified within a graph already using the NodeDef-specified
        # name).
        #
        # TODO: We could search for existing TF objects in the current graph by
        # name and raise exceptions when the desired meta information and name
        # do not correspond--effectively making the meta object impossible to
        # reify in said graph.

    # Next, we convert an existing TF object into a meta object
    # and make sure everything corresponds between the two.
    N = 100
    X = np.vstack([np.random.randn(N), np.ones(N)]).T

    X_tf = tf.convert_to_tensor(X)

    with tf.Graph().as_default() as test_graph:
        X_mt = mt(X)

    assert isinstance(X_mt, TFlowMetaTensor)
    assert X_mt.op.obj.name == 'Const'
    assert not hasattr(X_mt, '_name')
    assert X_mt.name == 'Const:0'
    assert X_mt._name == 'Const:0'

    # Make sure `reify` returns the cached base object.
    assert X_mt.reify() is X_mt.obj
    assert isinstance(X_mt.reify(), tf.Tensor)

    assert X_mt == mt(X_tf)

    # Create a (constant) tensor meta object manually.
    X_raw_mt = TFlowMetaTensor(X_tf.op, X_tf.value_index, X_tf.dtype, obj=X_tf)

    assert np.array_equal(X_raw_mt.op.node_def.attr['value'], X)

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

    a_mt = mt(tf.compat.v1.placeholder('float64', name='a', shape=[1, 2]))
    b_mt = mt(tf.compat.v1.placeholder('float64', name='b'))
    assert a_mt != b_mt

    assert a_mt.shape.ndims == 2
    assert a_mt.shape == TFlowMetaTensorShape([1, 2])

    # Make sure that names are properly inferred when there are no base objects
    # to reference
    with tf.Graph().as_default():
        one_mt = mt(1.0)
        log_mt = mt.log(one_mt)
        assert log_mt.name == 'Log:0'
        assert log_mt.dtype == tf.float32
        assert log_mt.op.outputs[0].dtype == tf.float32

        log_mt._name = None
        one_mt._obj = None
        log_mt._obj = None
        assert log_mt.dtype == tf.float32
        assert log_mt.name == 'Log:0'

        log_mt = mt.log(var(), name=var())
        assert isvar(log_mt.name)


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_meta_Op():

    t1_tf = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]])
    t2_tf = tf.convert_to_tensor([[7, 8, 9], [10, 11, 12]])
    test_out_tf = tf.concat([t1_tf, t2_tf], 0)

    # TODO: Without explicit conversion, each element within these arrays gets
    # converted to a `Tensor` by `metatize`.  That doesn't seem very
    # reasonable.  Likewise, the `0` gets converted, but it probably shouldn't be.
    test_op = TFlowMetaOp(mt.Concat, var(), [[t1_tf, t2_tf], 0])

    assert isvar(test_op.name)

    # Make sure we converted lists to tuples
    assert isinstance(test_op.inputs, tuple)
    assert isinstance(test_op.inputs[0], tuple)

    test_op = TFlowMetaOp(mt.Concat, var(), [[t1_tf, t2_tf], 0], outputs=[test_out_tf])

    # NodeDef is a logic variable, so this shouldn't actually reify.
    assert MetaSymbol.is_meta(test_op.reify())
    assert isinstance(test_op.outputs, tuple)
    assert MetaSymbol.is_meta(test_op.outputs[0])


@pytest.mark.usefixtures("run_with_tensorflow")
def test_meta_lvars():
    """Make sure we can use lvars as values."""

    nd_mt = TFlowMetaNodeDef(var(), var(), var())
    assert all(isvar(getattr(nd_mt, s)) for s in nd_mt.__all_props__)

    mo_mt = TFlowMetaOp(var(), var(), var(), var())
    assert all(isvar(getattr(mo_mt, s)) for s in mo_mt.__all_props__)

    ts_mt = TFlowMetaTensorShape(var())
    assert all(isvar(getattr(ts_mt, s)) for s in ts_mt.__all_props__)

    assert isvar(ts_mt.as_list())

    tn_mt = TFlowMetaTensor(var(), var(), var())
    assert all(isvar(getattr(tn_mt, s)) for s in tn_mt.__all_props__)


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
    add_mt._obj = None
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
        if (hasattr(meta_obj, '_obj') and
                not isinstance(meta_obj, TFlowMetaOpDef)):
            meta_obj._obj = None

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

    # Even though we gave it unhashable arguments, the operator should've
    # converted them
    assert isinstance(z_mt.inputs[0], tuple)
    assert z_mt.inputs[0][0].obj == z.op.inputs[0]
    assert z_mt.inputs[0][1].obj == z.op.inputs[1]
    assert z_mt.inputs[1].obj == z.op.inputs[2]


@pytest.mark.usefixtures("run_with_tensorflow")
def test_opdef_sig():
    """Make sure we can construct an `inspect.Signature` object for a protobuf OpDef when its corresponding function isn't present in `tf.raw_ops`."""
    from tensorflow.core.framework import op_def_pb2

    custom_opdef_tf = op_def_pb2.OpDef()
    custom_opdef_tf.name = "MyOpDef"

    arg1_tf = op_def_pb2.OpDef.ArgDef()
    arg1_tf.name = "arg1"
    arg1_tf.type_attr = "T"

    arg2_tf = op_def_pb2.OpDef.ArgDef()
    arg2_tf.name = "arg2"
    arg2_tf.type_attr = "T"

    custom_opdef_tf.input_arg.extend([arg1_tf, arg2_tf])

    attr1_tf = op_def_pb2.OpDef.AttrDef()
    attr1_tf.name = "T"
    attr1_tf.type = "type"

    attr2_tf = op_def_pb2.OpDef.AttrDef()
    attr2_tf.name = "axis"
    attr2_tf.type = "int"
    attr2_tf.default_value.i = 1

    custom_opdef_tf.attr.extend([attr1_tf, attr2_tf])

    opdef_sig, opdef_func = MetaOpDefLibrary.make_opdef_sig(custom_opdef_tf)

    import inspect
    # These are standard inputs
    assert opdef_sig.parameters['arg1'].default == inspect._empty
    assert opdef_sig.parameters['arg2'].default == inspect._empty
    # These are attributes that are sometimes required by the OpDef
    assert opdef_sig.parameters['axis'].default == inspect._empty
    # The obligatory tensor name parameter
    assert opdef_sig.parameters['name'].default is None


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_nodedef():
    X = np.random.normal(0, 1, (10, 10))
    S = tf.matmul(X, X, transpose_a=True)
    d, U, V = tf.linalg.svd(S)
    node_def_mt = mt(d.op.node_def)

    assert 'compute_uv' in node_def_mt.attr
    assert 'full_matrices' in node_def_mt.attr

    # Some outputs use nodedef information; let's test those.
    norm_rv = mt.RandomStandardNormal(mean=0, stddev=1, shape=(1000,), dtype=tf.float32, name=var())
    assert isinstance(norm_rv, TFlowMetaTensor)
    assert norm_rv.dtype == tf.float32

    # We shouldn't be metatizing all parsed `node_def.attr` values; otherwise,
    # we won't be able to reconstruct corresponding meta Ops using their meta
    # OpDefs and inputs.
    x_test = tf.constant([1.8, 2.2], dtype=tf.float32)
    y_test = tf.dtypes.cast(x_test, dtype=tf.int32, name="y")
    y_test_mt = mt(y_test)

    # `ytest_mt.inputs` should have two `.attr` values that are Python
    # primitives (i.e. int and bool); these shouldn't get metatized and break
    # our ability to reconstruct the object from its rator + rands.
    y_test_new_mt = y_test_mt.op.op_def(*y_test_mt.inputs)

    # We're changing this simply so we can use ==
    y_test_new_mt.op.node_def.name = 'y'

    assert y_test_mt == y_test_new_mt


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_metatize():
    class CustomClass(object):
        pass

    with pytest.raises(ValueError):
        mt(CustomClass())


@pytest.mark.usefixtures("run_with_tensorflow")
@run_in_graph_mode
def test_opdef_func():
    sum_mt = mt.Sum([[1, 2]], [1])
    sum_tf = sum_mt.reify()

    with tf.compat.v1.Session() as sess:
        assert sum_tf.eval() == np.r_[3]
