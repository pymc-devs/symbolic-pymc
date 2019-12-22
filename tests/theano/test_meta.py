from collections.abc import Iterator

import numpy as np
import theano
import theano.tensor as tt
import pytest

from unification import var, isvar, variables
from symbolic_pymc.meta import MetaSymbol, MetaOp
from symbolic_pymc.theano.meta import (
    metatize,
    TheanoMetaOp,
    TheanoMetaApply,
    TheanoMetaVariable,
    TheanoMetaTensorConstant,
    TheanoMetaTensorVariable,
    TheanoMetaTensorType,
    mt,
)
from symbolic_pymc.theano.utils import graph_equal


@pytest.mark.usefixtures("run_with_theano")
def test_metatize():
    vec_tt = tt.vector("vec")
    vec_m = metatize(vec_tt)
    assert vec_m.base == type(vec_tt)

    test_list = [1, 2, 3]
    metatize_test_list = metatize(test_list)
    assert isinstance(metatize_test_list, list)
    assert all(isinstance(m, MetaSymbol) for m in metatize_test_list)

    test_iter = iter([1, 2, 3])
    metatize_test_iter = metatize(test_iter)
    assert isinstance(metatize_test_iter, Iterator)
    assert all(isinstance(m, MetaSymbol) for m in metatize_test_iter)

    test_out = metatize(var())
    assert isvar(test_out)

    with variables(vec_tt):
        test_out = metatize(vec_tt)
        assert test_out == vec_tt
        assert isvar(test_out)

    test_out = metatize(np.r_[1, 2, 3])
    assert isinstance(test_out, MetaSymbol)

    class TestClass(object):
        pass

    with pytest.raises(Exception):
        metatize(TestClass())

    class TestOp(tt.gof.Op):
        pass

    test_out = metatize(TestOp)
    assert issubclass(test_out, MetaOp)

    test_op_tt = TestOp()
    test_obj = test_out(obj=test_op_tt)
    assert isinstance(test_obj, MetaSymbol)
    assert test_obj.obj == test_op_tt
    assert test_obj.base == TestOp


@pytest.mark.usefixtures("run_with_theano")
def test_meta_classes():
    vec_tt = tt.vector("vec")
    vec_m = metatize(vec_tt)
    assert vec_m.obj == vec_tt
    assert type(vec_m) == TheanoMetaTensorVariable

    # This should invalidate the underlying base object.
    vec_m.index = 0
    assert vec_m.obj is None
    assert vec_m.reify().type == vec_tt.type
    assert vec_m.reify().name == vec_tt.name

    vec_type_m = vec_m.type
    assert type(vec_type_m) == TheanoMetaTensorType
    assert vec_type_m.dtype == vec_tt.dtype
    assert vec_type_m.broadcastable == vec_tt.type.broadcastable
    assert vec_type_m.name == vec_tt.type.name

    assert graph_equal(tt.add(1, 2), mt.add(1, 2).reify())

    meta_var = mt.add(1, var()).reify()
    assert isinstance(meta_var, TheanoMetaTensorVariable)
    assert isinstance(meta_var.owner.op.obj, theano.Op)
    assert isinstance(meta_var.owner.inputs[0].obj, tt.TensorConstant)

    test_vals = [1, 2.4]
    meta_vars = metatize(test_vals)
    assert meta_vars == [metatize(x) for x in test_vals]
    # TODO: Do we really want meta variables to be equal to their
    # reified base objects?
    # assert meta_vars == [tt.as_tensor_variable(x) for x in test_vals]

    name_mt = var()
    add_tt = tt.add(0, 1)
    add_mt = mt.add(0, 1, name=name_mt)

    assert add_mt.name is name_mt
    assert add_tt.type == add_mt.type.reify()
    assert mt(add_tt.owner) == add_mt.owner
    # assert isvar(add_mt._obj)

    # Let's confirm that we can dynamically create a new meta `Op` type
    test_mat = np.c_[[2, 3], [4, 5]]

    svd_tt = tt.nlinalg.SVD()(test_mat)
    # First, can we create one from a new base `Op` instance?
    svd_op_mt = mt(tt.nlinalg.SVD())
    svd_mt = svd_op_mt(test_mat)

    assert svd_mt[0].owner.nin == 1
    assert svd_mt[0].owner.nout == 3

    svd_outputs = svd_mt[0].owner.outputs
    assert svd_outputs[0] == svd_mt[0]
    assert svd_outputs[1] == svd_mt[1]
    assert svd_outputs[2] == svd_mt[2]

    assert mt(svd_tt) == svd_mt

    # Next, can we create one from a base `Op` type/class?
    svd_op_type_mt = mt.nlinalg.SVD
    assert isinstance(svd_op_type_mt, type)
    assert issubclass(svd_op_type_mt, TheanoMetaOp)

    # svd_op_inst_mt = svd_op_type_mt(tt.nlinalg.SVD())
    # svd_op_inst_mt(test_mat) == svd_mt

    # Apply node with logic variable as outputs
    svd_apply_mt = TheanoMetaApply(svd_op_mt, [test_mat], outputs=var("out"))
    assert isinstance(svd_apply_mt.inputs, tuple)
    assert isinstance(svd_apply_mt.inputs[0], MetaSymbol)
    assert isvar(svd_apply_mt.outputs)
    assert svd_apply_mt.nin == 1
    assert svd_apply_mt.nout is None

    # Apply node with logic variable as inputs
    svd_apply_mt = TheanoMetaApply(svd_op_mt, var("in"), outputs=var("out"))
    assert svd_apply_mt.nin is None

    # A meta variable with None index
    var_mt = TheanoMetaVariable(svd_mt[0].type, svd_mt[0].owner, None, None)
    assert var_mt.index is None
    reified_var_mt = var_mt.reify()

    assert isinstance(reified_var_mt, TheanoMetaTensorVariable)
    assert reified_var_mt.index == 0
    assert var_mt.index == 0
    assert reified_var_mt == svd_mt[0]

    # A meta variable with logic variable index
    var_mt = TheanoMetaVariable(svd_mt[0].type, svd_mt[0].owner, var("index"), None)
    assert isvar(var_mt.index)
    reified_var_mt = var_mt.reify()
    assert isvar(var_mt.index)
    assert reified_var_mt.index == 0

    const_mt = mt(1)
    assert isinstance(const_mt, TheanoMetaTensorConstant)
    assert const_mt != mt(2)


@pytest.mark.usefixtures("run_with_theano")
def test_meta_str():
    assert str(mt.add) == "TheanoMetaElemwise(Elemwise{add,no_inplace})"


@pytest.mark.usefixtures("run_with_theano")
def test_meta_pretty():
    pretty_mod = pytest.importorskip("IPython.lib.pretty")
    assert pretty_mod.pretty(mt.add) == "TheanoMetaElemwise(Elemwise{add,no_inplace})"


@pytest.mark.usefixtures("run_with_theano")
def test_meta_helpers():
    zeros_mt = mt.zeros(2)
    assert np.array_equal(zeros_mt.reify().eval(), np.r_[0.0, 0.0])

    zeros_mt = mt.zeros(2, dtype=int)
    assert np.array_equal(zeros_mt.reify().eval(), np.r_[0, 0])

    mat_mt = mt(np.eye(2))
    diag_mt = mt.diag(mat_mt)
    assert np.array_equal(diag_mt.reify().eval(), np.r_[1.0, 1.0])

    diag_mt = mt.diag(mt(np.r_[1, 2, 3]))
    assert np.array_equal(diag_mt.reify().eval(), np.diag(np.r_[1, 2, 3]))
