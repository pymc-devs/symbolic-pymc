from collections.abc import Iterator

import numpy as np
import theano
import theano.tensor as tt
import pytest

from unification import var, isvar, variables
from symbolic_pymc.meta import MetaSymbol
from symbolic_pymc.theano.meta import (metatize,
                                       TheanoMetaTensorVariable,
                                       TheanoMetaTensorType, mt)
from symbolic_pymc.theano.utils import graph_equal


@pytest.mark.usefixtures("run_with_theano")
def test_metatize():
    vec_tt = tt.vector('vec')
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

    assert isinstance(test_out, MetaSymbol)
    assert test_out.obj == TestOp
    assert test_out.base == TestOp


@pytest.mark.usefixtures("run_with_theano")
def test_meta_classes():
    vec_tt = tt.vector('vec')
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
