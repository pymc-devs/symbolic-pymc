import theano
import theano.tensor as tt

from unification import var
from symbolic_pymc.meta import (MetaSymbol, MetaTensorVariable, MetaTensorType,
                                mt)
from symbolic_pymc.utils import graph_equal


def test_meta_classes():
    vec_tt = tt.vector('vec')
    vec_m = MetaSymbol.from_obj(vec_tt)
    assert vec_m.obj == vec_tt
    assert type(vec_m) == MetaTensorVariable

    # This should invalidate the underlying base object.
    vec_m.index = 0
    assert vec_m.obj is None
    assert vec_m.reify().type == vec_tt.type
    assert vec_m.reify().name == vec_tt.name

    vec_type_m = vec_m.type
    assert type(vec_type_m) == MetaTensorType
    assert vec_type_m.dtype == vec_tt.dtype
    assert vec_type_m.broadcastable == vec_tt.type.broadcastable
    assert vec_type_m.name == vec_tt.type.name

    assert graph_equal(tt.add(1, 2), mt.add(1, 2).reify())

    meta_var = mt.add(1, var()).reify()
    assert isinstance(meta_var, MetaTensorVariable)
    assert isinstance(meta_var.owner.op.obj, theano.Op)
    assert isinstance(meta_var.owner.inputs[0].obj, tt.TensorConstant)

    test_vals = [1, 2.4]
    meta_vars = MetaSymbol.from_obj(test_vals)
    assert meta_vars == [MetaSymbol.from_obj(x) for x in test_vals]
    # TODO: Do we really want meta variables to be equal to their
    # reified base objects?
    # assert meta_vars == [tt.as_tensor_variable(x) for x in test_vals]
