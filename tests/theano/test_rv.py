import numpy as np

import theano.tensor as tt

from symbolic_pymc.theano.random_variables import NormalRV, MvNormalRV


def rv_numpy_tester(rv, *params, size=None):
    """Test for correspondence between `RandomVariable` and NumPy shape and
    broadcast dimensions.
    """
    tt.config.compute_test_value = "ignore"

    test_rv = rv(*params, size=size)
    param_vals = [tt.gof.op.get_test_value(p) for p in params]
    size_val = None if size is None else tt.gof.op.get_test_value(size)
    test_val = getattr(np.random, rv.name)(*param_vals, size=size_val)
    test_shp = np.shape(test_val)

    # This might be a little too harsh, since purely symbolic `tensor.vector`
    # inputs have no broadcastable information, yet, they can take
    # broadcastable values.
    # E.g.
    #     x_tt = tt.vector('x')
    #     # non-symbolic value is broadcastable!
    #     x_tt.tag.test_value = np.array([5])
    #     # non-symbolic value is not broadcastable.
    #     x_tt.tag.test_value = np.array([5, 4])
    #
    # In the end, there's really no clear way to determine this without full
    # evaluation of a symbolic node, and that mostly defeats the purpose.
    # Unfortunately, this is what PyMC3 resorts to when constructing its
    # `TensorType`s (and shapes).

    test_bcast = [s == 1 for s in test_shp]
    np.testing.assert_array_equal(test_rv.type.broadcastable, test_bcast)

    eval_args = {
        p: v
        for p, v in zip(params, param_vals)
        if isinstance(p, tt.Variable) and not isinstance(p, tt.Constant)
    }
    np.testing.assert_array_equal(test_rv.shape.eval(eval_args), test_shp)
    np.testing.assert_array_equal(np.shape(test_rv.eval(eval_args)), test_shp)


def test_normalrv():
    rv_numpy_tester(NormalRV, 0.0, 1.0)
    rv_numpy_tester(NormalRV, 0.0, 1.0, size=[3])
    # Broadcast sd over independent means...
    rv_numpy_tester(NormalRV, [0.0, 1.0, 2.0], 1.0)
    rv_numpy_tester(NormalRV, [0.0, 1.0, 2.0], 1.0, size=[3, 3])
    rv_numpy_tester(NormalRV, [0], [1], size=[1])

    rv_numpy_tester(NormalRV, tt.as_tensor_variable([0]), [1], size=[1])
    rv_numpy_tester(NormalRV, tt.as_tensor_variable([0]), [1], size=tt.as_tensor_variable([1]))


def test_mvnormalrv():
    rv_numpy_tester(MvNormalRV, [0], np.diag([1]))
    rv_numpy_tester(MvNormalRV, [0], np.diag([1]), size=[1])
    rv_numpy_tester(MvNormalRV, [0], np.diag([1]), size=[4])
    rv_numpy_tester(MvNormalRV, [0], np.diag([1]), size=[4, 1])
    rv_numpy_tester(MvNormalRV, [0], np.diag([1]), size=[4, 1, 1])
    rv_numpy_tester(MvNormalRV, [0], np.diag([1]), size=[1, 5, 8])
    rv_numpy_tester(MvNormalRV, [0, 1, 2], np.diag([1, 1, 1]))
    # Broadcast cov matrix across independent means?
    # Looks like NumPy doesn't support that (and it's probably better off for
    # it).
    # rv_numpy_tester(MvNormalRV, [[0, 1, 2], [4, 5, 6]], np.diag([1, 1, 1]))
