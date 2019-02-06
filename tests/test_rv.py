import numpy as np

import theano
import theano.tensor as tt

from symbolic_pymc import *

tt.config.on_opt_error = 'raise'
theano.config.mode = 'FAST_COMPILE'
theano.config.cxx = ''


def rv_numpy_test(rv, *params, size=None):
    """Test for correspondence between `RandomVariable` and NumPy shape and
    broadcast dimensions.
    """
    test_rv = rv(*params, size=size)
    param_vals = [tt.gof.op.get_test_value(p) for p in params]
    size_val = None if size is None else tt.gof.op.get_test_value(size)
    test_val = getattr(np.random, rv.name)(*param_vals, size=size_val)
    test_shp = np.shape(test_val)

    # This might be a little too harsh, since purely symbolic `tensor.vector` inputs
    # have no broadcastable information, yet, they can take broadcastable values.
    # E.g.
    #     x_tt = tt.vector('x')
    #     x_tt.tag.test_value = np.array([5]) # non-symbolic value is broadcastable!
    #     x_tt.tag.test_value = np.array([5, 4]) # non-symbolic value is not broadcastable.
    #
    # In the end, there's really no clear way to determine this without full
    # evaluation of a symbolic node, and that mostly defeats the purpose.
    # Unfortunately, this is what PyMC3 resorts to when constructing its
    # `TensorType`s (and shapes).
    test_bcast = [s == 1 for s in test_shp]
    np.testing.assert_array_equal(test_rv.type.broadcastable, test_bcast)

    eval_args = {p: v for p, v in zip(params, param_vals)
                 if isinstance(p, tt.Variable) and not isinstance(p, tt.Constant)}
    np.testing.assert_array_equal(test_rv.shape.eval(eval_args), test_shp)
    np.testing.assert_array_equal(np.shape(test_rv.eval(eval_args)), test_shp)


def test_normalrv():
    rv_numpy_test(NormalRV, 0., 1.)
    rv_numpy_test(NormalRV, 0., 1., size=[3])
    # Broadcast sd over independent means...
    rv_numpy_test(NormalRV, [0., 1., 2.], 1.)
    rv_numpy_test(NormalRV, [0., 1., 2.], 1., size=[3, 3])
    rv_numpy_test(NormalRV, [0], [1], size=[1])

    rv_numpy_test(NormalRV, tt.as_tensor_variable([0]), [1], size=[1])
    rv_numpy_test(NormalRV, tt.as_tensor_variable([0]), [1], size=tt.as_tensor_variable([1]))


# XXX: Shouldn't work due to broadcastable comments in `rv_numpy_test`.
# test_mean = tt.vector('test_mean')
# test_mean.tag.test_value = np.r_[1]
# rv_numpy_test(NormalRV, test_mean, [1], size=tt.as_tensor_variable([1]))

# with pm.Model():
#     test_rv = pm.MvNormal('test_rv', [0], np.diag([1]), shape=1)
#
# test_rv.broadcastable

def test_mvnormalrv():
    rv_numpy_test(MvNormalRV, [0], np.diag([1]))
    rv_numpy_test(MvNormalRV, [0], np.diag([1]), size=[1])
    rv_numpy_test(MvNormalRV, [0], np.diag([1]), size=[4])
    rv_numpy_test(MvNormalRV, [0], np.diag([1]), size=[4, 1])
    rv_numpy_test(MvNormalRV, [0], np.diag([1]), size=[4, 1, 1])
    rv_numpy_test(MvNormalRV, [0], np.diag([1]), size=[1, 5, 8])
    rv_numpy_test(MvNormalRV, [0, 1, 2], np.diag([1, 1, 1]))
    # Broadcast cov matrix across independent means?
    # Looks like NumPy doesn't support that (and are probably better off for it).
    # rv_numpy_test(MvNormalRV, [[0, 1, 2], [4, 5, 6]], np.diag([1, 1, 1]))
