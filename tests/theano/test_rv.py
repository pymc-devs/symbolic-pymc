import numpy as np

import theano.tensor as tt

from theano.gof.op import get_test_value
from theano.gof.graph import inputs as tt_inputs

from pytest import importorskip

from symbolic_pymc.theano.opt import FunctionGraph
from symbolic_pymc.theano.random_variables import NormalRV, MvNormalRV, PolyaGammaRV, DirichletRV

from tests.theano import requires_test_values


def rv_numpy_tester(rv, *params, size=None):
    """Test for correspondence between `RandomVariable` and NumPy shape and
    broadcast dimensions.
    """
    tt.config.compute_test_value = "ignore"

    test_rv = rv(*params, size=size)
    param_vals = [get_test_value(p) for p in params]
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


@requires_test_values
def test_Normal_infer_shape():
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 3
    sd_tt = tt.scalar("sd")
    sd_tt.tag.test_value = 1.0

    test_params = [
        ([tt.as_tensor_variable(1.0), sd_tt], None),
        ([tt.as_tensor_variable(1.0), sd_tt], (M_tt,)),
        ([tt.as_tensor_variable(1.0), sd_tt], (2, M_tt)),
        ([tt.zeros((M_tt,)), sd_tt], None),
        ([tt.zeros((M_tt,)), sd_tt], (M_tt,)),
        ([tt.zeros((M_tt,)), sd_tt], (2, M_tt)),
        ([tt.zeros((M_tt,)), tt.ones((M_tt,))], None),
        ([tt.zeros((M_tt,)), tt.ones((M_tt,))], (2, M_tt)),
    ]
    for args, size in test_params:
        rv = NormalRV(*args, size=size)
        rv_shape = tuple(NormalRV._infer_shape(size or (), args, None))
        assert tuple(get_test_value(rv_shape)) == tuple(get_test_value(rv).shape)


@requires_test_values
def test_Normal_ShapeFeature():
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 3
    sd_tt = tt.scalar("sd")
    sd_tt.tag.test_value = 1.0

    d_rv = NormalRV(tt.ones((M_tt,)), sd_tt, size=(2, M_tt))
    d_rv.tag.test_value

    fg = FunctionGraph(
        [i for i in tt_inputs([d_rv]) if not isinstance(i, tt.Constant)],
        [d_rv],
        clone=True,
        features=[tt.opt.ShapeFeature()],
    )
    s1, s2 = fg.shape_feature.shape_of[fg.memo[d_rv]]

    assert get_test_value(s1) == get_test_value(d_rv).shape[0]
    assert get_test_value(s2) == get_test_value(d_rv).shape[1]


@requires_test_values
def test_normalrv_vs_numpy():
    rv_numpy_tester(NormalRV, 0.0, 1.0)
    rv_numpy_tester(NormalRV, 0.0, 1.0, size=[3])
    # Broadcast sd over independent means...
    rv_numpy_tester(NormalRV, [0.0, 1.0, 2.0], 1.0)
    rv_numpy_tester(NormalRV, [0.0, 1.0, 2.0], 1.0, size=[3, 3])
    rv_numpy_tester(NormalRV, [0], [1], size=[1])
    rv_numpy_tester(NormalRV, tt.as_tensor_variable([0]), [1], size=[1])
    rv_numpy_tester(NormalRV, tt.as_tensor_variable([0]), [1], size=tt.as_tensor_variable([1]))


@requires_test_values
def test_mvnormalrv_vs_numpy():
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


@requires_test_values
def test_mvnormalrv_ShapeFeature():
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 2

    d_rv = MvNormalRV(tt.ones((M_tt,)), tt.eye(M_tt), size=2)

    fg = FunctionGraph(
        [i for i in tt_inputs([d_rv]) if not isinstance(i, tt.Constant)],
        [d_rv],
        clone=True,
        features=[tt.opt.ShapeFeature()],
    )

    s1, s2 = fg.shape_feature.shape_of[fg.memo[d_rv]]

    assert s1.eval() == 2
    assert fg.memo[M_tt] in tt_inputs([s2])


def test_polyagammarv_vs_PolyaGammaRV():

    _ = importorskip("pypolyagamma")

    # Sampled values should be scalars
    pg_rv = PolyaGammaRV(1.1, -10.5)
    assert pg_rv.eval().shape == ()

    pg_rv = PolyaGammaRV(1.1, -10.5, size=[1])
    assert pg_rv.eval().shape == (1,)

    pg_rv = PolyaGammaRV(1.1, -10.5, size=[2, 3])
    bcast_smpl = pg_rv.eval()
    assert bcast_smpl.shape == (2, 3)
    # Make sure they're not all equal
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)

    pg_rv = PolyaGammaRV(np.r_[1.1, 3], -10.5)
    bcast_smpl = pg_rv.eval()
    assert bcast_smpl.shape == (2,)
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)

    pg_rv = PolyaGammaRV(np.r_[1.1, 3], -10.5, size=(2, 3))
    bcast_smpl = pg_rv.eval()
    assert bcast_smpl.shape == (2, 2, 3)
    assert np.all(np.abs(np.diff(bcast_smpl.flat)) > 0.0)


def test_dirichletrv_samples():

    import theano

    theano.config.cxx = ""
    theano.config.mode = "FAST_COMPILE"

    alphas = np.c_[[100, 1, 1], [1, 100, 1], [1, 1, 100]]

    res = DirichletRV(alphas).eval()
    assert np.all(np.diag(res) >= res)

    res = DirichletRV(alphas, size=2).eval()
    assert res.shape == (2, 3, 3)
    assert all(np.all(np.diag(r) >= r) for r in res)

    for i in range(alphas.shape[0]):
        res = DirichletRV(alphas[i]).eval()
        assert np.all(res[i] > np.delete(res, [i]))

        res = DirichletRV(alphas[i], size=2).eval()
        assert res.shape == (2, 3)
        assert all(np.all(r[i] > np.delete(r, [i])) for r in res)


@requires_test_values
def test_dirichlet_infer_shape():
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 3

    test_params = [
        ([tt.ones((M_tt,))], None),
        ([tt.ones((M_tt,))], (M_tt + 1,)),
        ([tt.ones((M_tt,))], (2, M_tt)),
        ([tt.ones((M_tt, M_tt + 1))], None),
        ([tt.ones((M_tt, M_tt + 1))], (M_tt + 2,)),
        ([tt.ones((M_tt, M_tt + 1))], (2, M_tt + 2, M_tt + 3)),
    ]
    for args, size in test_params:
        rv = DirichletRV(*args, size=size)
        rv_shape = tuple(DirichletRV._infer_shape(size or (), args, None))
        assert tuple(get_test_value(rv_shape)) == tuple(get_test_value(rv).shape)


def test_dirichlet_ShapeFeature():
    """Make sure `RandomVariable.infer_shape` works with `ShapeFeature`."""
    M_tt = tt.iscalar("M")
    M_tt.tag.test_value = 2
    N_tt = tt.iscalar("N")
    N_tt.tag.test_value = 3

    d_rv = DirichletRV(tt.ones((M_tt, N_tt)), name="Gamma")

    fg = FunctionGraph(
        [i for i in tt_inputs([d_rv]) if not isinstance(i, tt.Constant)],
        [d_rv],
        clone=True,
        features=[tt.opt.ShapeFeature()],
    )

    s1, s2 = fg.shape_feature.shape_of[fg.memo[d_rv]]

    assert fg.memo[M_tt] in tt_inputs([s1])
    assert fg.memo[N_tt] in tt_inputs([s2])
