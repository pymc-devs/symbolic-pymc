import theano

from symbolic_pymc.theano.utils import is_random_variable
from symbolic_pymc.theano.random_variables import NormalRV


@theano.change_flags(compute_test_value="ignore", cxx="")
def test_is_random_variable():

    X_rv = NormalRV(0, 1)
    res = is_random_variable(X_rv)
    assert res == (X_rv, X_rv)

    def scan_fn():
        Y_t = NormalRV(0, 1, name="Y_t")
        return Y_t

    Y_rv, scan_updates = theano.scan(
        fn=scan_fn,
        outputs_info=[{}],
        n_steps=10,
    )

    res = is_random_variable(Y_rv)
    assert res == (Y_rv, Y_rv.owner.op.outputs[0])
