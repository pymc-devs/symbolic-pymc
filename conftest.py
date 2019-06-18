import pytest

@pytest.fixture()
def run_with_theano():
    from symbolic_pymc.theano.meta import load_dispatcher

    load_dispatcher()


@pytest.fixture()
def run_with_tensorflow():
    from symbolic_pymc.tensorflow.meta import load_dispatcher

    load_dispatcher()

    # Let's make sure we have a clean graph slate
    from tensorflow.compat.v1 import reset_default_graph
    reset_default_graph()
