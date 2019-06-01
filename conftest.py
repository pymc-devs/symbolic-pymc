import pytest

@pytest.fixture()
def run_with_theano():
    from symbolic_pymc.theano.meta import load_dispatcher

    load_dispatcher()


@pytest.fixture()
def run_with_tensorflow():
    from symbolic_pymc.tensorflow.meta import load_dispatcher

    load_dispatcher()
