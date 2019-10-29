import pytest


@pytest.fixture()
def run_with_theano():
    # import symbolic_pymc.meta

    # from symbolic_pymc.meta import base_metatize

    import symbolic_pymc.theano.meta as tm

    tm.load_dispatcher()

    # yield

    # symbolic_pymc.meta._metatize = base_metatize
