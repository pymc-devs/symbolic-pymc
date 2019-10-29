import pytest


@pytest.fixture(autouse=True)
def setup_module():

    import symbolic_pymc.meta
    from symbolic_pymc.meta import base_metatize

    import symbolic_pymc.theano.meta as tm

    _metatize = tm.load_dispatcher()

    symbolic_pymc.meta._metatize = _metatize

    yield

    symbolic_pymc.meta._metatize = base_metatize
