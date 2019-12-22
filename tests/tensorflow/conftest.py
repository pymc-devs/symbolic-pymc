import pytest


@pytest.fixture(autouse=True)
def setup_module():
    import symbolic_pymc.meta
    from symbolic_pymc.meta import base_metatize

    import symbolic_pymc.tensorflow.meta as tm

    _metatize = tm.load_dispatcher()

    symbolic_pymc.meta._metatize = _metatize

    # Let's make sure we have a clean graph slate
    from tensorflow.compat.v1 import reset_default_graph

    reset_default_graph()

    yield

    symbolic_pymc.meta._metatize = base_metatize
