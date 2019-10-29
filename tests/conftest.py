import pytest

from copy import deepcopy


@pytest.fixture(autouse=True)
def setup_module():
    import symbolic_pymc.meta

    from symbolic_pymc.meta import base_metatize

    _old_metatize = symbolic_pymc.meta._metatize
    symbolic_pymc.meta._metatize = deepcopy(base_metatize)

    yield

    symbolic_pymc.meta._metatize = _old_metatize
