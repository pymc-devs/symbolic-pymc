from unification import var

from kanren import run

from symbolic_pymc.relations import concat


def test_concat():
    q = var()
    assert run(0, q, concat("a", "b", q)) == ("ab",)
    assert not run(0, q, concat("a", "b", "bc"))
    assert not run(0, q, concat(1, "b", "bc"))
    assert run(0, q, concat(q, "b", "bc")) == (q,)
