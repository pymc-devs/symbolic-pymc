from unification import var

from kanren import eq, run

from symbolic_pymc.relations import lconj, lconj_seq, ldisj, ldisj_seq, conde, concat


def test_concat():
    q = var()
    assert run(0, q, concat("a", "b", q)) == ("ab",)
    assert not run(0, q, concat("a", "b", "bc"))
    assert not run(0, q, concat(1, "b", "bc"))
    assert run(0, q, concat(q, "b", "bc")) == (q,)


def test_lconj_basics():

    res = list(lconj(eq(1, var("a")), eq(2, var("b")))({}))
    assert res == [{var("a"): 1, var("b"): 2}]

    res = list(lconj(eq(1, var("a")))({}))
    assert res == [{var("a"): 1}]

    res = list(lconj_seq([])({}))
    assert res == []

    res = list(lconj(eq(1, var("a")), eq(2, var("a")))({}))
    assert res == []

    res = list(lconj(eq(1, 2))({}))
    assert res == []

    res = list(lconj(eq(1, 1))({}))
    assert res == [{}]


def test_ldisj_basics():

    res = list(ldisj(eq(1, var("a")))({}))
    assert res == [{var("a"): 1}]

    res = list(ldisj(eq(1, 2))({}))
    assert res == []

    res = list(ldisj(eq(1, 1))({}))
    assert res == [{}]

    res = list(ldisj(eq(1, var("a")), eq(1, var("a")))({}))
    assert res == [{var("a"): 1}, {var("a"): 1}]

    res = list(ldisj(eq(1, var("a")), eq(2, var("a")))({}))
    assert res == [{var("a"): 1}, {var("a"): 2}]

    res = list(ldisj_seq([])({}))
    assert res == []


def test_conde_basics():

    res = list(conde([eq(1, var("a")), eq(2, var("b"))], [eq(1, var("b")), eq(2, var("a"))])({}))
    assert res == [{var("a"): 1, var("b"): 2}, {var("b"): 1, var("a"): 2}]

    res = list(conde([eq(1, var("a")), eq(2, 1)], [eq(1, var("b")), eq(2, var("a"))])({}))
    assert res == [{var("b"): 1, var("a"): 2}]

    res = list(
        conde(
            [eq(1, var("a")), conde([eq(11, var("aa"))], [eq(12, var("ab"))])],
            [
                eq(1, var("b")),
                conde([eq(111, var("ba")), eq(112, var("bb"))], [eq(121, var("bc"))]),
            ],
        )({})
    )
    assert res == [
        {var("a"): 1, var("aa"): 11},
        {var("b"): 1, var("ba"): 111, var("bb"): 112},
        {var("a"): 1, var("ab"): 12},
        {var("b"): 1, var("bc"): 121},
    ]

    res = list(conde([eq(1, 2)], [eq(1, 1)])({}))
    assert res == [{}]

    assert list(lconj(eq(1, 1))({})) == [{}]

    res = list(lconj(conde([eq(1, 2)], [eq(1, 1)]))({}))
    assert res == [{}]

    res = list(lconj(conde([eq(1, 2)], [eq(1, 1)]), conde([eq(1, 2)], [eq(1, 1)]))({}))
    assert res == [{}]
