from itertools import permutations

from unification import var, unify
from unification.core import _reify

from kanren import run, eq

from symbolic_pymc.relations import lconj
from symbolic_pymc.constraints import KanrenState, Disequality, neq, ConstrainedVar


def test_kanrenstate():

    ks = KanrenState()

    assert repr(ks) == 'KanrenState({}, OrderedDict())'

    assert ks == {}
    assert {} == ks
    assert not ks == {var('a'): 1}
    assert not ks == KanrenState({var('a'): 1})

    assert unify(1, 1, ks) is not None
    assert unify(1, 2, ks) is False

    assert unify(var('b'), var('a'), ks)
    assert unify(var('a'), var('b'), ks)
    assert unify(var('a'), var('b'), ks)

    # Now, try that with a constraint (that's never used).
    ks.add_constraint(Disequality({var('a'): {1}}))

    assert ks == {}
    assert {} == ks
    assert not ks == {var('a'): 1}
    assert not ks == KanrenState({var('a'): 1})

    assert unify(1, 1, ks) is not None
    assert unify(1, 2, ks) is False

    assert unify(var('b'), var('a'), ks)
    assert unify(var('a'), var('b'), ks)
    assert unify(var('a'), var('b'), ks)


def test_reify():
    ks = KanrenState()
    assert repr(ConstrainedVar(var('a'), ks)) == '~a: {}'

    de = Disequality({var('a'): {1, 2}})
    ks.add_constraint(de)

    assert repr(de) == '~a =/= {1, 2}'
    assert de.constraints_str(var()) == ""
    assert repr(ConstrainedVar(var('a'), ks)) == '~a: {=/= {1, 2}}'

    # TODO: Make this work with `reify` when `var('a')` isn't in `ks`.
    assert isinstance(_reify(var('a'), ks), ConstrainedVar)
    assert repr(_reify(var('a'), ks)) == '~a: {=/= {1, 2}}'


def test_disequality():

    ks = KanrenState()
    de = Disequality({var('a'): {1}})
    ks.add_constraint(de)

    assert unify(var('a'), 1, ks) is False

    ks = unify(var('a'), var('b'), ks)
    assert unify(var('b'), 1, ks) is False

    res = list(lconj(neq(var('a'), 1))({}))
    assert len(res) == 1
    assert isinstance(res[0], KanrenState)
    assert res[0].constraints[Disequality].mappings[var('a')] == {1}

    res = list(lconj(neq(var('a'), 1), eq(var('a'), 2))({}))
    assert len(res) == 1
    assert isinstance(res[0], KanrenState)
    assert res[0].constraints[Disequality].mappings[var('a')] == {1}
    assert res[0][var('a')] == 2

    res = list(lconj(eq(var('a'), 1), neq(var('a'), 1))({}))
    assert res == []

    goal_sets = [([neq(var('a'), 1)], 1),
                 ([neq(var('a'), 1), eq(var('a'), 1)], 0),
                 ([neq(var('a'), 1), eq(var('b'), 1), eq(var('a'), var('b'))], 0)]

    for goal, results in goal_sets:
        # The order of goals should not matter, so try them all
        for goal_ord in permutations(goal):

            res = list(lconj(*goal_ord)({}))
            assert len(res) == results

            res = list(lconj(*goal_ord)(KanrenState()))
            assert len(res) == results

            assert len(run(0, var('q'), *goal_ord)) == results
