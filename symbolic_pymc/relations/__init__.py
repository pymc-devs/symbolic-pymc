from itertools import tee, chain
from functools import reduce

from toolz import interleave

from kanren.core import goaleval
from kanren.facts import Relation

from unification import unify, reify, Var


# Hierarchical models that we recognize.
hierarchical_model = Relation("hierarchical")

# Conjugate relationships
conjugate = Relation("conjugate")


def concat(a, b, out):
    """Construct a non-relational string concatenation goal."""

    def concat_goal(S):
        nonlocal a, b, out

        a_rf, b_rf, out_rf = reify((a, b, out), S)

        if isinstance(a_rf, str) and isinstance(b_rf, str):
            S_new = unify(out_rf, a_rf + b_rf, S)

            if S_new is not False:
                yield S_new
                return
        elif isinstance(a_rf, (Var, str)) and isinstance(b_rf, (Var, str)):
            yield S

    return concat_goal


def ldisj_seq(goals):
    """Produce a goal that returns the appended state stream from all successful goal arguments.

    In other words, it behaves like logical disjunction/OR for goals.
    """

    def ldisj_seq_goal(S):
        nonlocal goals

        goals, _goals = tee(goals)

        yield from interleave(goaleval(g)(S) for g in _goals)

    return ldisj_seq_goal


def lconj_seq(goals):
    """Produce a goal that returns the appended state stream in which all goals are necessarily successful.

    In other words, it behaves like logical conjunction/AND for goals.
    """

    def lconj_seq_goal(S):
        nonlocal goals

        goals, _goals = tee(goals)

        g0 = next(iter(_goals), None)

        if g0 is None:
            return

        z0 = goaleval(g0)(S)

        yield from reduce(lambda z, g: chain.from_iterable(map(goaleval(g), z)), _goals, z0)

    return lconj_seq_goal


def ldisj(*goals):
    return ldisj_seq(goals)


def lconj(*goals):
    return lconj_seq(goals)


def conde(*goals):
    return ldisj_seq(lconj_seq(g) for g in goals)


lall = lconj
lany = ldisj
