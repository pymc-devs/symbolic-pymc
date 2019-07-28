from itertools import tee, chain
from functools import reduce

from toolz import interleave

from unification import isvar, reify

from kanren import eq
from kanren.core import goaleval
from kanren.facts import Relation
from kanren.goals import goalify
from kanren.term import term, operator, arguments
from kanren.goals import conso

from ..etuple import etuplize


# Hierarchical models that we recognize.
hierarchical_model = Relation("hierarchical")

# Conjugate relationships
conjugate = Relation("conjugate")


concat = goalify(lambda *args: "".join(args))


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


def buildo(op, args, obj):
    """Construct a goal that relates an object and its rand + rators decomposition.

    This version uses etuples.

    """

    def buildo_goal(S):
        nonlocal op, args, obj

        op, args, obj = reify((op, args, obj), S)

        if not isvar(obj):

            if not isvar(args):
                args = etuplize(args, shallow=True)

            oop, oargs = operator(obj), arguments(obj)

            yield from lall(eq(op, oop), eq(args, oargs))(S)

        elif isvar(args) or isvar(op):
            yield from conso(op, args, obj)(S)
        else:
            yield from eq(obj, term(op, args))(S)

    return buildo_goal
