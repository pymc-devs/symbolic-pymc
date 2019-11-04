from unification import isvar

from kanren import eq
from kanren.core import lallgreedy
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


def buildo(op, args, obj):
    if not isvar(obj):
        if not isvar(args):
            args = etuplize(args, shallow=True)
        oop, oargs = operator(obj), arguments(obj)
        return lallgreedy(eq(op, oop), eq(args, oargs))
    elif isvar(args) or isvar(op):
        return conso(op, args, obj)
    else:
        return eq(obj, term(op, args))
