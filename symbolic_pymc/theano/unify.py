import theano.tensor as tt

from collections.abc import Mapping

from kanren.term import term, operator, arguments

from unification.core import _reify, _unify, reify

from cons.core import _car, _cdr

from etuples import etuplize
from etuples.core import ExpressionTuple

from .meta import TheanoMetaSymbol
from ..meta import metatize
from ..unify import unify_MetaSymbol


tt_class_abstractions = tuple(c.base for c in TheanoMetaSymbol.base_subclasses())

_unify.add(
    (TheanoMetaSymbol, tt_class_abstractions, Mapping),
    lambda u, v, s: unify_MetaSymbol(u, metatize(v), s),
)
_unify.add(
    (tt_class_abstractions, TheanoMetaSymbol, Mapping),
    lambda u, v, s: unify_MetaSymbol(metatize(u), v, s),
)
_unify.add(
    (tt_class_abstractions, tt_class_abstractions, Mapping),
    lambda u, v, s: unify_MetaSymbol(metatize(u), metatize(v), s),
)


def _reify_TheanoClasses(o, s):
    meta_obj = metatize(o)
    return reify(meta_obj, s)


_reify.add((tt_class_abstractions, Mapping), _reify_TheanoClasses)

operator.add((tt.Variable,), lambda x: operator(metatize(x)))
_car.add((tt.Variable,), lambda x: operator(metatize(x)))

arguments.add((tt.Variable,), lambda x: arguments(metatize(x)))
_cdr.add((tt.Variable,), lambda x: arguments(metatize(x)))

term.add((tt.Op, ExpressionTuple), lambda op, args: term(metatize(op), args))

etuplize.add(tt_class_abstractions, lambda x, shallow=False: etuplize(metatize(x), shallow))

__all__ = []
