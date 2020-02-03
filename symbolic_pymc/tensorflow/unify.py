import tensorflow as tf

from collections.abc import Mapping

from kanren.term import operator, arguments

from unification.core import _reify, _unify, reify

from cons.core import _car, _cdr

from etuples import etuplize

from .meta import TFlowMetaSymbol
from ..meta import metatize
from ..unify import unify_MetaSymbol

tf_class_abstractions = tuple(c.base for c in TFlowMetaSymbol.base_subclasses())

_unify.add(
    (TFlowMetaSymbol, tf_class_abstractions, Mapping),
    lambda u, v, s: unify_MetaSymbol(u, metatize(v), s),
)
_unify.add(
    (tf_class_abstractions, TFlowMetaSymbol, Mapping),
    lambda u, v, s: unify_MetaSymbol(metatize(u), v, s),
)
_unify.add(
    (tf_class_abstractions, tf_class_abstractions, Mapping),
    lambda u, v, s: unify_MetaSymbol(metatize(u), metatize(v), s),
)


def _reify_TFlowClasses(o, s):
    meta_obj = metatize(o)
    return reify(meta_obj, s)


_reify.add((tf_class_abstractions, Mapping), _reify_TFlowClasses)


_car.add((tf.Tensor,), lambda x: operator(metatize(x)))
operator.add((tf.Tensor,), lambda x: operator(metatize(x)))

_cdr.add((tf.Tensor,), lambda x: arguments(metatize(x)))
arguments.add((tf.Tensor,), lambda x: arguments(metatize(x)))

etuplize.add(tf_class_abstractions, lambda x, shallow=False: etuplize(metatize(x), shallow))


__all__ = []
