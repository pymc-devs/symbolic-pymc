import abc
import types
import reprlib

import numpy as np

from itertools import chain
from functools import partial
from collections.abc import Iterator, Mapping

from unification import isvar, Var

from .utils import _check_eq

from multipledispatch import dispatch

meta_repr = reprlib.Repr()
meta_repr.maxstring = 100
meta_repr.maxother = 100
meta_repr.print_obj = False


def metatize(obj):
    """Convert object to base type then meta object."""
    if isvar(obj):
        return obj
    return _metatize(obj)


@dispatch((set, list, tuple))
def _metatize(obj):
    """Convert elements of an iterable to meta objects."""
    return type(obj)([metatize(o) for o in obj])


@dispatch(Iterator)
def _metatize(obj):
    """Convert elements of an iterator to meta objects."""
    return iter([metatize(o) for o in obj])


def _make_hashable(x):
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, Mapping):
        return frozenset(x.items())
    elif isinstance(x, np.ndarray):
        return x.tostring()
    else:
        return x


def _meta_reify_iter(rands):
    """Recursively reify an iterable object and return a boolean indicating the presence of un-reifiable objects, if any."""
    # We want as many of the rands reified as possible,
    any_unreified = False
    reified_rands = []
    if isinstance(rands, Mapping):
        _rands = rands.items()
    else:
        _rands = rands

    for s in _rands:
        if isinstance(s, MetaSymbol):
            rrand = s.reify()
            reified_rands.append(rrand)
            any_unreified |= isinstance(rrand, MetaSymbol)
            any_unreified |= isvar(rrand)
        elif MetaSymbol.is_meta(s):
            reified_rands.append(s)
            any_unreified |= True
        elif isinstance(s, (list, tuple)):
            _reified_rands, _any_unreified = _meta_reify_iter(s)
            reified_rands.append(type(s)(_reified_rands))
            any_unreified |= _any_unreified
        else:
            reified_rands += [s]

    return type(rands)(reified_rands), any_unreified


class MetaSymbolType(abc.ABCMeta):
    def __new__(cls, name, bases, clsdict):

        # We need to track the cumulative slots, because subclasses can define
        # their own--yet we'll need to track changes across all of them.
        all_slots = set(
            chain.from_iterable(s.__all_slots__ for s in bases if hasattr(s, "__all_slots__"))
        )
        all_slots |= set(clsdict.get("__slots__", []))
        clsdict["__all_slots__"] = all_slots

        def __setattr__(self, attr, obj):
            """If a slot value is changed, discard any associated non-meta/base objects."""
            if attr == "obj":
                if isinstance(obj, MetaSymbol):
                    raise ValueError("base object cannot be a meta object!")
            elif (
                getattr(self, "obj", None) is not None
                and not isinstance(self.obj, Var)
                and attr in getattr(self, "__all_slots__", {})
                and hasattr(self, attr)
                and getattr(self, attr) != obj
            ):
                self.obj = None

            object.__setattr__(self, attr, obj)

        clsdict["__setattr__"] = __setattr__

        @classmethod
        def __metatize(cls, obj):
            return cls(*[getattr(obj, s) for s in getattr(cls, "__slots__", [])], obj=obj)

        clsdict.setdefault("_metatize", __metatize)

        new_cls = super().__new__(cls, name, bases, clsdict)

        if isinstance(new_cls.base, type):
            _metatize.add((new_cls.base,), new_cls._metatize)

        # TODO: Could register base classes.
        # E.g. cls.register(bases)
        return new_cls


class MetaSymbol(metaclass=MetaSymbolType):
    """Meta objects for unification and such.

    TODO: Should `MetaSymbol.obj` be an abstract property and a `weakref`?
    """

    @property
    @abc.abstractmethod
    def base(self):
        """Return the underlying (e.g. a theano/tensorflow) base type/rator for this meta object."""
        raise NotImplementedError()

    @classmethod
    def base_classes(cls, mro_order=True):
        res = tuple(c.base for c in cls.__subclasses__())
        if hasattr(cls, "base"):
            res = (cls.base,) + res
        sorted(res, key=lambda c: len(c.mro()), reverse=mro_order)
        return res

    @classmethod
    def is_meta(cls, obj):
        return isinstance(obj, MetaSymbol) or isvar(obj)

    def __init__(self, obj=None):
        self.obj = obj

    def rands(self):
        """Create a tuple of the meta object's operator parameters (i.e. "rands")."""
        return tuple(getattr(self, s) for s in getattr(self, "__slots__", []))

    def reify(self):
        """Create a concrete base object from this meta object (and its rands)."""
        if self.obj is not None and not isinstance(self.obj, Var):
            return self.obj
        else:
            reified_rands, any_unreified = _meta_reify_iter(self.rands())

            # If not all the rands reified, then create another meta
            # object--albeit one with potentially more non-`None` `obj` fields.
            rator = self.base if not any_unreified else type(self)
            res = rator(*reified_rands)

            if not any_unreified:
                self.obj = res

            return res

    def __eq__(self, other):
        """Implement an equivalence between meta objects and their bases."""
        if self is other:
            return True

        if not (type(self) == type(other)):
            return False

        if not (self.base == other.base):
            return False

        a_slots = getattr(self, "__slots__", None)
        if a_slots is not None:
            if not all(_check_eq(getattr(self, attr), getattr(other, attr)) for attr in a_slots):
                return False
        elif getattr(other, "__slots__", None) is not None:
            # The other object has slots, but this one doesn't.
            return False
        else:
            # Neither have slots, so best we can do is compare
            # base objects (if any).
            # If there aren't base objects, we say they're not equal.
            # (Maybe we should *require* base objects in this case
            # and raise an exception?)
            return getattr(self, "obj", None) == getattr(other, "obj", None) is not None

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if getattr(self, "__slots__", None) is not None:
            rands = tuple(_make_hashable(p) for p in self.rands())
            return hash(rands + (self.base,))
        else:
            return hash((self.base, self.obj))

    def __str__(self):
        obj = getattr(self, "obj", None)
        if obj is None:
            res = self.__repr__()
        else:
            res = f"{self.__class__.__name__}({str(obj)})"
        return res

    def __repr__(self):
        obj = getattr(self, "obj", None)
        args = meta_repr.repr(self.rands()).strip("()")
        if args:
            args += ", "
        args += f"obj={meta_repr.repr(obj)}"
        return "{}({})".format(self.__class__.__name__, args)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self.__class__.__name__}(...)")
        else:
            with p.group(2, f"{self.__class__.__name__}(", ")"):
                p.breakable()
                idx = None
                if hasattr(self, "__slots__"):
                    for idx, (name, item) in enumerate(zip(self.__slots__, self.rands())):
                        if idx:
                            p.text(",")
                            p.breakable()
                        p.text(name)
                        p.text("=")
                        p.pretty(item)

                obj = getattr(self, "obj", None)

                if obj is not None and meta_repr.print_obj:
                    if idx is not None:
                        p.text(",")
                        p.breakable()
                    p.text("obj=")
                    p.pretty(obj)


@dispatch((MetaSymbol, type(None), types.FunctionType, partial, str, dict))
def _metatize(obj):
    return obj


class MetaOp(MetaSymbol):
    """A meta object that represents a `MetaVariable`-producing operator.

    Also, make sure to override `Op.out_meta_type` and make it return the
    expected meta variable type, if it isn't the default: `MetaTensorVariable`.

    In some cases, operators hold their own inputs and outputs
    (e.g. TensorFlow), and, in others, an intermediary "application" node holds
    that information.  This class leaves those details up to the
    implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def obj(self):
        return object.__getattribute__(self, "_obj")

    @obj.setter
    def obj(self, x):
        if hasattr(self, "_obj"):
            raise ValueError("Cannot reset obj in an `Op`")
        object.__setattr__(self, "_obj", x)

    @abc.abstractmethod
    def out_meta_types(self, inputs=None):
        """Return the types of meta variables this `Op` is expected to produce given the inputs."""
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, *args, ttype=None, index=None, **kwargs):
        raise NotImplementedError()


class MetaVariable(MetaSymbol):
    @property
    @abc.abstractmethod
    def operator(self):
        """Return a meta object representing an operator, if any, capable of producing this variable.

        It should be callable with all inputs necessary to reproduce this
        tensor given by `MetaVariable.inputs`.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def inputs(self):
        """Return the inputs necessary for `MetaVariable.operator` to produced this variable, if any."""
        raise NotImplementedError()


def _find_meta_type(obj_type, meta_abs_type):
    cls = meta_abs_type
    obj_cls = None
    while True:
        try:
            obj_cls = next(filter(lambda t: issubclass(obj_type, t.base), cls.__subclasses__()))
        except StopIteration:
            # The current class is the best fit.
            if cls.base == obj_type:
                return cls

            # The abstract meta type has no subclasses that match the given
            # object type.
            if obj_cls is None:
                return None

            # This object is a subclass of an existing meta class' base type,
            # but there is no implemented meta type for this subclass, so we
            # dynamically make one.
            new_type = type(f"Meta{obj_type.__name__}", (obj_cls,), {"base": obj_type})
            return new_type(obj_type)
        else:
            cls = obj_cls


@dispatch(type)
def _metatize(obj_type):
    """Return an existing meta type/class, or create a new one."""
    for meta_type in MetaSymbol.__subclasses__():
        obj_cls = _find_meta_type(obj_type, meta_type)

        if obj_cls is not None:
            return obj_cls
