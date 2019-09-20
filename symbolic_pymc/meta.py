import abc
import types
import reprlib

from itertools import chain
from functools import partial
from collections import OrderedDict
from collections.abc import Iterator, Mapping

from unification import isvar, Var

from .utils import _check_eq

from multipledispatch import dispatch

from cachetools import cached

meta_repr = reprlib.Repr()
meta_repr.maxstring = 100
meta_repr.maxother = 100
meta_repr.print_obj = False

metatize_cache = {}


def metatize(obj):
    """Convert object to base type then meta object."""
    if isvar(obj):
        return obj
    return _metatize(obj)


@dispatch((type(None), types.FunctionType, partial, str, dict))
def _metatize(obj):
    return obj


@_metatize.register((set, tuple))
@cached(metatize_cache)
def _metatize_set_tuple(obj):
    """Convert elements of an iterable to meta objects."""
    return type(obj)([metatize(o) for o in obj])


@_metatize.register(list)
def _metatize_list(obj):
    """Convert elements of an iterable to meta objects."""
    return type(obj)([metatize(o) for o in obj])


@_metatize.register(Iterator)
@cached(metatize_cache)
def _metatize_Iterator(obj):
    """Convert elements of an iterator to meta objects."""
    return iter([metatize(o) for o in obj])


def meta_reify_iter(rands):
    """Recursively reify an iterable object and return a boolean indicating the presence of un-reifiable objects, if any."""
    any_unreified = False
    reified_rands = []

    _rands = rands
    if isinstance(_rands, Mapping):
        _rands = _rands.items()

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
            _reified_rands, _any_unreified = meta_reify_iter(s)
            reified_rands.append(type(s)(_reified_rands))
            any_unreified |= _any_unreified
        else:
            reified_rands.append(s)

    return type(rands)(reified_rands), any_unreified


class MetaSymbolType(abc.ABCMeta):
    def __new__(cls, name, bases, clsdict):

        # We need to track the cumulative slots, because subclasses can define
        # their own--yet we'll need to track changes across all of them.
        slots = clsdict.get("__slots__", ())
        all_slots = tuple(
            OrderedDict.fromkeys(
                chain(
                    chain.from_iterable(
                        tuple(s.__all_slots__) for s in bases if hasattr(s, "__all_slots__")
                    ),
                    tuple(slots),
                )
            )
        )

        clsdict["__all_slots__"] = all_slots
        clsdict["__all_props__"] = tuple(s for s in all_slots if not s.startswith("_"))
        clsdict["__volatile_slots__"] = tuple(s for s in all_slots if s.startswith("_"))
        clsdict["__props__"] = tuple(s for s in slots if not s.startswith("_"))

        if clsdict["__volatile_slots__"]:

            def __setattr__(self, attr, obj):
                """If a slot value is changed, reset cached slots."""

                # Underscored-prefixed/volatile/stateful slots can be set
                # without affecting other such slots.
                if (
                    attr not in self.__volatile_slots__
                    # Are we trying to set a custom property?
                    and attr in getattr(self, "__all_props__", ())
                    # Is it a custom property that's already been set?
                    and hasattr(self, attr)
                    # Are we setting it to a new value?
                    and getattr(self, attr) is not obj
                ):
                    for s in self.__volatile_slots__:
                        object.__setattr__(self, s, None)

                object.__setattr__(self, attr, obj)

            clsdict["__setattr__"] = __setattr__

        @classmethod
        def __metatize(cls, obj):
            """Metatize using the `__all_props__` property."""
            return cls(*tuple(getattr(obj, s) for s in getattr(cls, "__all_props__", ())), obj=obj)

        clsdict.setdefault("_metatize", __metatize)

        new_cls = super().__new__(cls, name, bases, clsdict)

        if isinstance(new_cls.base, type):
            _metatize.add((new_cls.base,), new_cls._metatize)

        # Wrap the class implementation of `__hash__` with this value-caching
        # code.
        if "_hash" in clsdict["__volatile_slots__"]:
            _orig_hash = new_cls.__hash__
            new_cls._orig_hash = _orig_hash

            def _cached_hash(self):
                if getattr(self, "_hash", None) is not None:
                    return self._hash

                object.__setattr__(self, "_hash", _orig_hash(self))

                return self._hash

            new_cls.__hash__ = _cached_hash

        return new_cls


class MetaSymbol(metaclass=MetaSymbolType):
    """Meta objects for unification and such.

    TODO: Should `MetaSymbol.obj` be an abstract property and a `weakref`?
    """

    __slots__ = ("_obj", "_hash", "_rands")

    @property
    @abc.abstractmethod
    def base(self):
        """Return the underlying (e.g. a theano/tensorflow) base type/rator for this meta object."""
        raise NotImplementedError()

    @property
    def obj(self):
        return object.__getattribute__(self, "_obj")

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
        assert obj is None or isvar(obj) or isinstance(obj, self.base)
        self._obj = obj

    def rands(self):
        """Get a tuple of the meta object's operator parameters (i.e. "rands")."""
        if getattr(self, "_rands", None) is not None:
            return self._rands

        self._rands = tuple(getattr(self, s) for s in getattr(self, "__all_props__", ()))

        return self._rands

    def reify(self):
        """Attempt to create a concrete base object from this meta object.

        During the process, dependent objects will need to be reified, which
        may result in updates to the object(s) being reified.

        For instance, if a meta tensor's parent operator is fully reifiable to
        a base object, then the meta tensor's dtype and shape may be fixed:
        e.g. a tensor corresponding to the output of a sum of two float64
        scalars is necessarily a float64 scalar.

        This function will set any unspecified properties (e.g. dtype and shape
        values for the previous example), mutating the object in-place when
        possible.  It will return a [refined/partially reified] meta object
        when it can't fully reify to a base object (in which case, it will
        return the base object) or when partial reification results in a meta
        object from a subclass.
        """
        if self.obj is not None and not isinstance(self.obj, Var):
            return self.obj
        else:
            reified_rands, any_unreified = meta_reify_iter(self.rands())

            # If not all the rands reified, then create another meta
            # object--albeit one with potentially more non-`None` `obj` fields.
            rator = self.base if not any_unreified else type(self)
            res = rator(*reified_rands)

            if not any_unreified:
                self._obj = res

            return res

    def __eq__(self, other):
        """Implement an equivalence between meta objects and their bases."""
        if self is other:
            return True

        if not (type(self) == type(other)):
            return False

        assert self.base == other.base

        if self.rands():
            return all(_check_eq(s, o) for s, o in zip(self.rands(), other.rands()))
        else:
            return NotImplemented

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.base, self.rands()))

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
                if hasattr(self, "__all_props__"):
                    for idx, (name, item) in enumerate(zip(self.__all_props__, self.rands())):
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


@_metatize.register(MetaSymbol)
def _metatize_MetaSymbol(obj):
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

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def out_meta_types(self, inputs=None):
        """Return the types of meta variables this `Op` is expected to produce given the inputs."""
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __eq__(self, other):
        res = super().__eq__(other)

        if res is NotImplemented:
            return getattr(self, "obj", None) == getattr(other, "obj", None) is not None

        return res

    def __hash__(self):
        return hash((self.base, self.obj))


class MetaVariable(MetaSymbol):
    __slots__ = ()

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

            # FIXME, TODO: We should do something about `Op` constructor
            # arguments and properties.
            #
            # For instance, `tt.nlinalg.SVD` takes `full_matrices` and `compute_uv`
            # constructor arguments, but the dynamically constructed `TheanoMetaOp` type for
            # SVD is just the base `TheanoMetaOp.__init__`, which doesn't account for those.
            # To do this correctly, we would need to dynamically metatize the underlying
            # `Op`'s `__init__` and so on.
            new_type = type(f"Meta{obj_type.__name__}", (obj_cls,), {"base": obj_type})

            return new_type
        else:
            cls = obj_cls


@_metatize.register(type)
@cached(metatize_cache)
def _metatize_type(obj_type):
    """Return an existing meta type/class, or create a new one."""
    for meta_type in MetaSymbol.__subclasses__():
        obj_cls = _find_meta_type(obj_type, meta_type)

        if obj_cls is not None:
            return obj_cls
