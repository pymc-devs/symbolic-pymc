import reprlib

from multipledispatch import dispatch

from kanren.term import operator, arguments


etuple_repr = reprlib.Repr()
etuple_repr.maxstring = 100
etuple_repr.maxother = 100


class ExpressionTuple(tuple):
    """A tuple object that represents an expression.

    This object carries the underlying object, if any, and preserves it
    through limited forms of concatenation/cons-ing.

    """

    @property
    def eval_obj(self):
        """Return the evaluation of this expression tuple.

        XXX: If the object isn't cached, it will be evaluated recursively.

        """
        if hasattr(self, "_eval_obj"):
            return self._eval_obj
        else:
            evaled_args = [getattr(i, "eval_obj", i) for i in self[1:]]
            _eval_obj = self[0](*evaled_args)

            assert not isinstance(_eval_obj, ExpressionTuple)

            self._eval_obj = _eval_obj
            return self._eval_obj

    @eval_obj.setter
    def eval_obj(self, obj):
        raise ValueError("Value of evaluated expression cannot be set!")

    def __getitem__(self, key):
        # if isinstance(key, slice):
        #     return [self.list[i] for i in xrange(key.start, key.stop, key.step)]
        # return self.list[key]
        tuple_res = super().__getitem__(key)
        if isinstance(key, slice) and isinstance(tuple_res, tuple):
            tuple_res = type(self)(tuple_res)
            tuple_res.orig_expr = self
        return tuple_res

    def __add__(self, x):
        res = type(self)(super().__add__(x))
        if res == getattr(self, "orig_expr", None):
            return self.orig_expr
        return res

    def __radd__(self, x):
        res = type(self)(x + tuple(self))
        if res == getattr(self, "orig_expr", None):
            return self.orig_expr
        return res

    def __str__(self):
        return f"e({', '.join(tuple(str(i) for i in self))})"

    def __repr__(self):
        return f"ExpressionTuple({etuple_repr.repr(tuple(self))})"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self.__class__.__name__}(...)")
        else:
            with p.group(2, f"{self.__class__.__name__}((", "))"):
                p.breakable()
                for idx, item in enumerate(self):
                    if idx:
                        p.text(",")
                        p.breakable()
                    p.pretty(item)


def etuple(*args, **kwargs):
    """Create an expression tuple from the arguments.

    If the keyword 'eval_obj' is given, the `ExpressionTuple`'s
    evaluated object is set to the corresponding value.

    """
    res = ExpressionTuple(args)

    if "eval_obj" in kwargs:
        _eval_obj = kwargs.pop("eval_obj")

        assert not isinstance(_eval_obj, ExpressionTuple)

        res._eval_obj = _eval_obj

    return res


@dispatch(object)
def etuplize(x, shallow=False):
    """Return an expression-tuple for an object (i.e. a tuple of rand and rators).

    When evaluated, the rand and rators should [re-]construct the object.  When the
    object cannot be given such a form, the object itself is returned.

    NOTE: `etuplize(...)[2:]` and `arguments(...)` will *not* return
    the same thing by default, because the former is recursive and the latter
    is not.  In other words, this S-expression-like "decomposition" is
    recursive, and, as such, it requires an inside-out evaluation to
    re-construct a "decomposed" object.  In contrast, `operator` and
    `arguments` is necessarily a shallow "decomposition".

    Parameters
    ----------
    x: object
      Object to convert to expression-tuple form.
    shallow: bool
      Whether or not to do a shallow conversion.

    """
    if isinstance(x, ExpressionTuple):
        return x

    try:
        # This can throw an `IndexError` if `x` is an empty
        # `list`/`tuple`.
        op = operator(x)
        args = arguments(x)
    except (IndexError, NotImplementedError):
        return x

    assert isinstance(args, (list, tuple))

    # Not everything in a list/tuple should be considered an expression.
    if not callable(op):
        return x

    if shallow:
        et_args = args
    else:
        et_args = tuple(etuplize(a) for a in args)

    res = etuple(op, *et_args, eval_obj=x)
    return res


__all__ = ["etuple", "etuplize"]
