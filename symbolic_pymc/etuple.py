import inspect
import reprlib

import toolz

from collections import Sequence

from cons.core import ConsPair, ConsNull

from multipledispatch import dispatch

from kanren.term import operator, arguments


etuple_repr = reprlib.Repr()
etuple_repr.maxstring = 100
etuple_repr.maxother = 100


class KwdPair(object):
    """A class used to indicate a keyword + value mapping.

    TODO: Could subclass `ast.keyword`.

    """

    __slots__ = ("arg", "value")

    def __init__(self, arg, value):
        assert isinstance(arg, str)
        self.arg = arg
        self.value = value

    @property
    def eval_obj(self):
        return KwdPair(self.arg, getattr(self.value, "eval_obj", self.value))

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.arg)}, {repr(self.value)})"

    def __str__(self):
        return f"{self.arg}={self.value}"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class ExpressionTuple(Sequence):
    """A tuple-like object that represents an expression.

    This object caches the return value resulting from evaluation of the
    expression it represents.  Likewise, it holds onto the "parent" expression
    from which it was derived (e.g. as a slice), if any, so that it can
    preserve the return value through limited forms of concatenation/cons-ing
    that would reproduce the parent expression.

    TODO: Should probably use weakrefs for that.
    """

    __slots__ = ("_eval_obj", "_tuple", "_orig_expr")
    null = object()

    def __new__(cls, seq=None, **kwargs):

        # XXX: This doesn't actually remove the entry from the kwargs
        # passed to __init__!
        # It does, however, remove it for the check below.
        kwargs.pop("eval_obj", None)

        if seq is None and not kwargs and isinstance(seq, cls):
            return seq

        res = super().__new__(cls)

        return res

    def __init__(self, seq=None, **kwargs):
        """Create an expression tuple.

        If the keyword 'eval_obj' is given, the `ExpressionTuple`'s
        evaluated object is set to the corresponding value.
        XXX: There is no verification/check that the arguments evaluate to the
        user-specified 'eval_obj', so be careful.
        """

        _eval_obj = kwargs.pop("eval_obj", self.null)
        etuple_kwargs = tuple(KwdPair(k, v) for k, v in kwargs.items())

        if seq:
            self._tuple = tuple(seq) + etuple_kwargs
        else:
            self._tuple = etuple_kwargs

        # TODO: Consider making these a weakrefs.
        self._eval_obj = _eval_obj
        self._orig_expr = None

    @property
    def eval_obj(self):
        """Return the evaluation of this expression tuple.

        Warning: If the evaluation value isn't cached, it will be evaluated
        recursively.

        """
        if self._eval_obj is not self.null:
            return self._eval_obj
        else:
            evaled_args = [getattr(i, "eval_obj", i) for i in self._tuple[1:]]
            arg_grps = toolz.groupby(lambda x: isinstance(x, KwdPair), evaled_args)
            evaled_args = arg_grps.get(False, [])
            evaled_kwargs = arg_grps.get(True, [])

            op = self._tuple[0]
            op = getattr(op, "eval_obj", op)

            try:
                op_sig = inspect.signature(op)
            except ValueError:
                # This handles some builtin function types
                _eval_obj = op(*(evaled_args + [kw.value for kw in evaled_kwargs]))
            else:
                op_args = op_sig.bind(*evaled_args, **{kw.arg: kw.value for kw in evaled_kwargs})
                op_args.apply_defaults()

                _eval_obj = op(*op_args.args, **op_args.kwargs)

            # assert not isinstance(_eval_obj, ExpressionTuple)

            self._eval_obj = _eval_obj
            return self._eval_obj

    @eval_obj.setter
    def eval_obj(self, obj):
        raise ValueError("Value of evaluated expression cannot be set!")

    def __add__(self, x):
        res = self._tuple + x
        if self._orig_expr is not None and res == self._orig_expr._tuple:
            return self._orig_expr
        return type(self)(res)

    def __contains__(self, *args):
        return self._tuple.__contains__(*args)

    def __ge__(self, *args):
        return self._tuple.__ge__(*args)

    def __getitem__(self, key):
        tuple_res = self._tuple[key]
        if isinstance(key, slice) and isinstance(tuple_res, tuple):
            tuple_res = type(self)(tuple_res)
            tuple_res._orig_expr = self
        return tuple_res

    def __gt__(self, *args):
        return self._tuple.__gt__(*args)

    def __iter__(self, *args):
        return self._tuple.__iter__(*args)

    def __le__(self, *args):
        return self._tuple.__le__(*args)

    def __len__(self, *args):
        return self._tuple.__len__(*args)

    def __lt__(self, *args):
        return self._tuple.__lt__(*args)

    def __mul__(self, *args):
        return self._tuple.__mul__(*args)

    def __rmul__(self, *args):
        return self._tuple.__rmul__(*args)

    def __radd__(self, x):
        res = x + self._tuple  # type(self)(x + self._tuple)
        if self._orig_expr is not None and res == self._orig_expr._tuple:
            return self._orig_expr
        return type(self)(res)

    def __str__(self):
        return f"e({', '.join(tuple(str(i) for i in self._tuple))})"

    def __repr__(self):
        return f"ExpressionTuple({etuple_repr.repr(self._tuple)})"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"e(...)")
        else:
            with p.group(2, "e(", ")"):
                p.breakable(sep="")
                for idx, item in enumerate(self._tuple):
                    if idx:
                        p.text(",")
                        p.breakable()
                    p.pretty(item)

    def __eq__(self, other):
        return self._tuple == other

    def __hash__(self):
        return hash(self._tuple)


def etuple(*args, **kwargs):
    """Create an ExpressionTuple from the argument list.

    In other words:
        etuple(1, 2, 3) == ExpressionTuple((1, 2, 3))

    """
    return ExpressionTuple(args, **kwargs)


@dispatch(object)
def etuplize(x, shallow=False, return_bad_args=False):
    """Return an expression-tuple for an object (i.e. a tuple of rand and rators).

    When evaluated, the rand and rators should [re-]construct the object.  When
    the object cannot be given such a form, it is simply converted to an
    `ExpressionTuple` and returned.

    Parameters
    ----------
    x: object
      Object to convert to expression-tuple form.
    shallow: bool
      Whether or not to do a shallow conversion.
    return_bad_args: bool
      Return the passed argument when its type is not appropriate, instead
      of raising an exception.

    """
    if isinstance(x, ExpressionTuple):
        return x
    elif x is not None and isinstance(x, (ConsNull, ConsPair)):
        return etuple(*x)

    try:
        # This can throw an `IndexError` if `x` is an empty
        # `list`/`tuple`.
        op = operator(x)
        args = arguments(x)
    except (IndexError, NotImplementedError):
        op = None
        args = None

    if not callable(op) or not isinstance(args, (ConsNull, ConsPair)):
        if return_bad_args:
            return x
        else:
            raise TypeError(f"x is neither a non-str Sequence nor term: {type(x)}")

    if shallow:
        et_op = op
        et_args = args
    else:
        et_op = etuplize(op, return_bad_args=True)
        et_args = tuple(etuplize(a, return_bad_args=True) for a in args)

    res = etuple(et_op, *et_args, eval_obj=x)
    return res


__all__ = ["etuple", "etuplize"]
