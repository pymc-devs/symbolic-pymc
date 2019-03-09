import reprlib

from functools import wraps

from multipledispatch import dispatch

from kanren import isvar
from kanren.term import term, operator, arguments

from unification.more import unify
from unification.core import reify, _unify, _reify, Var

from .meta import MetaSymbol, MetaVariable
from .utils import _check_eq

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
        return f"e{etuple_repr.repr(tuple(self))}"

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


class UnificationFailure(Exception):
    pass


def debug_unify(enable=True):
    """Wrap unify functions so that they raise a `UnificationFailure` exception when unification fails."""
    if enable:

        def set_debug(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                s = f(*args, **kwargs)
                if s is False:
                    import pdb

                    pdb.set_trace()
                    raise UnificationFailure()
                return s

            return wrapper

        _unify.funcs = {
            sig: set_debug(getattr(f, "__wrapped__", f)) for sig, f in _unify.funcs.items()
        }
        _unify._cache.clear()
    else:
        _unify.funcs = {sig: getattr(f, "__wrapped__", f) for sig, f in _unify.funcs.items()}
        _unify._cache.clear()


def unify_MetaSymbol(u, v, s):
    if type(u) != type(v):
        return False
    if hasattr(u, "__slots__"):
        s = unify(
            [getattr(u, slot) for slot in u.__slots__],
            [getattr(v, slot) for slot in v.__slots__],
            s,
        )
    elif u != v:
        return False
    if s:
        # If these two meta objects unified, and one has a logic
        # variable as its base object, consider the unknown base
        # object unified by the other's base object (if any).
        # This way, the original base objects can be recovered during
        # reification (preserving base object equality and such).
        if isinstance(u.obj, Var) and v.obj:
            s[u.obj] = v.obj
        elif isinstance(v.obj, Var) and u.obj:
            s[v.obj] = u.obj
    return s


_unify.add((MetaSymbol, MetaSymbol, dict), unify_MetaSymbol)


def _reify_MetaSymbol(o, s):
    if isinstance(o.obj, Var):
        # We allow reification of the base object field for
        # a meta object.
        # TODO: This is a weird thing that we should probably reconsider.
        # It's part of the functionality that allows base objects to fill-in
        # as logic variables, though.
        obj = s.get(o.obj, o.obj)
    else:
        # Otherwise, if there's a base object, it should indicate that there
        # are no logic variables or meta terms.
        # TODO: Seems like we should be able to skip the reify and comparison
        # below.
        obj = None

    rands = o.rands()
    new_rands = reify(rands, s)

    if rands == new_rands:
        return o
    else:
        newobj = type(o)(*new_rands, obj=obj)
        return newobj


_reify.add((MetaSymbol, dict), _reify_MetaSymbol)


_isvar = isvar.dispatch(object)

isvar.add((MetaSymbol,), lambda x: _isvar(x) or (not isinstance(x.obj, Var) and isvar(x.obj)))


def operator_MetaSymbol(x):
    return type(x)


def operator_MetaVariable(x):
    """Get a tuple of the arguments used to construct this meta object.

    This function applies a special consideration for Theano `Variable`s:

    If the `Variable` has a non-`None` `owner` attribute, then we use the
    `owner`'s `op` and `inputs` to construct an expression-tuple that should
    (when evaluated) produce said `Variable`.

    Otherwise, we generate an expression-tuple for the graph representing the
    output of whatever `Apply` node (plus `Op`) would've produced said
    `Variable`.

    Graphically, the complete expressions resulting from the former and the
    later, respectively, are as follows:

      a + b => etuple(mt.add, a, b)

      a + b => etuple(mt.TensorVariable,
                      type,
                      mt.Apply(op=mt.add, inputs=[a, b]),
                      index

    XXX: To achieve the more succinct former representation, we assume that
    `owner.op(owner.inputs)` is consistent, of course.

    """
    x_op = x.operator
    if x_op is not None:
        return x_op
    return operator_MetaSymbol(x)


operator.add((MetaSymbol,), operator_MetaSymbol)
operator.add((MetaVariable,), operator_MetaVariable)


def arguments_MetaSymbol(x):
    """Get a tuple of the arguments used to construct this meta object.

    In other words, produce an expression-tuple/`etuple`'s `cdr`/tail.

    We build the full `etuple` for the argument, then return the
    `cdr`/tail, so that the original object is retained when/if the
    original object is later reconstructed and evaluated (e.g. using
    `term`).

    """
    x_e = etuple(type(x), *x.rands(), eval_obj=x)
    return x_e[1:]


def arguments_MetaVariable(x):
    """Get a tuple of the arguments used to construct this meta object.

    See the special considerations for `TensorVariable`s described in
    `operator_MetaVariable`.

    """
    x_op = x.operator
    if x_op is not None:
        x_e = etuple(x_op, *x.inputs, eval_obj=x)
        return x_e[1:]

    return arguments_MetaSymbol(x)


arguments.add((MetaSymbol,), arguments_MetaSymbol)
arguments.add((MetaVariable,), arguments_MetaVariable)


def _term_ExpressionTuple(rand, rators):
    res = (rand,) + rators
    return res.eval_obj


term.add((object, ExpressionTuple), _term_ExpressionTuple)


@dispatch(object)
def tuple_expression(x):
    """Return an expression-tuple for an object (i.e. a tuple of rand and rators).

    When evaluated, the rand and rators should [re-]construct the object.  When the
    object cannot be given such a form, the object itself is returned.

    NOTE: `tuple_expression(...)[2:]` and `arguments(...)` will *not* return
    the same thing, because the former is recursive and the latter is not.
    In other words, this S-expression-like "decomposition" is recursive, and,
    as such, it requires an inside-out evaluation to re-construct a
    "decomposed" object.  In contrast, `operator` and `arguments` is a shallow
    "decomposition".

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

    res = etuple(op, *tuple(tuple_expression(a) for a in args), eval_obj=x)
    return res


@_reify.register(ExpressionTuple, dict)
def _reify_ExpressionTuple(t, s):
    """When `kanren` reifies `etuple`s, we don't want them to turn into regular `tuple`s.

    We also don't want to lose the expression tracking/caching
    information.

    """
    res = tuple(reify(iter(t), s))
    t_chg = [_check_eq(a, b) for a, b in zip(t, res) if not isvar(a) and not isvar(b)]

    if all(t_chg):
        if len(t_chg) == len(t):
            # Nothing changed/updated; return the original `etuple`.
            return t

        if hasattr(t, "orig_expr"):
            # Everything is equal and/or there are some non-logic variables in
            # the result.  Keep tracking the original expression information,
            # in case the original expression is reproduced.
            res = etuple(*res)
            res.orig_expr = t.orig_expr
            return res

    res = etuple(*res)
    return res


__all__ = ["debug_unify", "etuple", "tuple_expression"]
