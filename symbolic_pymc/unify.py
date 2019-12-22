from functools import wraps
from operator import itemgetter

from cons.core import _cdr

from kanren.term import term, operator, arguments

from unification.more import unify
from unification.variable import Var
from unification.core import reify, _unify, _reify, isvar

from .meta import MetaSymbol, MetaVariable

from .etuple import etuple, ExpressionTuple

from .constraints import KanrenState


class UnificationFailure(Exception):
    pass


def debug_unify(enable=True):  # pragma: no cover
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
    if getattr(u, "__all_props__", False):
        s = unify(
            [getattr(u, slot) for slot in u.__all_props__],
            [getattr(v, slot) for slot in v.__all_props__],
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


_unify.add((MetaSymbol, MetaSymbol, (KanrenState, dict)), unify_MetaSymbol)

_unify.add(
    (ExpressionTuple, (tuple, ExpressionTuple), (KanrenState, dict)),
    lambda x, y, s: _unify.dispatch(tuple, tuple, type(s))(x, y, s),
)
_unify.add(
    (tuple, ExpressionTuple, (KanrenState, dict)),
    lambda x, y, s: _unify.dispatch(tuple, tuple, type(s))(x, y, s),
)


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

    try:
        rands = o.rands
    except NotImplementedError:
        return o

    new_rands = reify(rands, s)

    if rands == new_rands:
        return o
    else:
        newobj = type(o)(*new_rands, obj=obj)
        return newobj


_reify.add((MetaSymbol, (KanrenState, dict)), _reify_MetaSymbol)

_reify.add(
    (ExpressionTuple, (KanrenState, dict)), lambda x, s: _reify.dispatch(tuple, type(s))(x, s)
)


# _isvar = isvar.dispatch(object)
#
# isvar.add((MetaSymbol,), lambda x: _isvar(x) or (not isinstance(x.obj, Var) and isvar(x.obj)))


# We don't want to lose special functionality (and caching) because `cdr` uses
# `islice`.
_cdr.add((ExpressionTuple,), itemgetter(slice(1, None)))


def operator_MetaSymbol(x):
    """Return the operator/head/CAR of a meta symbol."""
    return type(x)


def operator_MetaVariable(x):
    """Return the operator/head/CAR of a meta variable."""
    return x.base_operator


operator.add((MetaSymbol,), operator_MetaSymbol)
operator.add((MetaVariable,), operator_MetaVariable)
operator.add((ExpressionTuple,), itemgetter(0))


def arguments_MetaSymbol(x):
    """Return the arguments/tail/CDR of a meta symbol.

    We build the full `etuple` for the argument, then return the
    `cdr`/tail, so that the original object is retained when/if the
    original object is later reconstructed and evaluated (e.g. using
    `term`).

    """
    x_e = etuple(type(x), *x.rands, eval_obj=x)
    return x_e[1:]


def arguments_MetaVariable(x):
    """Return the arguments/tail/CDR of a variable object.

    See `arguments_MetaSymbol`
    """
    x_op = x.base_operator
    if x_op is not None:
        x_e = etuple(x_op, *x.base_arguments, eval_obj=x)
        return x_e[1:]


arguments.add((MetaSymbol,), arguments_MetaSymbol)
arguments.add((MetaVariable,), arguments_MetaVariable)
arguments.add((ExpressionTuple,), itemgetter(slice(1, None)))


def _term_ExpressionTuple(rand, rators):
    res = (rand,) + rators
    return res.eval_obj


term.add((object, ExpressionTuple), _term_ExpressionTuple)


@_reify.register(ExpressionTuple, (KanrenState, dict))
def _reify_ExpressionTuple(t, s):
    """When `kanren` reifies `etuple`s, we don't want them to turn into regular `tuple`s.

    We also don't want to lose the expression tracking/caching
    information.

    """
    res = tuple(reify(iter(t), s))
    t_chg = tuple(a == b for a, b in zip(t, res) if not isvar(a) and not isvar(b))

    if all(t_chg):
        if len(t_chg) == len(t):
            # Nothing changed/updated; return the original `etuple`.
            return t

        if hasattr(t, "_orig_expr"):
            # Everything is equal and/or there are some non-logic variables in
            # the result.  Keep tracking the original expression information,
            # in case the original expression is reproduced.
            res = etuple(*res)
            res._orig_expr = t._orig_expr
            return res

    res = etuple(*res)
    return res


__all__ = ["debug_unify"]
