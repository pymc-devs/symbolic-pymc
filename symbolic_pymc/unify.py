from functools import wraps
from collections.abc import Mapping

from cons.core import _car, _cdr, ConsError

from kanren.term import arguments, operator

from unification.variable import Var
from unification.core import _reify, _unify, reify, unify

from etuples import etuple

from .meta import MetaSymbol, MetaVariable


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


_unify.add((MetaSymbol, MetaSymbol, Mapping), unify_MetaSymbol)


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


_reify.add((MetaSymbol, Mapping), _reify_MetaSymbol)


# def car_MetaSymbol(x):
#     """Return the operator/head/CAR of a meta symbol."""
#     return type(x)


def car_MetaVariable(x):
    """Return the operator/head/CAR of a meta variable."""
    try:
        return x.base_operator
    except NotImplementedError:
        raise ConsError("Not a cons pair.")


# _car.add((MetaSymbol,), car_MetaSymbol)
_car.add((MetaVariable,), car_MetaVariable)

# operator.add((MetaSymbol,), car_MetaSymbol)
operator.add((MetaVariable,), car_MetaVariable)


# def cdr_MetaSymbol(x):
#     """Return the arguments/tail/CDR of a meta symbol.
#
#     We build the full `etuple` for the argument, then return the
#     `cdr`/tail, so that the original object is retained when/if the
#     original object is later reconstructed and evaluated (e.g. using
#     `term`).
#
#     """
#     try:
#         x_e = etuple(_car(x), *x.rands, eval_obj=x)
#     except NotImplementedError:
#         raise ConsError("Not a cons pair.")
#
#     return x_e[1:]


def cdr_MetaVariable(x):
    """Return the arguments/tail/CDR of a variable object.

    See `cdr_MetaSymbol`
    """
    try:
        x_e = etuple(_car(x), *x.base_arguments, eval_obj=x)
    except NotImplementedError:
        raise ConsError("Not a cons pair.")

    return x_e[1:]


# _cdr.add((MetaSymbol,), cdr_MetaSymbol)
_cdr.add((MetaVariable,), cdr_MetaVariable)

# arguments.add((MetaSymbol,), cdr_MetaSymbol)
arguments.add((MetaVariable,), cdr_MetaVariable)


__all__ = ["debug_unify"]
