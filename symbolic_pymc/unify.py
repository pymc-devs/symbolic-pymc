import numpy as np

from functools import wraps
from operator import itemgetter

from cons.core import _cdr

from kanren.term import term, operator, arguments

from unification.more import unify
from unification.core import reify, _unify, _reify, Var, walk, assoc, isvar

from .meta import MetaSymbol, MetaVariable
from .utils import _check_eq

from .etuple import etuple, ExpressionTuple


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


_base_unify = unify.dispatch(object, object, dict)


def unify_numpy(u, v, s):
    """Handle NumPy arrays in a special way to avoid warnings/exceptions."""
    u = walk(u, s)
    v = walk(v, s)

    if u is v:
        return s
    if isvar(u):
        return assoc(s, u, v)
    if isvar(v):
        return assoc(s, v, u)

    if isinstance(u, np.ndarray) or isinstance(v, np.ndarray):
        if np.array_equal(v, u):
            return s
    elif u == v:
        return s

    return _unify(u, v, s)


unify.add((object, object, dict), unify_numpy)


def unify_MetaSymbol(u, v, s):
    if type(u) != type(v):
        return False
    if hasattr(u, "__all_props__"):
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


_unify.add((MetaSymbol, MetaSymbol, dict), unify_MetaSymbol)

_tuple__unify = _unify.dispatch(tuple, tuple, dict)

_unify.add(
    (ExpressionTuple, (tuple, ExpressionTuple), dict), lambda x, y, s: _tuple__unify(x, y, s)
)
_unify.add((tuple, ExpressionTuple, dict), lambda x, y, s: _tuple__unify(x, y, s))


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

_tuple__reify = _reify.dispatch(tuple, dict)

_reify.add((ExpressionTuple, dict), lambda x, s: _tuple__reify(x, s))


_isvar = isvar.dispatch(object)

isvar.add((MetaSymbol,), lambda x: _isvar(x) or (not isinstance(x.obj, Var) and isvar(x.obj)))


# We don't want to lose special functionality (and caching) because `cdr` uses
# `islice`.
_cdr.add((ExpressionTuple,), itemgetter(slice(1, None)))


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
operator.add((ExpressionTuple,), itemgetter(0))


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
arguments.add((ExpressionTuple,), itemgetter(slice(1, None)))


def _term_ExpressionTuple(rand, rators):
    res = (rand,) + rators
    return res.eval_obj


term.add((object, ExpressionTuple), _term_ExpressionTuple)


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
