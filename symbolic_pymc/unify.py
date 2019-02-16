import types
from functools import partial, wraps

import theano.tensor as tt

from kanren import isvar
from kanren.term import term, operator, arguments
from kanren.facts import fact
from kanren.assoccomm import commutative, associative

from unification.more import unify
from unification.core import reify, _unify, _reify, Var

from .meta import MetaSymbol, MetaVariable, MetaOp, mt

tt_class_abstractions = tuple(c.base for c in MetaSymbol.__subclasses__())


class UnificationFailure(Exception):
    pass


def debug_unify(enable=True):
    """Wrap unify functions so that they raise a `UnificationFailure` exception
    when unification fails.
    """
    if enable:
        def set_debug(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                s = f(*args, **kwargs)
                if s is False:
                    import pdb; pdb.set_trace()
                    raise UnificationFailure()
                return s
            return wrapper

        _unify.funcs = {sig: set_debug(getattr(f, '__wrapped__', f))
                        for sig, f in _unify.funcs.items()}
        _unify._cache.clear()
    else:
        _unify.funcs = {sig: getattr(f, '__wrapped__', f)
                        for sig, f in _unify.funcs.items()}
        _unify._cache.clear()


def unify_MetaSymbol(u, v, s):
    if type(u) != type(v):
        return False
    if hasattr(u, '__slots__'):
        s = unify([getattr(u, slot) for slot in u.__slots__],
                  [getattr(v, slot) for slot in v.__slots__],
                  s)
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
_unify.add((MetaSymbol, tt_class_abstractions, dict),
           lambda u, v, s: unify_MetaSymbol(u, MetaSymbol.from_obj(v), s))
_unify.add((tt_class_abstractions, MetaSymbol, dict),
           lambda u, v, s: unify_MetaSymbol(MetaSymbol.from_obj(u), v, s))
_unify.add((tt_class_abstractions, tt_class_abstractions, dict),
           lambda u, v, s: unify_MetaSymbol(MetaSymbol.from_obj(u),
                                            MetaSymbol.from_obj(v), s))


def _reify_MetaSymbol(o, s):
    if isinstance(o.obj, Var):
        obj = s.get(o.obj, o.obj)
    else:
        obj = None

    rands = o.rands()
    new_rands = reify(rands, s)

    if rands == new_rands:
        return o
    else:
        newobj = type(o)(*new_rands, obj=obj)
        return newobj


_reify.add((MetaSymbol, dict), _reify_MetaSymbol)


def _reify_TheanoClasses(o, s):
    meta_obj = MetaSymbol.from_obj(o)
    return reify(meta_obj, s)


_reify.add((tt_class_abstractions, dict), _reify_TheanoClasses)


_isvar = isvar.dispatch(object)

isvar.add((MetaSymbol,), lambda x: _isvar(x) or (not isinstance(x.obj, Var)
                                                 and isvar(x.obj)))


def operator_MetaVariable(x):
    # Get an apply node, if any
    x_owner = getattr(x, 'owner', None)
    if x_owner and hasattr(x_owner, 'op'):
        return x_owner.op
    return None


operator.add((MetaVariable,), operator_MetaVariable)
operator.add((tt.Variable,), lambda x: operator(MetaVariable.from_obj(x)))


def arguments_MetaVariable(x):
    # Get an apply node, if any
    x_owner = getattr(x, 'owner', None)
    if x_owner and hasattr(x_owner, 'op'):
        return x_owner.inputs
    return None


arguments.add((MetaVariable,), arguments_MetaVariable)
arguments.add((tt.Variable,), lambda x: arguments(MetaVariable.from_obj(x)))

# Enable [re]construction of terms
term.add((tt.Op, (list, tuple)), lambda op, args: term(MetaOp.from_obj(op), args))
term.add((MetaOp, (list, tuple)), lambda op, args: op(*args))

# Function application for tuples/lists starting with a function or partial
term.add(((types.FunctionType, partial), (tuple, list)), lambda op, args: op(*args))


def reify_all_terms(obj, s=None):
    """Recursively reifies all terms tuples/lists with some awareness
    for meta objects."""
    try:
        if isinstance(obj, MetaSymbol):
            # Avoid using `operator`/`arguments` and unnecessarily
            # breaking apart meta objects and the base objects they
            # hold onto (i.e. their reified forms).
            res = obj.reify()
            if not MetaSymbol.is_meta(res):
                return res
        op, args = operator(obj), arguments(obj)
        op = reify_all_terms(op, s)
        args = reify_all_terms(args, s)
        return term(op, args)
    except (IndexError, NotImplementedError):
        return reify(obj, s or {})


fact(commutative, mt.add)
fact(commutative, mt.mul)
fact(associative, mt.add)
fact(associative, mt.mul)

__all__ = []
