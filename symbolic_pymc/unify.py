import types
from functools import partial, wraps

import theano.tensor as tt

from multipledispatch import dispatch, MDNotImplementedError

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


def operator_MetaSymbol(x):
    return type(x)


def operator_MetaVariable(x):
    """Get a tuple of the arguments used to construct this meta object.

    This applies a special consideration for Theano `Variable`s: if it has a
    non-`None` `owner` with non-`None` `op` and `inputs`, then the `Variable`
    is more aptly given as the output of `op(inputs)`.

    Otherwise, considering the `Variable` in isolation, it can be constructed
    directly using its `type` constructor.
    """
    x_owner = getattr(x, 'owner', None)
    if x_owner and hasattr(x_owner, 'op'):
        return x_owner.op
    return operator_MetaSymbol(x)


operator.add((MetaSymbol,), operator_MetaSymbol)
operator.add((MetaVariable,), operator_MetaVariable)
operator.add((tt.Variable,), lambda x: operator(MetaVariable.from_obj(x)))


def arguments_MetaSymbol(x):
    return tuple(x.rands())


def arguments_MetaVariable(x):
    """Get a tuple of the arguments used to construct this meta object.

    This applies a special consideration for Theano `Variable`s: if it has a
    non-`None` `owner` with non-`None` `op` and `inputs`, then the `Variable`
    is more aptly given as the output of `op(inputs)`.

    Otherwise, considering the `Variable` in isolation, it can be constructed
    directly using its `type` constructor.
    """
    # Get an apply node, if any
    x_owner = getattr(x, 'owner', None)
    if x_owner and hasattr(x_owner, 'op'):
        return x_owner.inputs
    return arguments_MetaSymbol(x)


arguments.add((MetaSymbol,), arguments_MetaSymbol)
arguments.add((MetaVariable,), arguments_MetaVariable)
arguments.add((tt.Variable,), lambda x: arguments(MetaVariable.from_obj(x)))

# Enable [re]construction of terms
term.add((tt.Op, (list, tuple)),
         lambda op, args: term(MetaOp.from_obj(op), args))
term.add((MetaOp, (list, tuple)), lambda op, args: op(*args))

# Function application for tuples/lists starting with a function or partial
term.add(((types.FunctionType, partial),
          (tuple, list)), lambda op, args: op(*args))


class ExpressionTuple(tuple):
    """A tuple object that represents an expression.

    This object carries the underlying object, if any, and preserves
    it through limited forms of concatenation/cons-ing.
    """

    @property
    def eval_obj(self):
        """Return the evaluation of this expression tuple.

        XXX: If the object isn't cached, it will be evaluated recursively.
        """
        if hasattr(self, '_eval_obj'):
            return self._eval_obj
        else:
            self._eval_obj = self[0](
                *[getattr(i, 'eval_obj', i) for i in self[1:]])
            return self._eval_obj

    @eval_obj.setter
    def eval_obj(self, obj):
        raise ValueError('Value of evaluated expression cannot be set!')

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
        if res == getattr(self, 'orig_expr', None):
            return self.orig_expr
        return res

    def __radd__(self, x):
        return type(self)(x) + self

    def __str__(self):
        return f'e{super().__repr__()}'

    def __repr__(self):
        return f'ExpressionTuple({super().__repr__()})'


def etuple(*args, **kwargs):
    """Create an expression tuple from the arguments.

    If the keyword 'eval_obj' is given, the `ExpressionTuple`'s evaluated
    object is set to the corresponding value.
    """
    res = ExpressionTuple(args)

    if 'eval_obj' in kwargs:
        res._eval_obj = kwargs.pop('eval_obj')

    return res


def _term_ExpressionTuple(rand, rators):
    res = (rand,) + rators
    return res.eval_obj


term.add(((object, MetaOp, types.FunctionType, partial), ExpressionTuple),
         _term_ExpressionTuple)
term.add((tt.Op, ExpressionTuple),
         lambda op, args: term(MetaOp.from_obj(op), args))


@dispatch(object)
def tuple_expression(x):
    """Return a tuple of rand and rators that, when evaluated, would
    construct the object; otherwise, return the object itself.
    """
    try:
        # This can throw an `IndexError` if `x` is an empty
        # `list`/`tuple`.
        op = operator(x)
        args = arguments(x)
    except (IndexError, NotImplementedError):
        return x

    assert isinstance(args, (list, tuple))

    res = etuple(op, *tuple(tuple_expression(a) for a in args), eval_obj=x)
    return res


@dispatch(tt_class_abstractions)
def tuple_expression(x):
    return tuple_expression(mt(x))


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

__all__ = ['debug_unify', 'reify_all_terms', 'etuple', 'tuple_expression']
