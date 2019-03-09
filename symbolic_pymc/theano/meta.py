import types
import inspect

import weakref
import theano
import theano.tensor as tt

from functools import partial, wraps

from unification import var, isvar, Var
from multipledispatch import dispatch

from kanren.facts import fact
from kanren.assoccomm import commutative, associative

from .ops import RandomVariable
from ..meta import (
    MetaSymbol,
    MetaSymbolType,
    MetaOp,
    MetaVariable,
    _meta_reify_iter,
    metatize,
    _metatize,
)


def _metatize_theano_object(obj):
    try:
        obj = tt.as_tensor_variable(obj)
    except (ValueError, tt.AsTensorError):
        raise ValueError("Could not find a MetaSymbol class for {}".format(obj))
    return _metatize(obj)


def load_dispatcher():
    """Set/override dispatcher to default to TF objects."""
    _metatize.add((object,), _metatize_theano_object)


load_dispatcher()


class TheanoMetaSymbol(MetaSymbol):
    pass


class TheanoMetaType(TheanoMetaSymbol):
    base = theano.Type

    def __call__(self, name=None):
        if self.obj:
            return metatize(self.obj(name=name))
        return metatize(self.base.Variable)(self, name)


class TheanoMetaRandomStateType(TheanoMetaType):
    base = tt.raw_random.RandomStateType


class TheanoMetaTensorType(TheanoMetaType):
    base = tt.TensorType
    __slots__ = ["dtype", "broadcastable", "name"]

    def __init__(self, dtype, broadcastable, name, obj=None):
        super().__init__(obj=obj)
        self.dtype = dtype
        self.broadcastable = broadcastable
        self.name = name


class TheanoMetaOp(MetaOp, TheanoMetaSymbol):
    """A meta object that represents Theano `Op`s.

    NOTE: By default it will use `Op.make_node`'s signature to produce meta
    `Apply` node inputs, so be sure to override that signature when
    `Op.make_node`'s arguments aren't one-to-one with the expected `Apply` node
    inputs.  See `MetaOp.__call__` for more details.

    Also, make sure to override `Op.out_meta_types` and make it return the
    expected meta variable types, if it isn't the default: `TheanoMetaTensorVariable`.
    """

    base = tt.Op

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_sig = inspect.signature(self.obj.make_node)

    def out_meta_types(self, inputs=None):
        """Return the types of meta variables this `Op` is expected to produce given the inputs.

        The default is `TheanoMetaTensorVariable` (corresponding to
        `TheanoTensorVariable` outputs from the base `Op`).
        """
        return (TheanoMetaTensorVariable,)

    def __call__(self, *args, ttype=None, index=None, **kwargs):
        """Emulate `make_node` for this `Op` and return .

        NOTE: Meta objects will use positional arguments and non-"name" keyword
        args as `Apply` node inputs.  Also, if some of the `Op` constructor
        arguments that end up as `Apply` node input arguments are keywords,
        *use the keywords* and not their positions!

        Otherwise, if a base object can't be referenced, unknown Theano types
        and index values will be fill-in with logic variables (that can also be
        specified manually though the keyword arguments `ttype` and `index`).

        Parameters
        ----------
        ttype: object (optional)
            Value to use for an unknown Theano type.  Defaults to a logic
            variable.
        index: object (optional)
            Value to use for an unknown output index value.  Defaults to a
            logic variable.

        """
        name = kwargs.pop("name", None)

        # Use the `Op`'s default `make_node` arguments, if any.
        op_arg_bind = self.op_sig.bind(*args, **kwargs)
        op_arg_bind.apply_defaults()
        op_args, op_args_unreified = _meta_reify_iter(op_arg_bind.args)

        if not op_args_unreified:
            tt_out = self.obj(*op_args)
            res_var = metatize(tt_out)

            if not isinstance(tt_out, (list, tuple)):
                # If the name is indeterminate, we still want all the reified
                # info, but we need to make sure that certain parts aren't
                # known.
                # TODO: In this case, the reified Theano object is a sort of
                # "proxy" object; we should use this approach for dtype, as
                # well.
                # TODO: We should also put this kind of logic in the
                # appropriate places (e.g. `MetaVariable.reify`), when
                # possible.
                if TheanoMetaSymbol.is_meta(name):
                    # This should also invalidate `res_var.obj`.
                    res_var.name = name
                    # Allow the base object to be unified, so that reification
                    # can recover the underlying object--instead of recreating
                    # it and sacrificing equality.
                    res_var.obj = var()

                elif tt_out.name != name:
                    tt_out.name = name
                    res_var.name = name

        else:
            # XXX: It's not always clear how `Op.make_node` arguments map to
            # `Apply` node inputs, which is one of the big problem with
            # Theano's design.  (More generally, it's that `Op`s don't provide
            # a spec for `Apply` node inputs and outputs at all.)

            # Also, `Apply` inputs can't be `None` (they could be
            # `tt.none_type_t()`, though).
            res_apply = TheanoMetaApply(
                self, tuple(filter(lambda x: x is not None, op_arg_bind.args))
            )

            # TODO: Elemwise has an `output_types` method that can be
            # used to infer the output type of this variable.
            ttype = ttype or var()

            # Use the given index or the base `Op`'s `default_output`;
            # otherwise, create a logic variable place-holder.
            index = index if index is not None else getattr(self.obj, "default_output", None)
            index = index if index is not None else var()

            # XXX: We don't have a higher-order meta object model, so being
            # wrong about the exact type of output variable will cause
            # problems.
            out_meta_type, = self.out_meta_types(op_args)
            res_var = out_meta_type(ttype, res_apply, index, name)
            res_var.obj = var()

        return res_var


class TheanoMetaElemwise(TheanoMetaOp):
    base = tt.Elemwise

    def __call__(self, *args, ttype=None, index=None, **kwargs):
        obj_nout = getattr(self.obj, "nfunc_spec", None)
        obj_nout = obj_nout[-1] if obj_nout is not None else None
        if obj_nout == 1 and index is None:
            index = 0
        return super().__call__(*args, ttype=ttype, index=index, **kwargs)


class TheanoMetaDimShuffle(TheanoMetaOp):
    base = tt.DimShuffle
    __slots__ = ["input_broadcastable", "new_order", "inplace"]

    def __init__(self, input_broadcastable, new_order, inplace=True, obj=None):
        super().__init__(obj=obj)
        self.input_broadcastable = input_broadcastable
        self.new_order = new_order
        self.inplace = inplace


class TheanoMetaRandomVariable(TheanoMetaOp):
    base = RandomVariable

    def __init__(self, obj=None):
        super().__init__(obj=obj)
        # The `name` keyword parameter isn't an `Apply` node input, so we need
        # to remove it from the automatically generated signature.
        self.op_sig = self.op_sig.replace(parameters=list(self.op_sig.parameters.values())[0:4])


class TheanoMetaApply(TheanoMetaSymbol):
    base = tt.Apply
    __slots__ = ["op", "inputs"]

    def __init__(self, op, inputs, outputs=None, obj=None):
        super().__init__(obj=obj)
        self.op = metatize(op)
        self.inputs = tuple(metatize(i) for i in inputs)
        self.outputs = outputs
        if outputs is not None:
            # TODO: Convert these to meta objects?
            self.outputs = tuple(weakref.ref(o) for o in outputs)
        else:
            self.outputs = None

    def reify(self):
        if self.obj and not isinstance(self.obj, Var):
            return self.obj
        else:
            tt_op = self.op.reify()
            if not self.is_meta(tt_op):
                reified_rands, any_unreified = _meta_reify_iter(self.inputs)
                if not any_unreified:
                    tt_var = tt_op(*reified_rands)
                    self.obj = tt_var.owner
                    return tt_var.owner
            return self

    @property
    def nin(self):
        return len(self.inputs)

    @property
    def nout(self):
        if self.outputs is not None:
            return len(self.outputs)
        elif self.obj:
            return len(self.obj.outputs)


class TheanoMetaVariable(MetaVariable, TheanoMetaSymbol):
    base = theano.Variable
    __slots__ = ["type", "owner", "index", "name"]

    def __init__(self, type, owner, index, name, obj=None):
        super().__init__(obj=obj)
        self.type = metatize(type)
        if owner is not None:
            self.owner = metatize(owner)
        else:
            self.owner = None
        self.index = index
        self.name = name

    @property
    def operator(self):
        if self.owner is not None:
            return self.owner.op

    @property
    def inputs(self):
        if self.owner is not None:
            return self.owner.inputs

    def reify(self):
        if self.obj and not isinstance(self.obj, Var):
            return self.obj

        if not self.owner:
            return super().reify()

        # Having an `owner` causes issues (e.g. being consistent about
        # other, unrelated outputs of an `Apply` node), and, in this case,
        # the `Apply` node that owns this variable needs to construct it.
        reified_rands, any_unreified = _meta_reify_iter(self.rands())
        tt_apply = self.owner.obj

        if tt_apply and not isvar(tt_apply):
            # If the owning `Apply` reified, then one of its `outputs`
            # corresponds to this variable.  Our `self.index` value should
            # tell us which, but, when that's not available, we can
            # sometimes infer it.
            if tt_apply.nout == 1:
                tt_index = 0
                # Make sure we didn't have a mismatched non-meta index value.
                assert isvar(self.index) or self.index is None or self.index == 0
                # Set/replace `None` or meta value
                self.index = 0
                tt_var = tt_apply.outputs[tt_index]
            elif not self.is_meta(self.index):
                tt_var = tt_apply.outputs[self.index]
            elif self.index is None:
                tt_var = tt_apply.default_output()
                self.index = tt_apply.outputs.index(tt_var)
            else:
                return self
            # If our name value is not set/concrete, then use the reified
            # value's.  Otherwise, use ours.
            if isvar(self.name) or self.name is None:
                self.name = tt_var.name
            else:
                tt_var.name = self.name
            assert tt_var is not None
            self.obj = tt_var
            return tt_var

        return super().reify()


class TheanoMetaTensorVariable(TheanoMetaVariable):
    # TODO: Could extend `theano.tensor.var._tensor_py_operators`, too.
    base = tt.TensorVariable

    @property
    def ndim(self):
        # TODO: Would be cool if we could return
        # a logic variable representing this.
        if isinstance(self.type, TheanoMetaTensorType) and isinstance(
            self.type.broadastable, (list, tuple)
        ):
            return len(self.type.broadcastable)


class TheanoMetaConstant(TheanoMetaVariable):
    base = theano.Constant
    __slots__ = ["type", "data"]

    def __init__(self, type, data, name=None, obj=None):
        super().__init__(type, None, None, name, obj=obj)
        self.data = data


class TheanoMetaTensorConstant(TheanoMetaConstant):
    # TODO: Could extend `theano.tensor.var._tensor_py_operators`, too.
    base = tt.TensorConstant
    __slots__ = ["type", "data", "name"]

    def __init__(self, type, data, name=None, obj=None):
        super().__init__(type, data, name, obj=obj)


class TheanoMetaSharedVariable(TheanoMetaVariable):
    base = tt.sharedvar.SharedVariable
    __slots__ = ["name", "type", "data", "strict"]

    @classmethod
    def _metatize(cls, obj):
        res = TheanoMetaSharedVariable(
            obj.name, obj.type, obj.container.data, obj.container.strict, obj=obj
        )
        return res

    def __init__(self, name, type, data, strict, obj=None):
        super().__init__(type, None, None, name, obj=obj)
        self.data = data
        self.strict = strict


class TheanoMetaTensorSharedVariable(TheanoMetaSharedVariable):
    # TODO: Could extend `theano.tensor.var._tensor_py_operators`, too.
    base = tt.sharedvar.TensorSharedVariable


class TheanoMetaScalarSharedVariable(TheanoMetaSharedVariable):
    base = tt.sharedvar.ScalarSharedVariable


class TheanoMetaAccessor(object):
    """Create an object that can be used to implicitly convert Theano functions and objects into meta objects.

    Use it like a namespace/module/package object, e.g.

    >>> mt = TheanoMetaAccessor()
    >>> mt.vector('a')
    MetaTensorVariable(MetaTensorType('float64', (False,), None,
    obj=TensorType(float64, vector)), None, None, 'a', obj=a)

    Call it as a function to perform direct conversion to a meta
    object, e.g.

    >>> mt(tt.vector('a'))
    MetaTensorVariable(MetaTensorType('float64', (False,), None,
    obj=TensorType(float64, vector)), None, None, 'a', obj=a)

    """

    namespaces = [tt]

    def __init__(self, namespace=None):
        if namespace is None:
            from symbolic_pymc.theano import (  # pylint: disable=import-self
                meta,
                ops,
                random_variables,
            )

            self.namespaces += [meta, ops, random_variables]
        else:
            self.namespaces = [namespace]

    def __call__(self, x):
        return metatize(x)

    def __getattr__(self, obj):

        ns_obj = next((getattr(ns, obj) for ns in self.namespaces if hasattr(ns, obj)), None)

        if ns_obj is None:
            # Try caller's namespace
            frame = inspect.currentframe()
            f_back = frame.f_back
            if f_back:
                ns_obj = f_back.f_locals.get(obj, None)
                if ns_obj is None:
                    ns_obj = f_back.f_globals.get(obj)

        if isinstance(ns_obj, (types.FunctionType, partial)):
            # It's a function, so let's provide a wrapper that converts
            # to-and-from theano and meta objects.
            @wraps(ns_obj)
            def meta_obj(*args, **kwargs):
                args = [o.reify() if hasattr(o, "reify") else o for o in args]
                res = ns_obj(*args, **kwargs)
                return metatize(res)

        elif isinstance(ns_obj, types.ModuleType):
            # It's a sub-module, so let's create another
            # `TheanoMetaAccessor` and check within there.
            meta_obj = TheanoMetaAccessor(namespace=ns_obj)
        else:
            # Hopefully, it's convertible to a meta object...
            meta_obj = metatize(ns_obj)

        if isinstance(meta_obj, (TheanoMetaSymbol, MetaSymbolType, types.FunctionType)):
            setattr(self, obj, meta_obj)
            return getattr(self, obj)
        elif isinstance(meta_obj, TheanoMetaAccessor):
            setattr(self, obj, meta_obj)
            return meta_obj
        else:
            raise AttributeError(f"Meta object for {obj} not found.")


mt = TheanoMetaAccessor()

mt.dot = metatize(tt.basic._dot)


#
# The wrapped Theano functions will only work when the meta objects
# are fully reifiable (i.e. can be turned to Theano objects), but it's
# fairly straight-forward to adjust many of those functions so that they
# work with meta objects.
# TODO: Would be nice if we could trick Theano into using meta objects, or
# a robust use of "proxy" Theano objects
#


def mt_zeros(shape, dtype=None):
    if not isinstance(shape, (list, tuple, TheanoMetaTensorVariable, tt.TensorVariable)):
        shape = [shape]
    if dtype is None:
        dtype = tt.config.floatX
    return mt.alloc(np.array(0, dtype=dtype), *shape)


mt.zeros = mt_zeros


def mt_diag(v, k=0):
    if v.ndim == 1:
        return mt.AllocDiag(k)(v)
    elif v.ndim is not None and v.ndim >= 2:
        return mt.diagonal(v, offset=k)
    else:
        raise ValueError("Input must has v.ndim >= 1.")


fact(commutative, mt.add)
fact(commutative, mt.mul)
fact(associative, mt.add)
fact(associative, mt.mul)
