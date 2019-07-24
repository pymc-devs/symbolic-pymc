import types
import inspect

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from contextlib import suppress

from inspect import Parameter, Signature

from collections import OrderedDict, UserString

from functools import partial

from unification import Var, var, isvar

from google.protobuf.message import Message

from tensorflow.python.framework import tensor_util, op_def_registry, op_def_library, tensor_shape
from tensorflow.core.framework.op_def_pb2 import OpDef
from tensorflow.core.framework.node_def_pb2 import NodeDef

# from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

from tensorflow_probability import distributions as tfd


from ..meta import (
    MetaSymbol,
    MetaSymbolType,
    MetaOp,
    MetaVariable,
    _meta_reify_iter,
    _metatize,
    metatize,
)


class MetaOpDefLibrary(op_def_library.OpDefLibrary):
    def __init__(self, *args, **kwargs):
        # This is a lame way to fix the numerous naming inconsistencies between
        # TF `Operation`s, `OpDef`s, and the corresponding user-level functions.
        self.lower_op_name_to_raw = {
            op_name.lower(): op_name
            for op_name in dir(tf.raw_ops)
            if callable(getattr(tf.raw_ops, op_name))
        }
        super().__init__(*args, **kwargs)

    @classmethod
    def make_opdef_sig(cls, opdef):
        """Create a `Signature` object for an `OpDef`.

        Annotations are include so that one can partially verify arguments.
        """
        input_args = OrderedDict([(a.name, a.type or a.type_attr) for a in opdef.input_arg])
        attrs = OrderedDict([(a.name, a.type) for a in opdef.attr])

        opdef_py_func = getattr(tf.raw_ops, opdef.name, None)

        params = OrderedDict()
        if opdef_py_func:
            # We assume we're dealing with a function from `tf.raw_ops`.
            # Those functions have only the necessary `input_arg`s and
            # `attr` inputs as arguments.
            opdef_func_sig = Signature.from_callable(opdef_py_func)

            for name, param in opdef_func_sig.parameters.items():
                # We make positional parameters permissible (since the
                # functions in `tf.raw_ops` are keyword-only), and we use the
                # `tf.raw_ops` arguments to determine the *actual* required
                # arguments (because `OpDef`'s `input_arg`s and `attrs` aren't
                # exactly clear about that).
                if name in input_args:
                    new_default = Parameter.empty
                    new_annotation = input_args[name]
                else:
                    new_default = None
                    new_annotation = attrs.get(name, None)

                new_param = param.replace(
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    default=new_default,
                    annotation=new_annotation,
                )
                params[name] = new_param

        else:
            params = []
            for i_name, i_type in input_args:
                p = Parameter(i_name, Parameter.POSITIONAL_OR_KEYWORD, annotation=i_type)
                params[i_name] = p

            # These are the ambiguities we're attempting to overcome
            # with the `tf.raw_ops` functions above.
            for a_name, a_type in attrs:
                if a_name == "T":
                    # This is a type value that will most likely be inferred
                    # from/by the inputs.
                    # TODO: We could check for an `allowed_values` attribute.
                    continue
                p = Parameter(
                    a_name,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    # TODO: We could use the `default_value`
                    # attribute.
                    default=None,
                    annotation=a_type,
                )
                params[a_name] = p

        # Always assume that a name can be specified.
        if "name" not in params:
            params["name"] = Parameter("name", Parameter.POSITIONAL_OR_KEYWORD, default=None)

        opdef_sig = Signature(
            params.values(), return_annotation=[(o.name, o.type_attr) for o in opdef.output_arg]
        )
        return opdef_sig

    def add_op(self, opdef):
        op_info = self._ops.get(opdef.name, None)
        if op_info is None:
            super().add_op(opdef)
            op_info = self._ops[opdef.name]
            opdef_sig = self.make_opdef_sig(op_info.op_def)
            op_info.opdef_sig = opdef_sig
        return op_info

    def get_opinfo(self, opdef):
        if isinstance(opdef, str):
            opdef = op_def_registry.get_registered_ops()[opdef]
        return self.add_op(opdef)


op_def_lib = MetaOpDefLibrary()


class TFlowOpName(UserString):
    """A wrapper for Tensor names.

    TF `Operation` names, and the variables that result from them, cannot be
    compared directly due to their uniqueness (within a TF namespace/scope).

    This wrapper class ignores those TF distinctions during string comparison.
    """

    def __init__(self, s):
        super().__init__(s)

        if isinstance(s, type(self)):
            self._scope_op = s._scope_op
            self._in_idx = s._in_idx
            self._scope = s._scope
            self._op_name = s._op_name
            self._unique_name = s._unique_name
        else:
            self._scope_op, _, self._in_idx = self.data.partition(":")
            self._scope, _, self._op_name = self._scope_op.rpartition("/")
            self._unique_name = self._op_name.split("_", 1)[0]

    def __eq__(self, other):
        if self is other:
            return True

        if isinstance(other, str):
            return self.data == other or self._unique_name == other

        if type(self) != type(other):
            return False

        return self._unique_name == other._unique_name and self._in_idx == other._in_idx

    def __hash__(self):
        return hash((self._unique_name, self._in_idx))


def _metatize_tf_object(obj):
    try:
        obj = tf.convert_to_tensor(obj)
    except TypeError:
        raise ValueError("Could not find a TensorFlow MetaSymbol class for {obj}")

    if isinstance(obj, tf.Tensor):
        try:
            obj.op
        except AttributeError:
            raise ValueError(
                f"TensorFlow Operation not available; "
                "try recreating the object with eager-mode disabled"
                " (e.g. within `tensorflow.python.eager.context.graph_mode`)"
            )

    return _metatize(obj)


def load_dispatcher():
    """Set/override dispatcher to default to TF objects."""
    _metatize.add((object,), _metatize_tf_object)


load_dispatcher()


class TFlowMetaSymbol(MetaSymbol):
    def reify(self):
        # TODO: Follow `tfp.distribution.Distribution`'s lead?
        # with tf.name_scope(self.name):
        #     pass
        return super().reify()


class TFlowMetaOpDef(MetaOp, TFlowMetaSymbol):
    """A meta `OpDef`.

    This is like an `Op` node in Theano.

    Some useful info/links:
        - https://stackoverflow.com/questions/41147734/looking-for-source-code-of-from-gen-nn-ops-in-tensorflow/41149557#41149557

        - A better way to view an `OpDef`:

            >>> from google.protobuf import json_format
            >>> print(json_format.MessageToJson(opdef))
        - If you want to use an `OpDef` to construct a node, see
          `op_def_library.OpDefLibrary.apply_op`.

    """

    base = OpDef

    def __init__(self, obj=None):
        op_info = op_def_lib.add_op(obj)
        self.apply_func_sig = op_info.opdef_sig
        self.apply_func = partial(op_def_lib.apply_op, obj.name)
        super().__init__(obj=obj)

    def out_meta_types(self, inputs=None):
        def _convert_outputs(o):
            if o.type_attr == "T":
                return (TFlowMetaTensor, var())
            elif o.type_attr == "dtype":
                return (TFlowMetaTensor, inputs.get("dtype", var()))
            else:
                return (TFlowMetaTensor, var())

        out_meta_types = tuple(_convert_outputs(o) for o in self.obj.output_arg)
        # TODO: We also have permissible dtype information:
        # from objects in the array `self.obj.attr` under the field
        # `allowed_values`.
        return out_meta_types

    def input_args(self, *args, **kwargs):
        kwargs = OrderedDict(
            [
                (k, v)
                for k, v in kwargs.items()
                # Filter out the optional keyword arguments so they we only pass
                # expected arguments to the `OpDef`'s apply function.
                if k in self.apply_func_sig.parameters
            ]
        )
        op_args = self.apply_func_sig.bind(*args, **kwargs)
        op_args.apply_defaults()
        return op_args.arguments

    def __call__(self, *args, **kwargs):
        """Create the meta object(s) resulting from an application of this `OpDef`'s implied `Operation`."""
        op_args, op_args_unreified = _meta_reify_iter(args)
        op_kwargs, op_kwargs_unreified = _meta_reify_iter(kwargs)
        apply_arguments = self.input_args(*op_args, **op_kwargs)

        if not op_args_unreified and not op_kwargs_unreified:
            # In this case, we can actually create the TF objects and then turn
            # them into meta objects.  Doing so will yield information we
            # wouldn't be able to produce otherwise (e.g. shape info).

            # TODO: We could make this action/approach configurable (i.e.
            # do not perform intermediate/"eager" object construction).
            # Especially, when/if we're comfortable with our ability to infer
            # the TF-`Operation` inferred values (e.g. shapes, dtypes, etc.)

            # We have to use a primitive string or TF will complain.
            name = apply_arguments.get("name", None)
            if name is not None:
                apply_arguments["name"] = str(name)

            tf_out = self.apply_func(**apply_arguments)
            res_var = metatize(tf_out)
            return res_var

        #
        # If we're here, that means we have to create the meta objects
        # manually.
        #

        # TODO: `tf.placeholder`s are pretty flexible, we could probably use
        # one as a stand-in for any un-reified tensor arguments and at least
        # get some partial `dtype`, `shape` and `name` info.

        op_input_args = tuple(
            apply_arguments.get(i.name) for i in self.obj.input_arg if i.name in apply_arguments
        )

        # Get the `OpDef`-instantiating parameters and call them a "node_def".
        node_attr = {a.name: apply_arguments.get(a.name, a) for a in self.obj.attr}

        op_name = op_kwargs.get("name", self.obj.name.lower())

        # input_arg_names = [(getattr(a, 'name', None), i.name)
        #                    for a, i in zip(args, self.obj.input_arg)]
        # if isvar(op_name) or any(isvar(a) or isvar(i) for a, i in input_arg_names):
        #     node_input = [var() for i in self.obj.input_arg]
        # else:
        #     node_input = [getattr(a, 'name', None) or f"{op_name}/{i.name}"
        #                   for a, i in input_arg_names]

        node_def = TFlowMetaNodeDef(self.obj.name, op_name, node_attr)

        op_mt = TFlowMetaOp(self, node_def, op_input_args, name=op_name)

        res_var = op_mt.default_output

        return res_var

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj.name})"

    def _repr_pretty_(self, p, cycle):
        return p.text(f"{self.__class__.__name__}({self.obj.name})")

    def __eq__(self, other):
        if self is other:
            return True

        if not (type(self) == type(other)):
            return False

        if not (self.base == other.base):
            return False

        return self.obj.name == other.obj.name

    def __hash__(self):
        return hash((self.base, self.obj.name))


class TFlowMetaNodeDef(TFlowMetaSymbol):
    """A meta `NodeDef`.

    NOTE: We're ignoring `node_def.input`; it's just an unnecessary hassle.
    """

    base = NodeDef
    __slots__ = ["op", "name", "attr"]

    @classmethod
    def _protobuf_convert(cls, k, v):
        """Convert a small subset of protobuf objects.

        FYI: This would cover a lot at once (but not any meta object
        conversions):
            from google.protobuf.json_format import MessageToDict
            MessageToDict(obj, use_integers_for_enums=True)
        """

        if k == "shape":
            return tensor_shape.as_shape(v.shape)
        elif k == "dtype":
            return tf.as_dtype(v.type).name
        elif k == "value":
            return tensor_util.MakeNdarray(v.tensor)
        else:
            # Consider only the narrow case where a single object is converted
            # (e.g. a Python builtin type under `v.b`, `v.f`, etc.)
            v = tuple(v for k, v in v.ListFields())
            if len(v) == 1:
                return v[0]
            else:
                raise TypeError(f"Could not convert {k}")

    def __init__(self, op, name, attr, obj=None):
        self.op = metatize(op)
        assert name is not None
        self.name = name if isvar(name) else TFlowOpName(name)

        if not isvar(attr):
            self.attr = OrderedDict()

            # We want to limit the attributes we'll consider to those that show
            # up in an OpDef function's signature (e.g. ignore info about
            # permissible types).
            opinfo = op_def_lib.get_opinfo(self.op)
            op_param_names = opinfo.opdef_sig.parameters.keys()

            for k, v in sorted(dict(attr).items()):
                if isinstance(v, Message):
                    try:
                        v = self._protobuf_convert(k, v)
                    except TypeError:
                        continue

                if k != "T" and k in op_param_names:
                    # XXX: We can't let `metatize` convert NumPy values;
                    # otherwise, we'll loop endlessly on "Const" Ops.
                    if k != "value":
                        with suppress(ValueError):
                            v = metatize(v)

                    self.attr[k] = v
        else:
            self.attr = attr

        super().__init__(obj=obj)


class TFlowMetaOp(TFlowMetaSymbol):
    """A meta `Operation`.

    This is like an `Apply` node in Theano.

    TODO: This whole thing should probably be a "NodeDef" class?
    """

    base = tf.Operation
    __slots__ = ["op_def", "node_def", "inputs", "name"]

    @classmethod
    def _metatize(cls, obj):
        """Reformat inputs to match the OpDef."""
        new_input = obj._reconstruct_sequence_inputs(obj.op_def, obj.inputs, obj.node_def.attr)
        new_args = [
            getattr(obj, s) if s != "inputs" else new_input for s in getattr(cls, "__slots__", [])
        ]
        return cls(*new_args, obj=obj)

    def __init__(self, op_def, node_def, inputs, name=None, outputs=None, obj=None):
        if isinstance(op_def, str):
            op_def = op_def_registry.get_registered_ops()[op_def]

        if isvar(op_def):
            self.op_def = op_def
            # Create a logic variables to fill missing properties
            # obtained/inferred from a missing OpDef.
            self.type = var()
            self.name = var() if name is None else name
            if outputs is None:
                self._outputs = var()
            elif isvar(outputs):
                self._outputs = outputs
            else:
                self._outputs = tuple(metatize(o) for o in outputs)
        else:
            self.op_def = metatize(op_def)
            self.type = self.op_def.obj.name

            if isinstance(name, (str, TFlowOpName)) or name is None:
                if name is None:
                    name = op_def.obj.name.lower()
                # from tensorflow.python.framework import ops
                # if name and name[-1] == "/":
                #     name = ops._name_from_scope_name(str(name))
                # else:
                #     g_tf = ops.get_default_graph()
                #     name = g_tf.unique_name(str(name))
                self.name = TFlowOpName(name)
            else:
                self.name = name

            if not isvar(outputs) and outputs is not None:
                # TODO: Use `weakref`?
                self._outputs = tuple(metatize(o) for o in outputs)
            else:
                self._outputs = outputs

        self.node_def = metatize(node_def)

        if isvar(inputs):
            self.inputs = inputs
        else:

            def _convert_inputs(i):
                i = metatize(i)
                # Inputs are supposed to be immutable, so we're able to convert
                # lists to tuples.
                if isinstance(i, list):
                    i = tuple(i)
                return i

            self.inputs = tuple(_convert_inputs(i) for i in inputs)

        super().__init__(obj=obj)

    @property
    def outputs(self):
        """Compute meta object outputs for this meta `Operation`.

        If the outputs were specified during construction of this meta
        `Operation`, then those outputs are always returned.

        NOTE: These values are dynamically computed, but they could be cached.
        One of the reasons that they're dynamically computed: as constituent
        meta elements are unified, we may obtain more information about the
        """
        if self._outputs is not None:
            return self._outputs

        apply_arguments = self.op_def.input_args(*self.inputs, **self.node_def.attr)
        out_types_mt = self.op_def.out_meta_types(inputs=apply_arguments)

        mt_outs = tuple(
            o_type(
                var() if o_dtype is None else o_dtype,
                op=self,
                value_index=i,
                shape=var(),
                name=(
                    TFlowOpName(f"{self.name.lower()}:{i}")
                    if isinstance(self.name, (str, TFlowOpName))
                    else var()
                ),
            )
            for i, (o_type, o_dtype) in enumerate(out_types_mt)
        )

        self._outputs = mt_outs

        return self._outputs

    @property
    def default_output(self):
        """Return the default output for this `Operation`.

        TODO: It might be worth considering a direct approach, and not one that
        requires the generation of all meta outputs.
        """
        if hasattr(self, "_default_output"):
            return self._default_output

        mt_outs = self.outputs

        if isvar(mt_outs):
            out_var = var()
        if len(mt_outs) == 1:
            out_var = mt_outs[0]
        else:
            out_var = mt_outs

        self._default_output = out_var
        return self._default_output

    def reify(self):
        if self.obj and not isinstance(self.obj, Var):
            return self.obj

        # tt_op = self.op.reify()
        # if not self.is_meta(tt_op):
        op_inputs, op_inputs_unreified = _meta_reify_iter(self.inputs)
        op_attrs, op_attrs_unreified = _meta_reify_iter(self.node_def.attr)
        if not op_inputs_unreified and not op_attrs_unreified and not MetaSymbol.is_meta(self.name):

            # We have to use a primitive string or TF will complain.
            name = self.name
            if self.name is not None:
                name = str(name)

            apply_arguments = self.op_def.input_args(*op_inputs, name=name, **op_attrs)
            tf_out = self.op_def.apply_func(**apply_arguments)
            op_tf = tf_out.op

            assert op_tf is not None
            self.obj = op_tf
            return self.obj

        return self


class TFlowMetaOpFactory(MetaSymbolType):
    """A type that enables the creation of meta `OpDef`s by their string names.

    Example
    -------
    >>> TFlowMetaTensor('float64', 'Placeholder')
    TFlowMetaTensor(tf.float64, TFlowMetaOp(TFlowMetaOpDef(obj=name: "Placeholde...bj=<tf.Operation 'Placeholder' type=Placeholder>), 0, TFlowMetaTensorShape(None,, obj=TensorShape(None)), 'Placeholder:0', obj=<tf.Tensor 'Placeholder:0' shape=<unknown> dtype=float64>)

    """

    _op_types = {}

    def __new__(cls, name, bases, clsdict):
        op_type = clsdict.get("op_type", None)
        new_cls = super().__new__(cls, name, bases, clsdict)
        if op_type is not None:
            cls._op_types[op_type] = new_cls
        return new_cls

    def __call__(cls, dtype=None, op=None, value_index=None, shape=None, name=None, obj=None):
        if isinstance(op, str):
            # Attempt to construct this meta `Tensor` from the `OpDef` and
            # these `Tensor` arguments.
            # NOTE: This is really only expected to work for nullary
            # `Operation`s (with, perhaps, some optional arguments).
            op_def_mt = TFlowMetaOpDef(obj=op_def_registry.get_registered_ops()[op.capitalize()])

            obj_mt = op_def_mt(dtype=dtype, shape=shape, name=name)
            return obj_mt

        return type.__call__(
            cls, dtype, op=op, value_index=value_index, shape=shape, name=name, obj=obj
        )


class TFlowMetaTensor(MetaVariable, TFlowMetaSymbol, metaclass=TFlowMetaOpFactory):
    base = tf.Tensor
    __slots__ = ("dtype", "op", "value_index", "shape", "name")

    @classmethod
    def _metatize(cls, obj):
        """Specialize the meta type based on a `tf.Tensor`'s op."""

        try:
            obj.op
        except AttributeError:
            raise ValueError(
                f"TensorFlow Operation not available; "
                "try recreating the object with eager-mode disabled"
                " (e.g. within `tensorflow.python.eager.context.graph_mode`)"
            )

        cls = TFlowMetaTensor._op_types.get(obj.op.type, cls)
        return cls(*[getattr(obj, s) for s in getattr(cls, "__slots__", [])], obj=obj)

    def __init__(self, dtype, op=None, value_index=None, shape=None, name=None, obj=None):
        self.dtype = dtype
        self.op = metatize(op)
        self.shape = metatize(shape)
        self.value_index = value_index
        self.name = TFlowOpName(name) if isinstance(name, str) else name
        super().__init__(obj=obj)

    @property
    def operator(self):
        if self.op is not None and not isvar(self.op):
            return self.op.op_def

    @property
    def inputs(self):
        """Return the tensor's inputs/rands.

        NOTE: These inputs differ from `self.op.inputs` in that contain
        the `node_def` parameters, as well.
        In other words, these can be used to recreate this object (per
        the meta object spec).
        """
        if self.op is not None and not isvar(self.op):
            input_args = self.op.op_def.input_args(
                *self.op.inputs, name=self.op.name, **self.op.node_def.attr
            )
            return tuple(input_args.values())

    def reify(self):
        if self.obj is not None and not isinstance(self.obj, Var):
            return self.obj

        if not self.op:
            op_res = super().reify()
            return op_res

        tf_op = self.op.reify()

        if not MetaSymbol.is_meta(tf_op):

            if not MetaSymbol.is_meta(self.value_index):
                tf_res = tf_op.outputs[self.value_index]
            elif len(tf_op.outputs) == 1:
                tf_res = tf_op.outputs[0]
            else:
                # TODO: Anything else we should/can do here?
                return self

            self.obj = tf_res
            return tf_res

        return self


class TFlowMetaTensorShape(TFlowMetaSymbol):
    base = tf.TensorShape
    __slots__ = ("dims",)

    def __init__(self, dims, **kwargs):
        self.dims = dims
        if self.dims is not None and not isvar(self.dims):
            self.dims = tuple(tensor_shape.as_dimension(d).value for d in self.dims)
        super().__init__(**kwargs)

    @property
    def rank(self):
        if self.dims is not None and not isvar(self.dims):
            return len(self.dims)

    @property
    def ndims(self):
        return self.rank

    def as_list(self):
        if self.dims is not None and not isvar(self.dims):
            return list(self.dims)
        else:
            return self.dims

    def __hash__(self):
        return hash((self.base, self.dims))


class TFlowConstantType(type):
    # def __subclasscheck__(self, c):

    def __instancecheck__(self, o):
        if isinstance(o, tf.Tensor):
            return o.op.type == "Const"
        return False


class _TFlowConstant(tf.Tensor, metaclass=TFlowConstantType):
    """A helper for `isinstance` functionality."""

    pass


class TFlowMetaConstant(TFlowMetaTensor):
    base = _TFlowConstant
    __slots__ = ()
    op_type = "Const"

    def __init__(self, dtype=None, op=None, value_index=None, shape=None, name=None, obj=None):

        assert obj is not None

        # If `obj` is a NumPy array, create the corresponding TF object, and,
        # if it's a TF object, create the corresponding NumPy array.
        if not isinstance(obj, tf.Tensor):
            tf_obj = tf.constant(obj, dtype=dtype, shape=shape, name=name)
        else:
            tf_obj = obj
            obj = tf_obj.op.node_def.attr["value"]
            obj = tensor_util.MakeNdarray(obj.tensor)

        assert tf_obj.op.type == "Const"
        self.data = obj

        super().__init__(
            tf_obj.dtype.name,
            op=tf_obj.op,
            value_index=tf_obj.value_index,
            name=tf_obj.name,
            obj=tf_obj,
        )

    def __eq__(self, other):
        if self is other:
            return True

        if not (type(self) == type(other)):
            return False

        if not (self.base == other.base):
            return False

        return np.array_equal(self.data, other.data)

    def __hash__(self):
        return hash((self.base, self.data.tostring()))


class TFlowMetaAccessor(object):
    """An accessor object that simplifies the use of meta objects.

    Instances of this class can be used to implicitly convert TensorFlow
    functions and objects into meta objects.
    """

    namespaces = [tf, tf.raw_ops, tfp, tfd]

    def __init__(self, namespace=None):
        if namespace is None:
            from symbolic_pymc.tensorflow import meta  # pylint: disable=import-self

            self.namespaces += [meta]
        else:
            self.namespaces = [namespace]

    def __call__(self, x):
        return metatize(x)

    @classmethod
    def find_opdef(cls, name):
        """Attempt to create a meta `OpDef` for a given TF function/`Operation` name."""
        raw_op_name = op_def_lib.lower_op_name_to_raw.get(name, name)
        op_def = op_def_registry.get_registered_ops()[raw_op_name]

        if op_def is not None:
            meta_obj = TFlowMetaOpDef(obj=op_def)
            return meta_obj

        return None

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
            # We assume that the user requested an `Operation`
            # constructor/helper.  Return the meta `OpDef`, because
            # it implements a constructor/helper-like `__call__`.
            meta_obj = self.find_opdef(obj)

            # if meta_obj is None:
            #     # It's a function, so let's provide a wrapper that converts
            #     # to-and-from theano and meta objects.
            #     @wraps(ns_obj)
            #     def meta_obj(*args, **kwargs):
            #         args = [o.reify() if hasattr(o, "reify") else o for o in args]
            #         res = ns_obj(*args, **kwargs)
            #         return metatize(res)

        elif isinstance(ns_obj, types.ModuleType):
            # It's a sub-module, so let's create another
            # `TheanoMetaAccessor` and check within there.
            meta_obj = TFlowMetaAccessor(namespace=ns_obj)
        else:

            # Hopefully, it's convertible to a meta object...
            meta_obj = metatize(ns_obj)

            if meta_obj is None:
                # Last resort
                meta_obj = self.find_opdef(obj)

        if isinstance(meta_obj, (TFlowMetaSymbol, MetaSymbolType, types.FunctionType)):
            setattr(self, obj, meta_obj)
            return getattr(self, obj)
        elif isinstance(meta_obj, TFlowMetaAccessor):
            setattr(self, obj, meta_obj)
            return meta_obj
        else:
            raise AttributeError(f"Meta object for {obj} not found.")


mt = TFlowMetaAccessor()
