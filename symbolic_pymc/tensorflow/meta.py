import types
import inspect

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from inspect import Parameter, Signature

from collections import OrderedDict

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
    meta_reify_iter,
    _metatize,
    metatize,
)


class MetaOpDefLibrary(object):

    lower_op_name_to_raw = {
        op_name.lower(): op_name
        for op_name in dir(tf.raw_ops)
        if callable(getattr(tf.raw_ops, op_name))
    }
    opdef_signatures = {}

    @classmethod
    def apply_op(cls, *args, **kwargs):
        return op_def_library.apply_op(*args, **kwargs)

    @classmethod
    def make_opdef_sig(cls, opdef, opdef_py_func=None):
        """Create a `Signature` object for an `OpDef`.

        Annotations are include so that one can partially verify arguments.
        """
        input_args = OrderedDict([(a.name, a.type or a.type_attr) for a in opdef.input_arg])
        attrs = OrderedDict([(a.name, a.type) for a in opdef.attr])

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
            # We're crafting the Operation from a low-level via `apply_op`.
            opdef_py_func = partial(op_def_lib.apply_op, opdef.name)

            for i_name, i_type in input_args.items():
                p = Parameter(i_name, Parameter.POSITIONAL_OR_KEYWORD, annotation=i_type)
                params[i_name] = p

            # These are the ambiguities we're attempting to overcome
            # with the `tf.raw_ops` functions above.
            for a_name, a_type in attrs.items():

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
                    default=Parameter.empty,
                    annotation=a_type,
                )
                params[a_name] = p

        # Always assume that a name can be specified.
        if "name" not in params:
            params["name"] = Parameter("name", Parameter.POSITIONAL_OR_KEYWORD, default=None)

        opdef_sig = Signature(
            params.values(), return_annotation=[(o.name, o.type_attr) for o in opdef.output_arg]
        )
        return opdef_sig, opdef_py_func

    @classmethod
    def get_op_info(cls, opdef):
        if isinstance(opdef, str):
            opdef_name = opdef
            opdef = op_def_registry.get(opdef_name)
        else:
            opdef_name = opdef.name

        opdef_sig = cls.opdef_signatures.get(opdef_name, None)

        if opdef_sig is None and opdef is not None:
            opdef_func = getattr(tf.raw_ops, opdef.name, None)
            opdef_sig = cls.make_opdef_sig(opdef, opdef_func)
            cls.opdef_signatures[opdef.name] = cls.make_opdef_sig(opdef, opdef_func)

        return opdef_sig


op_def_lib = MetaOpDefLibrary()


def _metatize_tf_object(obj):
    try:
        obj = tf.convert_to_tensor(obj)
    except (TypeError, ValueError):
        raise ValueError("Could not find a TensorFlow MetaSymbol class for {obj}")

    if isinstance(obj, tf.Tensor):
        try:
            obj.op
        except AttributeError:
            raise AttributeError(
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
    __slots__ = ()


class OpDefFactoryType(MetaSymbolType):
    __opdefs__ = {}

    def __call__(cls, obj=None):

        if obj is not None:
            obj_hash = obj.name  # obj.SerializeToString()
            opdef = cls.__opdefs__.get(obj_hash, None)
        else:
            obj_hash = None
            opdef = None

        if opdef is None:
            opdef = super().__call__(obj=obj)

            if obj is not None:
                cls.__opdefs__[obj_hash] = opdef

        return opdef


class TFlowMetaOpDef(MetaOp, metaclass=OpDefFactoryType):
    """A meta `OpDef`.

    This is like an `Op` node in Theano.

    Some useful info/links:
        - https://stackoverflow.com/questions/41147734/looking-for-source-code-of-from-gen-nn-ops-in-tensorflow/41149557#41149557

        - A better way to view an `OpDef`:

            >>> from google.protobuf import json_format
            >>> print(json_format.MessageToJson(opdef))
        - If you want to use an `OpDef` to construct a node, see
          `op_def_library.apply_op`.

    """

    base = OpDef
    __slots__ = ["_apply_func_sig", "_apply_func"]

    def __init__(self, obj=None):
        super().__init__(obj=obj)
        self._apply_func_sig, self._apply_func = op_def_lib.get_op_info(obj)

    def out_meta_types(self, inputs=None, node_def=None):
        def _convert_outputs(o):
            if o.type_attr == "T" and node_def:
                return (TFlowMetaTensor, node_def.attr.get("T", var()))
            elif o.type_attr == "dtype" and inputs:
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
            (k, v)
            for k, v in kwargs.items()
            # Filter out the optional keyword arguments so they we only pass
            # expected arguments to the `OpDef`'s apply function.
            if k in self._apply_func_sig.parameters
        )
        op_args = self._apply_func_sig.bind(*args, **kwargs)
        op_args.apply_defaults()
        return op_args.arguments

    def __call__(self, *args, **kwargs):
        """Create the meta object(s) resulting from an application of this `OpDef`'s implied `Operation`."""
        op_args, op_args_unreified = meta_reify_iter(args)
        op_kwargs, op_kwargs_unreified = meta_reify_iter(kwargs)
        apply_arguments = self.input_args(*op_args, **op_kwargs)

        if not op_args_unreified and not op_kwargs_unreified:

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

            tf_out = self._apply_func(**apply_arguments)
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

        node_attr = {a.name: apply_arguments.get(a.name, a) for a in self.obj.attr}

        op_name = op_kwargs.get("name", self.obj.name)

        node_def = TFlowMetaNodeDef(self.obj.name, op_name, node_attr)

        op_mt = TFlowMetaOp(self, node_def, op_input_args)

        res_var = op_mt.default_output

        return res_var

    def __str__(self):
        return f"{self.__class__.__name__}({self.obj.name})"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self.__class__.__name__}(...)")
        else:
            with p.group(2, f"{self.__class__.__name__}(", ")"):
                p.breakable(sep="")
                p.text(self.obj.name)

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
    __slots__ = ["op", "name", "attr", "_frozen_attr"]

    @classmethod
    def _protobuf_convert(cls, k, v):
        """Convert a small subset of protobuf objects.

        FYI: This would cover a lot at once (but not any meta object
        conversions):
            from google.protobuf.json_format import MessageToDict
            MessageToDict(obj, use_integers_for_enums=True)
        """
        if k == "shape":
            return metatize(tensor_shape.as_shape(v.shape))
        elif k == "dtype":
            return tf.as_dtype(v.type).name
        elif k == "T":
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
        super().__init__(obj=obj)
        self.op = metatize(op)
        assert name is not None
        self.name = name if isvar(name) else str(name)

        if not isvar(attr):
            opdef_sig, _ = op_def_lib.get_op_info(self.op)
            _attr = dict()

            for k, v in attr.items():
                if isinstance(v, Message):
                    try:
                        v = self._protobuf_convert(k, v)
                    except TypeError:
                        v = var()

                _attr[k] = v

            self.attr = _attr
        else:
            self.attr = attr

    @property
    def frozen_attr(self):
        if getattr(self, "_frozen_attr", None) is not None:
            return self._frozen_attr

        if isvar(self.attr):
            self._frozen_attr = self.attr
        else:
            self._frozen_attr = frozenset(
                (k, v.tostring() if isinstance(v, np.ndarray) else v) for k, v in self.attr.items()
            )
        return self._frozen_attr

    def __eq__(self, other):

        if self is other:
            return True

        if not (type(self) == type(other)):
            return False

        if (
            self.op == other.op
            and self.name == other.name
            and self.frozen_attr == other.frozen_attr
        ):
            return True

        return False

    def __hash__(self):
        return hash((hash(self.op), hash(self.name), hash(self.frozen_attr)))


class TFlowMetaOp(TFlowMetaSymbol):
    """A meta `Operation`.

    This is like an `Apply` node in Theano.

    TODO: This whole thing should probably be a "NodeDef" class?
    """

    base = tf.Operation
    __slots__ = ["op_def", "node_def", "inputs", "_name", "_type", "_outputs", "_default_output"]

    @classmethod
    def _metatize(cls, obj):
        """Reformat inputs to match the OpDef."""
        new_input = obj._reconstruct_sequence_inputs(obj.op_def, obj.inputs, obj.node_def.attr)
        new_args = [
            getattr(obj, s) if s != "inputs" else new_input for s in getattr(cls, "__props__", [])
        ]
        return cls(*new_args, obj=obj)

    def __init__(self, op_def, node_def, inputs, outputs=None, obj=None):
        """Create a TensorFlow meta `Operation`.

        The real signature of `tf.Operation.__init__` includes the graph
        object, so we can't really the signature directly.  This is part of the
        reason why we have `TFlowMetaOpFactory.__call__` and
        `TFlowMetaTensor.operator` + `TFlowMetaTensor.inputs` that do not
        directly use `__all_props__`/`TFlowMetaTensor.rands` and construct the
        objects directly.
        """
        super().__init__(obj=obj)

        if isinstance(op_def, str):
            op_def = op_def_registry.get(op_def)

        self.op_def = metatize(op_def)
        self.node_def = metatize(node_def)

        if isvar(inputs):
            self.inputs = inputs
        else:
            # Inputs are supposed to be immutable, so we're able to convert
            # lists to tuples.
            def _convert_inputs(arg, nested):
                if nested and isinstance(arg, list):
                    arg = tuple(metatize(i) for i in arg)
                else:
                    arg = metatize(arg)

                return arg

            if not isvar(self.op_def):
                self.inputs = tuple(
                    _convert_inputs(i, hasattr(info, "number_attr"))
                    for i, info in zip(inputs, self.op_def.obj.input_arg)
                )
            else:
                self.inputs = tuple(_convert_inputs(i, False) for i in inputs)

        if outputs is not None:
            if isvar(outputs):
                self._outputs = outputs
            else:
                self._outputs = tuple(metatize(o) for o in outputs)

    @property
    def type(self):
        if getattr(self, "_type", None) is not None:
            return self._type

        if isvar(self.op_def):
            self._type = var()
        else:
            self._type = self.op_def.obj.name

        return self._type

    @property
    def name(self):
        if getattr(self, "_name", None) is not None:
            return self._name

        if isvar(self.node_def):
            self._name = var()
        else:
            self._name = self.node_def.name

        return self._name

    @property
    def outputs(self):
        """Compute meta object outputs for this meta `Operation`.

        If the outputs were specified during construction of this meta
        `Operation`, then those outputs are always returned.

        NOTE: These values are dynamically computed, but they could be cached.
        One of the reasons that they're dynamically computed: as constituent
        meta elements are unified, we may obtain more information about the
        """
        if getattr(self, "_outputs", None) is not None:
            return self._outputs

        if (
            isvar(self.op_def)
            or isvar(self.inputs)
            or isvar(self.node_def)
            or isvar(self.node_def.attr)
        ):
            self._outputs = var()
        else:

            apply_arguments = self.op_def.input_args(*self.inputs, **self.node_def.attr)
            out_types_mt = self.op_def.out_meta_types(
                inputs=apply_arguments, node_def=self.node_def
            )

            mt_outs = tuple(
                o_type(self, i, o_dtype) for i, (o_type, o_dtype) in enumerate(out_types_mt)
            )

            self._outputs = mt_outs

        return self._outputs

    @property
    def default_output(self):
        """Return the default output for this `Operation`.

        TODO: It might be worth considering a direct approach, and not one that
        requires the generation of all meta outputs.
        """
        if getattr(self, "_default_output", None):
            return self._default_output

        mt_outs = self.outputs

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
        op_inputs, op_inputs_unreified = meta_reify_iter(self.inputs)

        if isvar(self.node_def):
            return self

        op_attrs, op_attrs_unreified = meta_reify_iter(
            # Only use NodeDef attrs that appear in the OpDef's call signature.
            # Other NodeDef attrs, like dtype and shape, can be computed.
            {
                k: v
                for k, v in self.node_def.attr.items()
                if k in self.op_def._apply_func_sig.parameters
            }
        )

        if not (op_inputs_unreified or op_attrs_unreified or MetaSymbol.is_meta(self.name)):

            # We have to use a primitive string or TF will complain.
            name = self.name
            if self.name is not None:
                name = str(name)

            apply_arguments = self.op_def.input_args(*op_inputs, name=name, **op_attrs)
            tf_out = self.op_def._apply_func(**apply_arguments)
            op_tf = tf_out.op

            # TODO: Update NodeDef attrs?

            assert op_tf is not None
            self._obj = op_tf
            return self.obj

        return self


class TFlowMetaTensor(TFlowMetaSymbol, MetaVariable):
    base = tf.Tensor
    __slots__ = ("op", "value_index", "dtype", "_shape", "_name")

    def __init__(self, op, value_index, dtype, obj=None):
        self.op = metatize(op)
        # TODO: Sync this value with `op.node_def.attr['dtype']` and/or
        # `op.node_def.attr['T']`?
        self.dtype = dtype
        self.value_index = value_index
        super().__init__(obj=obj)

    @property
    def shape(self):
        if getattr(self, "_shape", None):
            return self._shape

        if self.obj is not None and not isinstance(self.obj, Var):
            self._shape = metatize(self.obj.shape)
        else:
            self._shape = TFlowMetaTensorShape(var())

        return self._shape

    @property
    def name(self):
        if getattr(self, "_name", None):
            return self._name

        if self.obj is not None and not isinstance(self.obj, Var):
            name = self.obj.name
        elif isinstance(getattr(self.op, "name", None), str) and not isvar(self.value_index):
            name = f"{self.op.name}:{self.value_index}"
        else:
            name = var()

        self._name = name
        return self._name

    @property
    def operator(self):
        if self.op is not None and not isvar(self.op):
            return self.op.op_def

    @property
    def inputs(self):
        """Return the tensor's inputs/rands.

        NOTE: These inputs differ from `self.op.inputs` in that they contain
        the `node_def` parameters, as well.
        In other words, these can be used to recreate this object (per
        the meta object spec).
        """
        # TODO: In keeping with our desire to return logic variables in cases
        # where params aren't given/inferred, we could return something like
        # `cons(var(), var())` here (although that wouldn't be necessarily imply
        # that the result is a proper list/tuple).
        if self.op is not None and not isvar(self.op):
            input_args = self.op.op_def.input_args(
                *self.op.inputs,
                name=self.op.name if not isvar(self.op.name) else None,
                **self.op.node_def.attr,
            )
            return tuple(input_args.values())

    def reify(self):
        if self.obj is not None and not isinstance(self.obj, Var):
            return self.obj

        if (not self.op) or isvar(self.op):
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

            self._obj = tf_res
            return tf_res

        return self


class TFlowMetaTensorShape(TFlowMetaSymbol):
    base = tf.TensorShape
    __slots__ = ("dims", "_rank")

    def __init__(self, dims, obj=None):
        super().__init__(obj=obj)
        self.dims = dims
        if self.dims is not None and not isvar(self.dims):
            # TODO: Just like the comment in `TFlowMetaTensor.inputs`,
            # `self.dims` should be something like `cons(var(), ...)` and not a
            # straight logic variable.
            self.dims = tuple(tensor_shape.as_dimension(d).value for d in self.dims)

    @property
    def rank(self):
        if getattr(self, "_rank", None):
            return self._rank

        if self.dims is not None and not isvar(self.dims):
            rank = len(self.dims)
        else:
            # TODO: How do we represent/tie in len(var())?
            rank = var()

        self._rank = rank
        return self._rank

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
        raw_op_name = op_def_lib.lower_op_name_to_raw.get(name.lower(), name)
        op_def = op_def_registry.get(raw_op_name)

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

        if isinstance(meta_obj, (MetaSymbol, MetaSymbolType, types.FunctionType)):
            setattr(self, obj, meta_obj)
            return getattr(self, obj)
        elif isinstance(meta_obj, TFlowMetaAccessor):
            setattr(self, obj, meta_obj)
            return meta_obj
        else:
            raise AttributeError(f"Meta object for {obj} not found.")


mt = TFlowMetaAccessor()
