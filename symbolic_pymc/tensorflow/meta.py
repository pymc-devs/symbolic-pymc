import types
import inspect

import tensorflow as tf
import tensorflow_probability as tfp

from inspect import Parameter, Signature

from collections import OrderedDict, UserString

from functools import partial, wraps

from unification import Var, var

from tensorflow.python.framework import tensor_util, op_def_registry, op_def_library, tensor_shape
from tensorflow.core.framework.op_def_pb2 import OpDef

from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

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

    def add_op(self, op_def):
        op_info = self._ops.get(op_def.name, None)
        if op_info is None:
            super().add_op(op_def)
            op_info = self._ops[op_def.name]
            opdef_sig = self.make_opdef_sig(op_info.op_def)
            op_info.opdef_sig = opdef_sig
        return op_info


op_def_lib = MetaOpDefLibrary()


class TFlowOpName(UserString):
    """A wrapper for Tensor names.

    TF `Operation` names, and the variables that result from them, cannot be
    compared directly due to their uniqueness (within a TF namespace/scope).

    This wrapper class ignores those TF distinctions during string comparison.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


def _metatize_tf_object(obj):
    try:
        obj = tf.convert_to_tensor(obj)
    except TypeError:
        raise ValueError("Could not find a TensorFlow MetaSymbol class for {}".format(obj))
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
        out_meta_types = tuple(
            # TODO: What other types should we expect?
            TFlowMetaTensor if o.type_attr == "T" else None
            for i, o in enumerate(self.obj.output_arg)
        )
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
        node_def = {
            a.name: apply_arguments[a.name] for a in self.obj.attr if a.name in apply_arguments
        }

        op_mt = TFlowMetaOp(self, node_def, op_input_args, name=op_kwargs.get("name", None))

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


class TFlowMetaOp(TFlowMetaSymbol):
    """A meta `Operation`.

    This is like an `Apply` node in Theano.

    TODO: This whole thing should probably be a "NodeDef" class?
    """

    base = tf.Operation
    __slots__ = ["op_def", "node_def", "inputs", "name"]

    @classmethod
    def process_node_def(cls, node_def, op_def, obj=None):
        """Minimally convert -Proto objects in a `NodeDef` to their corresponding meta classes.

        The result is a dict that somewhat mirrors a NodeDef, but with meta objects.

        FYI: We're using `node_def` as a hackish way to include user-specified
        `OpDef` parameters that aren't standard inputs (e.g. imagine that
        `OpDef`s are parameterized and an `OpDef` instance is only possible
        with a concrete type parameter).
        """
        if isinstance(node_def, NodeDef):
            assert node_def.op == op_def.obj.name
            assert obj is not None
            node_def = dict(node_def.attr)
        elif not isinstance(node_def, dict):
            raise TypeError("Invalid node_def type")

        op_def_attr_names = tuple(
            a.name
            for a in op_def.obj.attr
            # This is a quick way to filter out the useful
            # `attr`s (i.e. the ones that are required `OpDef`
            # parameters).
            if a.name in op_def.apply_func_sig.parameters
        )
        meta_node_def = {}
        for k, v in node_def.items():
            if k not in op_def_attr_names:
                continue
            v = obj.get_attr(k) if obj else v
            if k == "shape":
                if isinstance(v, TensorShapeProto):
                    v = tensor_shape.as_shape(v)
                v = metatize(v)
            elif k == "dtype":
                v = tf.as_dtype(v)
            meta_node_def[k] = v

        return meta_node_def

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

        self.op_def = metatize(op_def)

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

        if self.type is None:
            self.type = self.op_def.obj.name

        self.node_def = self.process_node_def(node_def, self.op_def, obj)
        self.inputs = tuple(metatize(i) for i in inputs)

        if outputs is not None:
            # TODO: Use `weakref`?
            self._outputs = tuple(metatize(o) for o in outputs)
        else:
            self._outputs = None

        super().__init__(obj=obj)

    @property
    def type(self):
        return self.op_def.obj.name

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
            return self.outputs

        apply_arguments = self.op_def.input_args(*self.inputs, **self.node_def)
        out_types_mt = self.op_def.out_meta_types(inputs=apply_arguments)

        mt_outs = tuple(
            o(
                var(),
                op=self,
                value_index=i,
                shape=var(),
                name=(
                    TFlowOpName(f"{self.name.lower()}:{i}")
                    if isinstance(self.name, (str, TFlowOpName))
                    else var()
                ),
            )
            for i, o in enumerate(out_types_mt)
        )

        return mt_outs

    @property
    def default_output(self):
        """Return the default output for this `Operation`.

        TODO: It might be worth considering a direct approach, and not one that
        requires the generation of all meta outputs.
        """
        mt_outs = self.outputs

        if len(mt_outs) == 1:
            out_var = mt_outs[0]
        else:
            out_var = mt_outs

        return out_var

    def reify(self):
        if self.obj and not isinstance(self.obj, Var):
            return self.obj

        # tt_op = self.op.reify()
        # if not self.is_meta(tt_op):
        op_inputs, op_inputs_unreified = _meta_reify_iter(self.inputs)
        op_attrs, op_attrs_unreified = _meta_reify_iter(self.node_def)
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
        if self.op is not None:
            return self.op.op_def

    @property
    def inputs(self):
        """Return the tensor's inputs/rands.

        NOTE: These inputs differ from `self.op.inputs` in that contain
        the `node_def` parameters, as well.
        In other words, these can be used to recreate this object (per
        the meta object spec).
        """
        if self.op is not None:
            input_args = self.op.op_def.input_args(
                *self.op.inputs, name=self.op.name, **self.op.node_def
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
        if self.dims is not None:
            self.dims = [tensor_shape.as_dimension(d) for d in self.dims]
        super().__init__(**kwargs)

    @property
    def rank(self):
        if self.dims is not None:
            return len(self.dims)

    @property
    def ndims(self):
        return self.rank

    def as_list(self):
        return [d.value for d in self.dims]


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

        if not isinstance(obj, tf.Tensor):
            tf_obj = tf.constant(obj, dtype=dtype, shape=shape, name=name)
        else:
            tf_obj = obj
            obj = tensor_util.MakeNdarray(tf_obj.op.get_attr("value"))

        assert tf_obj.op.type == "Const"
        self._data = obj

        super().__init__(
            tf_obj.dtype.name,
            op=tf_obj.op,
            value_index=tf_obj.value_index,
            name=tf_obj.name,
            obj=tf_obj,
        )

    @property
    def data(self):
        """Return the data for a tensor constant as a Python object.

        TF tensors can also be constants, but there's no separate
        class/type for them, so, for access to the underlying constant value,
        we provide this property.
        """
        if hasattr(self, "_data"):
            return self._data
        else:
            self._data = tensor_util.MakeNdarray(self.op.obj.get_attr("value"))
            return self._data

    def __eq__(self, other):
        if self is other:
            return True

        if not (type(self) == type(other)):
            return False

        if not (self.base == other.base):
            return False

        our_data = self.op.obj.get_attr("value")
        other_data = other.op.obj.get_attr("value")
        return our_data == other_data

    def __hash__(self):
        data = self.op.obj.get_attr("value")
        return hash((self.base, data))


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
        raw_op_name = op_def_lib.lower_op_name_to_raw.get(name, None)
        if raw_op_name is not None:
            meta_obj = TFlowMetaOpDef(obj=op_def_registry.get_registered_ops()[raw_op_name])
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

            if meta_obj is None:
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
