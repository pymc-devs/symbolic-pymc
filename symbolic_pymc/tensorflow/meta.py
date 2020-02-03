import types
import inspect

import tensorflow as tf
import tensorflow_probability as tfp

from inspect import Parameter, Signature

from collections import OrderedDict, Sequence

from functools import partial

from cachetools import cachedmethod, Cache

from unification import Var, var, isvar

from google.protobuf.message import Message

from tensorflow.python.framework import (
    tensor_util,
    op_def_registry,
    op_def_library,
    tensor_shape,
    ops,
)
from tensorflow.core.framework.op_def_pb2 import OpDef
from tensorflow.core.framework.node_def_pb2 import NodeDef

from tensorflow_probability import distributions as tfd


from ..meta import (
    MetaSymbol,
    MetaSymbolType,
    MetaOp,
    MetaVariable,
    MetaReificationError,
    meta_reify_iter,
    _metatize,
    metatize,
)

from .. import meta

from ..utils import HashableNDArray


tf_metatize_cache = Cache(50)


class MetaOpDefLibrary(object):
    """A singleton-like object that holds correspondences between TF Python API functions and the `OpDef`s they construct.

    It provides a map of `OpDef` names (lower-cased) to the Python API
    functions in `tensorflow.raw_ops`, as well as `inspect.Signature` objects
    for said functions so that default values and lists of arguments (keywords
    included) can be more easily used.

    """

    lower_op_name_to_raw = {
        op_name.lower(): op_name
        for op_name in dir(tf.raw_ops)
        if callable(getattr(tf.raw_ops, op_name))
    }
    opdef_signatures = {}

    def __init__(self):
        #
        # We need this in order to construct "Const" tensors directly, since
        # the "value" attr in a meta `NodeDef` is just a NumPy array and not
        # the `TensorProto` expected by `raw_ops.Const`.
        #
        def mt_const(value, dtype, name=None):
            return tf.raw_ops.Const(
                value=tensor_util.make_tensor_proto(value), dtype=dtype, name=name
            )

        opdef = op_def_registry.get("Const")
        self.opdef_signatures[opdef.name] = self.make_opdef_sig(opdef, mt_const)

    @classmethod
    def get_op_info(cls, opdef):
        """Return the TF Python API function signature for a given `OpDef`.

        Parameter
        ---------
           opdef: str or `OpDef` object (meta or base)
        """
        if isinstance(opdef, str):
            opdef_name = opdef
            opdef = op_def_registry.get(opdef_name)
        else:
            opdef_name = opdef.name

        opdef_sig = cls.opdef_signatures.get(opdef_name, None)

        if opdef_sig is None and opdef is not None:
            opdef_func = getattr(tf.raw_ops, opdef.name, None)
            opdef_sig = cls.make_opdef_sig(opdef, opdef_func)
            cls.opdef_signatures[opdef.name] = opdef_sig

        return opdef_sig

    @classmethod
    def make_opdef_sig(cls, opdef, opdef_py_func=None):
        """Create a `Signature` object for an `OpDef`.

        Annotations are include so that one can partially verify arguments.
        """
        if opdef_py_func:
            #
            # We assume we're dealing with a function from `tf.raw_ops`.
            # Those functions have only the necessary `input_arg`s and `attr`
            # inputs as arguments.
            #
            opdef_func_sig = Signature.from_callable(opdef_py_func)
            params = opdef_func_sig.parameters

        else:
            #
            # We're crafting an `Operation` at a low-level via `apply_op`
            # (like the functions in `tf.raw_ops` do)
            #
            input_args = OrderedDict([(a.name, a.type or a.type_attr) for a in opdef.input_arg])
            attrs = OrderedDict([(a.name, a) for a in opdef.attr])
            params = OrderedDict()

            opdef_py_func = partial(op_def_library.apply_op, opdef.name)

            for i_name, i_type in input_args.items():
                p = Parameter(i_name, Parameter.POSITIONAL_OR_KEYWORD, annotation=i_type)
                params[i_name] = p

            # These are the ambiguities we're attempting to overcome
            # with the `tf.raw_ops` functions above.
            for a_name, a_value in attrs.items():

                # TODO: Check
                if a_value.type == "type":
                    # This is a type value that will most likely be inferred
                    # from/by the inputs.
                    # TODO: We could check for an `allowed_values` attribute.
                    continue

                default_value = Parameter.empty
                # if a_value.HasField('default_value'):
                #     # TODO: Check `a_value.type` and extract Python value.
                #     default_value = a_value.default_value

                p = Parameter(
                    a_name,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=default_value,
                    annotation=a_value.type,
                )
                params[a_name] = p

        # Always assume that a name can be specified.
        if "name" not in params:
            params["name"] = Parameter("name", Parameter.POSITIONAL_OR_KEYWORD, default=None)

        opdef_sig = Signature(
            params.values(), return_annotation=[(o.name, o.type_attr) for o in opdef.output_arg]
        )
        return opdef_sig, opdef_py_func


op_def_lib = MetaOpDefLibrary()


def _metatize_tf_object(obj):
    try:
        tf_obj = tf.convert_to_tensor(obj)
    except (TypeError, ValueError):
        raise ValueError(f"Error converting {obj} to a TensorFlow tensor.")

    return _metatize(tf_obj)


def load_dispatcher():
    """Set/override dispatcher to default to TF objects."""

    from tensorflow.python.ops.gen_linalg_ops import _SvdOutput

    def _metatize_tf_svd(obj):
        """Turn a TensorFlow `Svd` object/tuple into a standard tuple."""
        return meta._metatize(tuple(obj))

    meta._metatize.add((_SvdOutput,), _metatize_tf_svd)

    def _metatize_tf_eager(obj):
        """Catch eager tensor metatize issues early."""
        raise AttributeError(
            f"TensorFlow Operation not available; "
            "try recreating the object with eager-mode disabled"
            " (e.g. within `tensorflow.python.eager.context.graph_mode`)"
        )

    meta._metatize.add((ops.EagerTensor,), _metatize_tf_eager)

    meta._metatize.add((object,), _metatize_tf_object)
    meta._metatize.add((HashableNDArray,), _metatize_tf_object)

    for new_cls in TFlowMetaSymbol.base_subclasses():
        meta._metatize.add((new_cls.base,), new_cls._metatize)

    meta._metatize.add((TFlowMetaOpDef.base,), TFlowMetaOpDef._metatize)

    # Apply TF-specific `kanren` settings
    from ..relations import tensorflow

    return meta._metatize


class TFlowMetaSymbol(MetaSymbol):
    __slots__ = ()

    @classmethod
    def _metatize(cls, obj):

        res = super()._metatize(obj)
        res.validate_objs()

        return res

    def validate_objs(self):
        # If there is no base object associated with the inputs, then we can't
        # trust a base object associated with this object (e.g. for the case in
        # which metatize altered a property in an input).
        try:
            rands = self.rands
        except NotImplementedError:
            return

        for prop in rands:
            if isinstance(prop, MetaSymbol) and prop.obj is None:
                self.reset()
                break


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


class TFlowMetaOpDef(TFlowMetaSymbol, metaclass=OpDefFactoryType):
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
    __slots__ = ("_attr",)

    def __init__(self, obj=None):
        super().__init__(obj=obj)
        self._attr = {o.name: o for o in obj.attr}

    @property
    def attr(self):
        return self._attr

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

        assert self.base == other.base

        return self.obj.name == other.obj.name

    def __hash__(self):
        return hash((self.base, self.obj.name))

    def reify(self):
        return self.obj


class TFlowMetaNodeDef(TFlowMetaSymbol):
    """A meta `NodeDef`.

    NOTE: We're ignoring `node_def.input`; it's just an unnecessary hassle.
    """

    base = NodeDef
    __slots__ = ["op", "name", "attr", "_frozen_attr"]

    @classmethod
    def _metatize(cls, obj):
        res = super()._metatize(obj)

        if obj.op != "Const" and "node_attrs" in meta._lvar_defaults_enabled:
            res.attr = var()

        if "names" in meta._lvar_defaults_enabled:
            res.name = var()

        return res

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
            return tensor_util.MakeNdarray(v.tensor).view(HashableNDArray)
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
            self._frozen_attr = frozenset(self.attr.items())
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
        res = cls(*new_args, obj=obj)

        res.validate_objs()

        return res

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
        """Compute outputs for this meta `Operation`."""
        if getattr(self, "_outputs", None) is not None:
            return self._outputs

        if isvar(self.op_def):
            self._outputs = var()
        else:

            if isvar(self.node_def) or not isinstance(getattr(self.node_def, "attr", None), dict):
                node_attr = {}
            else:
                node_attr = self.node_def.attr

            operator = TFlowMetaOperator(self.op_def, self.node_def)

            if isvar(self.inputs):
                inputs = (None,) * len(operator._apply_func_sig.parameters)
                apply_defaults = False
            else:
                inputs = self.inputs
                apply_defaults = True

            apply_arguments = operator.input_args(
                *inputs, apply_defaults=apply_defaults, **node_attr
            )

            # TODO: The above could probably be simplified into a
            # NodeDef-from-input-args function.
            out_types_mt = operator.output_meta_types(inputs=apply_arguments)

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

        if isinstance(mt_outs, Sequence) and len(mt_outs) == 1:
            out_var = mt_outs[0]
        else:
            out_var = mt_outs

        self._default_output = out_var
        return self._default_output

    def reify(self):
        if self.obj and not isinstance(self.obj, Var):
            return self.obj

        if isvar(self.inputs):
            return self

        op_inputs, op_inputs_unreified = meta_reify_iter(self.inputs)

        node_attr = getattr(self.node_def, "attr", None)

        if node_attr is None or isvar(node_attr):
            return self

        operator = TFlowMetaOperator(self.op_def, self.node_def)

        op_attrs, op_attrs_unreified = meta_reify_iter(
            # Only use NodeDef attrs that appear in the OpDef's call signature.
            # Other NodeDef attrs, like dtype and shape, can be computed.
            {k: v for k, v in node_attr.items() if k in operator._apply_func_sig.parameters}
        )

        if not (op_inputs_unreified or op_attrs_unreified or isvar(self.name)):
            #
            # An operation with this name might already exist in the graph
            #
            try:
                existing_op = ops.get_default_graph().get_operation_by_name(self.name)
            except KeyError:
                #
                # There is no such `Operation`, so we attempt to create it
                #
                apply_arguments = operator.input_args(*op_inputs, name=self.name, **op_attrs)
                tf_out = operator._apply_func(**apply_arguments)
                op_tf = tf_out.op
            else:
                #
                # An `Operation` with this name exists, let's make sure it's
                # equivalent to this meta `Operation`
                #
                if self != mt(existing_op):
                    raise MetaReificationError(
                        f"An Operation with the name {self.name}"
                        " already exists in the graph and is not"
                        " equal to this meta object."
                    )
                op_tf = existing_op

            assert op_tf is not None
            self._obj = op_tf
            return self.obj

        return self


class TFlowMetaTensor(TFlowMetaSymbol, MetaVariable):
    base = tf.Tensor
    __slots__ = ("op", "value_index", "dtype", "_shape", "_name", "_operator")

    @classmethod
    @cachedmethod(lambda cls: tf_metatize_cache)
    def _metatize(cls, obj):
        """Cache Tensors specifically."""
        return super()._metatize(obj)

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

        if isinstance(getattr(self.op, "name", None), str) and not isvar(self.value_index):
            name = f"{self.op.name}:{self.value_index}"
        else:
            name = var()

        if self.obj is not None and not isinstance(self.obj, Var):
            assert name == self.obj.name

        self._name = name
        return self._name

    @property
    def base_operator(self):
        if getattr(self, "_operator", None):
            return self._operator

        if isvar(self.op) or (not isvar(self.op.inputs) and len(self.op.inputs) == 0):
            raise NotImplementedError(f"{self} does not have a base_operator.")

        self._operator = TFlowMetaOperator(self.op.op_def, self.op.node_def)

        return self._operator

    @property
    def base_arguments(self):
        # TODO: In keeping with our desire to return logic variables in cases
        # where params aren't given/inferred, we could return something like
        # `cons(var(), var())` here (although that wouldn't be necessarily
        # imply that the result is a proper list/tuple).
        if isvar(self.op) or (not isvar(self.op.inputs) and len(self.op.inputs) == 0):
            raise NotImplementedError(f"{self} does not have base arguments.")

        return self.op.inputs

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

    def __truediv__(self, y):
        # TODO: TF performs some dtype logic (using `dtype.base_dtype`) and casting here.
        return mt.realdiv(self, y, name="truediv")

    def __rtruediv__(self, x):
        # TODO: TF performs some dtype logic (using `dtype.base_dtype`) and casting here.
        return mt.realdiv(x, self, name="truediv")

    def __add__(self, y):
        # TODO: If `self.dtype == tf.dtypes.string`, use `mt.add`
        return mt.addv2(self, y, name="add")

    def __radd__(self, x):
        # TODO: If `x.dtype == tf.dtypes.string`, use `mt.add`
        return mt.addv2(x, self, name="add")

    def __sub__(self, y):
        return mt.sub(self, y, name="sub")

    def __rsub__(self, x):
        return mt.sub(x, self, name="sub")

    def __mul__(self, y):
        return mt.mul(self, y, name="mul")

    def __rmul__(self, x):
        return mt.mul(x, self, name="mul")

    def __abs__(self):
        return mt.abs(self, name="Abs")

    def __pow__(self, y):
        return mt.pow(self, y, name="pow")

    def __neg__(self):
        return mt.neg(self, name="Neg")


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


class TFlowMetaOperator(TFlowMetaSymbol, MetaOp):
    """A class that implements the notion of an operator on top of TensorFlow's OpDef and NodeDef objects.

    With this abstraction, we can better model operators by distinguishing
    parameterized operators and their respective parameter values from the
    operator's inputs, which may have similar properties across the entire
    family of operators (i.e. across all parameter values).

    For example, addition is commutative in its arguments, so modeling addition
    as an operator parameterized on dtypes and/or names, we may want to
    preserve the distinction of the operators inputs and its parameterized
    values so that we can implement commutativity exclusively on the
    non-dtype/name inputs.
    """

    base = None
    __slots__ = ("op_def", "node_def", "_apply_func_sig", "_apply_func")

    @classmethod
    def get_metaopdef(cls, name):
        """Obtain a MetaOpDef for a given string name.

        This is more flexible because it ignores things like string case
        (when the non-`raw_ops` name differs from the TF user-level API).
        """
        raw_op_name = op_def_lib.lower_op_name_to_raw.get(name.lower(), name)
        op_def = op_def_registry.get(raw_op_name)
        if op_def is not None:
            return TFlowMetaOpDef(obj=op_def)

    def __init__(self, op_def, node_def=None, obj=None):
        assert obj is None
        super().__init__(None)

        self.op_def = op_def

        if isinstance(self.op_def, str):
            self.op_def = self.get_metaopdef(self.op_def)

        if self.op_def is None:
            raise ValueError(f"Could not find an OpDef for {op_def}")

        if isvar(self.op_def):
            self._apply_func_sig, self._apply_func = None, None
        else:
            self._apply_func_sig, self._apply_func = op_def_lib.get_op_info(self.op_def.obj)

        self.node_def = node_def

    def reify(self):
        return self

    def output_meta_types(self, inputs=None):
        """Return a list of tuples containing object types and corresponding dtypes for the outputs of this OpDef.

        This work is done in
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/op_def_library.py#L337.
        Would be very nice if the dtype inference and/or `NodeDef` generation
        was abstracted-out from that function.
        """

        if isvar(self.op_def):
            return None

        type_map = {k: None for k, v in self.op_def.attr.items() if v.type == "type"}

        input_type_map = {i.name: i.type_attr for i in self.op_def.obj.input_arg}

        if inputs:
            # Infer/verify types from inputs
            for i_name, i_value in inputs.items():
                type_name = input_type_map.get(i_name, None)

                if type_name is None:
                    continue

                i_dtype = getattr(i_value, "dtype", None)

                dtype = type_map[type_name]

                if i_dtype is not None and not isvar(i_dtype):
                    if dtype is None or isvar(dtype):
                        # The input dtype is more informative, so use it.
                        type_map[type_name] = i_dtype
                    else:
                        # They're both informative and should be the same
                        assert dtype == i_dtype

        def infer_output_types(o):
            """Get dtypes for specific outputs using known dtype info and API inputs."""
            # Get the explicit type from a `NodeDef` attribute.
            type_name = o.type_attr

            # Existing dtype value
            dtype = type_map[type_name]
            # New value
            _dtype = None

            # The information could be in the `NodeDef`
            if isinstance(getattr(self.node_def, "attr", None), dict):
                _dtype = self.node_def.attr.get(type_name)

            # It could also be in the inputs (i.e. when called via the API
            # route)
            # TODO: If it's in the inputs, we should just create the
            # corresponding `NodeDef`.
            if inputs and type_name in inputs:
                _dtype = inputs.get(type_name)

            if _dtype is None:
                _dtype = var()

            if not isvar(_dtype):
                try:
                    _dtype = tf.dtypes.as_dtype(_dtype).base_dtype
                except TypeError:
                    _dtype = var()

            # Make sure dtypes derived from `NodeDef` info and API inputs are
            # consistent.
            if dtype is None or isvar(dtype):
                # Newly inferred dtype is at least as informative as the
                # current one
                dtype = _dtype
                type_map[type_name] = dtype
            elif _dtype is None or isvar(_dtype):
                # Newly inferred dtype is less informative
                pass
            else:
                assert dtype == _dtype

            return (TFlowMetaTensor, dtype)

        # TODO: We could update missing dtype information in input meta
        # types and `NodeDef`s.

        # TODO: We also have permissible dtype information from objects in the
        # array `self.obj.attr` under the field `allowed_values`.

        return tuple(infer_output_types(o) for o in self.op_def.obj.output_arg)

    def input_args(self, *args, apply_defaults=True, **kwargs):
        """Make args and kwargs conform to an OpDef's "apply function" arguments.

        In order to do this, we effectively need to map `OpDef` and `NodeDef`
        values to `tf.raw_ops.*` function arguments (i.e. the reverse of what
        `op_def_library._apply_op_helper` does).

        Returns an `OrderedDict`.

        """
        kwargs = OrderedDict(
            (k, v)
            for k, v in kwargs.items()
            # Filter out the optional keyword arguments so they we only pass
            # expected arguments to the `OpDef`'s apply function.
            if k in self._apply_func_sig.parameters
        )

        op_args = self._apply_func_sig.bind_partial(*args, **kwargs)

        if apply_defaults:
            op_args.apply_defaults()

        return op_args.arguments

    def __api_call__(self, *args, **kwargs):
        """Create the meta object(s) using the TF Python API's operator functions.

        Each meta `OpDef` is associated with a TF Python function
        (`self._apply_func`) that is used to construct its `Operation`s.

        See `TFlowMetaTensor.operator` and `TFlowMetaTensor.operator`.

        """

        apply_arguments = self.input_args(*args, **kwargs)

        if not meta._auto_reification_disabled:
            op_args, op_args_unreified = meta_reify_iter(apply_arguments)
        else:
            op_args, op_args_unreified = apply_arguments, True

        if not op_args_unreified:

            res_var = None
            # name = op_args.get("name", None)
            #
            # if name is not None:
            #     #
            #     # An operation with this name might already exist in the graph
            #     #
            #
            #     from tensorflow.python.framework import ops
            #
            #     try:
            #         this_op = ops.get_default_graph().get_operation_by_name(name)
            #     except KeyError:
            #         pass
            #     else:
            #         # TODO: Make sure the existing `Operation` matches our arguments
            #         assert this_op.type == self.op_def.obj.name
            #
            #         this_op = mt(this_op)
            #         op_inputs, op_node_def = self.op_args_to_operation_inputs(op_args)
            #         assert op_inputs == this_op.inputs
            #         assert op_node_def == this_op.node_def
            #         res_var = this_op.default_output

            if res_var is None:
                #
                # We create the `Operation` in the graph
                #

                tf_out = self._apply_func(**op_args)

                # Ensure that the original meta objects will be available
                # for use in the `metatize` that follows
                tf_metatize_cache.update(
                    {
                        k: v
                        for k, v in zip(op_args.values(), apply_arguments.values())
                        if isinstance(k, tf.Tensor)
                    }
                )

                res_var = metatize(tf_out)

            if "names" in meta._lvar_defaults_enabled:
                # This should also reset the NodeDef's `obj`
                res_var.op.node_def.name = var()
                res_var.op.reset()
                res_var.reset()

            if "node_attrs" in meta._lvar_defaults_enabled:
                # This should also reset the NodeDef's `obj`
                res_var.op.node_def.attr = var()
                res_var.op.reset()
                res_var.reset()
        else:
            #
            # If we're here, we have to create the meta objects manually.
            #

            op_input_args, node_def = self.op_args_to_operation_inputs(apply_arguments)

            op_mt = TFlowMetaOp(self.op_def, node_def, op_input_args)

            res_var = op_mt.default_output

        return res_var

    def op_args_to_operation_inputs(self, apply_arguments):
        """Map an `OpDef`'s "apply function" arguments to `Operation` inputs and a meta `NodeDef`."""

        if isvar(self.op_def):
            return None

        op_def_tf = self.op_def.obj

        op_inputs = tuple(
            apply_arguments.get(i.name) for i in op_def_tf.input_arg if i.name in apply_arguments
        )

        # TODO: Include inferred attr values (e.g. dtypes).
        if "node_attrs" not in meta._lvar_defaults_enabled:
            node_attr = {a.name: apply_arguments.get(a.name, a) for a in op_def_tf.attr}
        else:
            node_attr = var()

        if "names" not in meta._lvar_defaults_enabled:
            op_name = apply_arguments.get("name", op_def_tf.name) or op_def_tf.name
        else:
            op_name = var()

        node_def = TFlowMetaNodeDef(op_def_tf.name, op_name, node_attr)

        return op_inputs, node_def

    def __call__(self, *inputs, **kwargs):

        if self.node_def is not None:
            op = TFlowMetaOp(self.op_def, self.node_def, inputs)

            res = op.default_output

            if isvar(res):
                return op.outputs
            else:
                return res
        else:
            return self.__api_call__(*inputs, **kwargs)


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

        if isinstance(ns_obj, types.ModuleType):
            # It's a sub-module, so let's create another
            # `TheanoMetaAccessor` and check within there.
            meta_obj = TFlowMetaAccessor(namespace=ns_obj)
        else:

            # Check for a an OpDef first
            meta_obj = TFlowMetaOperator.get_metaopdef(obj)

            if meta_obj is not None:
                # We assume that the user requested an `Operation`
                # constructor/helper.  Return the meta `OpDef`, because
                # it implements a constructor/helper-like `__call__`.
                if meta_obj is not None:
                    meta_obj = TFlowMetaOperator(meta_obj, None)

            # elif isinstance(ns_obj, (types.FunctionType, partial)):
            #     # It's a function, so let's provide a wrapper that converts
            #     # to-and-from theano and meta objects.
            #     @wraps(ns_obj)
            #     def meta_obj(*args, **kwargs):
            #         args = [o.reify() if hasattr(o, "reify") else o for o in args]
            #         res = ns_obj(*args, **kwargs)
            #         return metatize(res)

            else:
                # Hopefully, it's convertible to a meta object...
                meta_obj = metatize(ns_obj)

        # Finally, we store the result as a meta namespace attribute, or raise
        # an exception.
        if isinstance(
            meta_obj, (MetaSymbol, MetaSymbolType, TFlowMetaOperator, types.FunctionType)
        ):
            setattr(self, obj, meta_obj)
            return getattr(self, obj)
        elif isinstance(meta_obj, TFlowMetaAccessor):
            setattr(self, obj, meta_obj)
            return meta_obj
        else:
            raise AttributeError(f"Meta object for {obj} not found.")


mt = TFlowMetaAccessor()


load_dispatcher()
