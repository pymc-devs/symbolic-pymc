import tensorflow as tf

from multipledispatch import dispatch

from ..meta import MetaSymbol, MetaOp, MetaVariable, _metatize


@dispatch(object)
def _metatize(obj):
    try:
        obj = tf.convert_to_tensor(obj)
    except TypeError:
        raise ValueError("Could not find a TensorFlow MetaSymbol class for {}".format(obj))
    return _metatize(obj)


class TFlowMetaSymbol(MetaSymbol):
    def reify(self):
        # super().reify()

        # TODO: Follow `tfp.distribution.Distribution`'s lead?
        # with tf.name_scope(self.name):
        #     pass
        pass


class TFlowMetaOp(MetaOp, TFlowMetaSymbol):
    base = tf.Operation

    def __call__(self, *args, ttype=None, index=None, **kwargs):
        pass


class TFTensorVariable(MetaVariable, TFlowMetaSymbol):
    base = tf.Tensor
    __slots__ = ["op", "value_index", "dtype"]

    def __init__(self, op, value_index, dtype):
        self.op = op
        self.value_index = value_index
        self.dtype = dtype
