import sys

from functools import singledispatch
from contextlib import contextmanager

import tensorflow as tf

from symbolic_pymc.tensorflow.meta import TFlowMetaTensor
from symbolic_pymc.tensorflow.meta import TFlowMetaOp


class TFlowPrinter(object):
    """A printer that indents and keeps track of already printed subgraphs."""

    def __init__(self, formatter, buffer):
        self.buffer = buffer
        self.formatter = formatter
        self.indentation = ""
        self.printed_subgraphs = set()

    @contextmanager
    def indented(self, indent):
        pre_indentation = self.indentation
        if isinstance(indent, int):
            self.indentation += " " * indent
        else:
            self.indentation += indent
        try:
            yield
        finally:
            self.indentation = pre_indentation

    def format(self, obj):
        return self.indentation + self.formatter(obj)

    def print(self, obj):
        self.buffer.write(self.format(obj))
        self.buffer.flush()

    def println(self, obj):
        self.buffer.write(self.format(obj) + "\n")
        self.buffer.flush()


def tf_dprint(obj, printer=None):
    """Print a textual representation of a TF graph.

    The output roughly follows the format of `theano.printing.debugprint`.
    """
    if printer is None:
        printer = TFlowPrinter(str, sys.stdout)

    _tf_dprint(obj, printer)


@singledispatch
def _tf_dprint(obj, printer):
    printer.println(obj)


@_tf_dprint.register(tf.Tensor)
@_tf_dprint.register(TFlowMetaTensor)
def _(obj, printer):

    try:
        shape_str = str(obj.shape.as_list())
    except (ValueError, AttributeError):
        shape_str = "Unknown"

    prefix = f'Tensor({obj.op.type}):{obj.value_index},\tshape={shape_str}\t"{obj.name}"'
    _tf_dprint(prefix, printer)
    if len(obj.op.inputs) > 0:
        with printer.indented("|  "):
            if obj not in printer.printed_subgraphs:
                printer.printed_subgraphs.add(obj)
                _tf_dprint(obj.op, printer)
            else:
                _tf_dprint("...", printer)


@_tf_dprint.register(tf.Operation)
@_tf_dprint.register(TFlowMetaOp)
def _(obj, printer):
    prefix = f'Op({obj.type})\t"{obj.name}"'
    _tf_dprint(prefix, printer)
    with printer.indented("|  "):
        for op_input in obj.inputs:
            _tf_dprint(op_input, printer)
