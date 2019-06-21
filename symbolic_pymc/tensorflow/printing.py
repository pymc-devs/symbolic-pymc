import sys

from functools import singledispatch
from contextlib import contextmanager

import tensorflow as tf


class IndentPrinter(object):
    def __init__(self, formatter):
        self.formatter = formatter
        self.indentation = ""

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
        sys.stdout.write(self.format(obj))
        sys.stdout.flush()

    def println(self, obj):
        sys.stdout.write(self.format(obj) + "\n")
        sys.stdout.flush()


@singledispatch
def tf_dprint(obj, printer=IndentPrinter(str)):
    """Print a textual representation of a TF graph.

    The output roughly follows the format of `theano.printing.debugprint`.
    """
    printer.println(obj)


@tf_dprint.register(tf.Tensor)
def _(obj, printer=IndentPrinter(str)):
    try:
        shape_str = str(obj.shape.as_list())
    except ValueError:
        shape_str = "Unknown"
    prefix = f'Tensor({obj.op.type}):{obj.value_index},\tshape={shape_str}\t"{obj.name}"'
    tf_dprint(prefix, printer)
    if len(obj.op.inputs) > 0:
        with printer.indented("|  "):
            tf_dprint(obj.op, printer)


@tf_dprint.register(tf.Operation)
def _(obj, printer=IndentPrinter(str)):
    prefix = f'Op({obj.type})\t"{obj.name}"'
    tf_dprint(prefix, printer)
    with printer.indented("|  "):
        for op_input in obj.inputs:
            tf_dprint(op_input, printer)
