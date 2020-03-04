import sys

import numpy as np
import tensorflow as tf

from functools import singledispatch
from contextlib import contextmanager

from unification import isvar

# from tensorflow.python.framework import tensor_util

from symbolic_pymc.tensorflow.meta import TFlowMetaTensor
from symbolic_pymc.tensorflow.meta import TFlowMetaOp


class DepthExceededException(Exception):
    pass


class TFlowPrinter(object):
    """A printer that indents and keeps track of already printed subgraphs."""

    def __init__(self, formatter, buffer, depth_lower_idx=0, depth_upper_idx=sys.maxsize):
        # The buffer to which results are printed
        self.buffer = buffer
        # A function used to pre-process printed results
        self.formatter = formatter

        self.depth_count = 0
        self.depth_lower_idx, self.depth_upper_idx = depth_lower_idx, depth_upper_idx

        # This is the current indentation string
        if self.depth_lower_idx > 0:
            self.indentation = "... "
        else:
            self.indentation = ""

        # The set of graphs that have already been printed
        self.printed_subgraphs = set()

    @contextmanager
    def indented(self, indent):
        pre_indentation = self.indentation

        self.depth_count += 1

        if self.depth_lower_idx < self.depth_count <= self.depth_upper_idx:
            self.indentation += indent

        try:
            yield
        except DepthExceededException:
            pass
        finally:
            self.indentation = pre_indentation
            self.depth_count -= 1

    def format(self, obj):
        return self.indentation + self.formatter(obj)

    def print(self, obj, suffix=""):
        if self.depth_lower_idx <= self.depth_count < self.depth_upper_idx:
            self.buffer.write(self.format(obj) + suffix)
            self.buffer.flush()
        elif self.depth_count == self.depth_upper_idx:
            # Only print the cut-off indicator at the first occurrence
            self.buffer.write(self.format(f"...{suffix}"))
            self.buffer.flush()

            # Prevent the caller from traversing at this level or higher
            raise DepthExceededException()

    def println(self, obj):
        self.print(obj, suffix="\n")

    def subgraph_add(self, obj):
        if self.depth_lower_idx <= self.depth_count < self.depth_upper_idx:
            # Only track printed subgraphs when they're actually printed
            self.printed_subgraphs.add(obj)

    def __repr__(self):  # pragma: no cover
        return (
            "TFlowPrinter\n"
            f"\tdepth_lower_idx={self.depth_lower_idx},\tdepth_upper_idx={self.depth_upper_idx}\n"
            f"\tindentation='{self.indentation}',\tdepth_count={self.depth_count}"
        )


def tf_dprint(obj, depth_lower=0, depth_upper=10, printer=None):
    """Print a textual representation of a TF graph. The output roughly follows the format of `theano.printing.debugprint`.

    Parameters
    ----------
    obj : Tensorflow object
        Tensorflow graph object to be represented.
    depth_lower : int
        Used to index specific portions of the graph.
    depth_upper : int
        Used to index specific portions of the graph.
    printer : optional
        Backend used to display the output.

    """

    if isinstance(obj, tf.Tensor):
        try:
            obj.op
        except AttributeError:
            raise ValueError(
                f"TensorFlow Operation not available; "
                "try recreating the object with eager-mode disabled"
                " (e.g. within `tensorflow.python.eager.context.graph_mode`)"
            )

    if printer is None:
        printer = TFlowPrinter(str, sys.stdout, depth_lower, depth_upper)

    _tf_dprint(obj, printer)


@singledispatch
def _tf_dprint(obj, printer):
    printer.println(obj)


@_tf_dprint.register(tf.Tensor)
@_tf_dprint.register(TFlowMetaTensor)
def _tf_dprint_TFlowMetaTensor(obj, printer):

    try:
        shape_str = str(obj.shape.as_list())
    except (ValueError, AttributeError):
        shape_str = "Unknown"

    prefix = f'Tensor({getattr(obj.op, "type", obj.op)}):{obj.value_index},\tdtype={getattr(obj.dtype, "name", obj.dtype)},\tshape={shape_str},\t"{obj.name}"'

    _tf_dprint(prefix, printer)

    if isvar(obj.op):
        return
    elif isvar(obj.op.inputs):
        with printer.indented("|  "):
            _tf_dprint(f"{obj.op.inputs}", printer)
    elif obj.op.type == "Const":
        with printer.indented("|  "):
            if isinstance(obj, tf.Tensor):
                numpy_val = obj.eval(session=tf.compat.v1.Session(graph=obj.graph))
            elif isvar(obj.op.node_def):
                _tf_dprint(f"{obj.op.node_def}", printer)
                return
            else:
                numpy_val = obj.op.node_def.attr["value"]

            _tf_dprint(
                np.array2string(numpy_val, threshold=20, prefix=printer.indentation), printer
            )
    elif len(obj.op.inputs) > 0:
        with printer.indented("|  "):
            if obj in printer.printed_subgraphs:
                _tf_dprint("...", printer)
            else:
                printer.subgraph_add(obj)
                _tf_dprint(obj.op, printer)


@_tf_dprint.register(tf.Operation)
@_tf_dprint.register(TFlowMetaOp)
def _tf_dprint_TFlowMetaOp(obj, printer):
    for op_input in obj.inputs:
        _tf_dprint(op_input, printer)
