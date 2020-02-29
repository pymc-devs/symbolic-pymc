from collections.abc import Mapping

import tensorflow as tf
from tensorflow.python.framework import ops


def assert_ops_equal(a, b, compare_fn=lambda a, b: a.op.type == b.op.type):
    if hasattr(a, "op") or hasattr(b, "op"):
        assert hasattr(a, "op") and hasattr(b, "op")

        assert compare_fn(a, b)

        if isinstance(a.op, tf.Operation):
            a_inputs = ops._reconstruct_sequence_inputs(
                a.op.op_def, a.op.inputs, a.op.node_def.attr
            )
        elif isinstance(a.op, Mapping):
            a_inputs = list(a.op.inputs.values())
        else:
            a_inputs = list(a.op.inputs)

        if isinstance(b.op, tf.Operation):
            b_inputs = ops._reconstruct_sequence_inputs(
                b.op.op_def, b.op.inputs, b.op.node_def.attr
            )
        elif isinstance(b.op, Mapping):
            b_inputs = list(b.op.inputs.values())
        else:
            b_inputs = list(b.op.inputs)

        assert len(a_inputs) == len(b_inputs)

        for i_a, i_b in zip(a_inputs, b_inputs):
            assert_ops_equal(i_a, i_b)
