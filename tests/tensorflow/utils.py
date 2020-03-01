import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops

from collections.abc import Mapping

from symbolic_pymc.tensorflow.meta import mt


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


def tfp_normal_log_prob(x, loc, scale):
    """Create a graph of the Grappler-canonicalized form of a TFP normal log-likelihood."""
    log_unnormalized = -0.5 * tf.math.squared_difference(x / scale, loc / scale)
    log_normalization = 0.5 * np.log(2.0 * np.pi) + tf.math.log(scale)
    return log_unnormalized - log_normalization


def mt_normal_log_prob(x, loc, scale):
    """Create a meta graph for Grappler-canonicalized standard or non-standard TFP normal log-likelihoods."""
    if loc == 0:
        log_unnormalized_mt = mt(np.array(-0.5, "float32"))
        log_unnormalized_mt *= mt.squareddifference(
            mt(np.array(0.0, "float32")),
            mt.realdiv(x, scale) if scale != 1 else mt.mul(np.array(1.0, "float32"), x),
        )
    else:
        log_unnormalized_mt = mt(np.array(-0.5, "float32"))
        log_unnormalized_mt *= mt.squareddifference(
            mt.realdiv(x, scale) if scale != 1 else mt.mul(np.array(1.0, "float32"), x),
            mt.realdiv(loc, scale) if scale != 1 else mt.mul(np.array(1.0, "float32"), loc),
        )

    log_normalization_mt = mt((0.5 * np.log(2.0 * np.pi)).astype("float32"))

    if scale != 1:
        log_normalization_mt = log_normalization_mt + mt.log(scale)

    return log_unnormalized_mt - log_normalization_mt
