import numpy as np
import tensorflow as tf

from symbolic_pymc.tensorflow.graph import normalize_tf_graph

from tests.tensorflow import run_in_graph_mode


@run_in_graph_mode
def test_normalize():

    tf.config.optimizer.set_experimental_options(
        {
            "shape_optimizations": True,
            "arithmetic_optimzation": True,
            "function_optimization": True,
            "min_graph_nodes": 0,
        }
    )
    with tf.Graph().as_default() as norm_graph:
        a_tf = tf.compat.v1.placeholder("float")
        const_log_tf = 0.5 * np.log(2.0 * np.pi) + tf.math.log(a_tf)
        normal_const_log_tf = normalize_tf_graph(const_log_tf)

        # Grappler appears to put log ops before const
        assert normal_const_log_tf.op.inputs[0].op.type == "Log"
        assert normal_const_log_tf.op.inputs[1].op.type == "Const"
