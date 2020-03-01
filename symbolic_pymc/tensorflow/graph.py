import tensorflow as tf

from tensorflow.core.protobuf import config_pb2

from tensorflow.python.framework import ops
from tensorflow.python.framework import importer
from tensorflow.python.framework import meta_graph

from tensorflow.python.grappler import cluster
from tensorflow.python.grappler import tf_optimizer


try:  # pragma: no cover
    gcluster = cluster.Cluster()
except tf.errors.UnavailableError:  # pragma: no cover
    pass

config = config_pb2.ConfigProto()


def normalize_tf_graph(graph_output, new_graph=True, verbose=False):
    """Use grappler to normalize a graph.

    Arguments
    ---------
    graph_output: Tensor
      A tensor we want to consider as "output" of a `FuncGraph`.

    Returns
    -------
    The simplified graph.
    """
    train_op = graph_output.graph.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.clear()
    train_op.extend([graph_output])

    metagraph = meta_graph.create_meta_graph_def(graph=graph_output.graph)

    optimized_graphdef = tf_optimizer.OptimizeGraph(
        config, metagraph, verbose=verbose, cluster=gcluster
    )

    output_name = graph_output.name

    if new_graph:
        optimized_graph = ops.Graph()
    else:  # pragma: no cover
        optimized_graph = ops.get_default_graph()
        del graph_output

    with optimized_graph.as_default():
        importer.import_graph_def(optimized_graphdef, name="")

    opt_graph_output = optimized_graph.get_tensor_by_name(output_name)

    return opt_graph_output
