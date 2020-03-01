import tensorflow as tf

from tensorflow.python.framework.ops import disable_tensor_equality

from tensorflow_probability import distributions as tfd

from unification import var, unify

from kanren import run, eq, lall
from kanren.graph import walko
from kanren.assoccomm import eq_comm, commutative

from symbolic_pymc.meta import enable_lvar_defaults
from symbolic_pymc.tensorflow.meta import mt
from symbolic_pymc.tensorflow.graph import normalize_tf_graph

from tests.tensorflow import run_in_graph_mode
from tests.tensorflow.utils import mt_normal_log_prob

disable_tensor_equality()


@run_in_graph_mode
def test_walko():
    with enable_lvar_defaults("names"):
        add_1_mt = mt(1) + mt(2)

    def walk_rel(x, y):
        return lall(eq(x, mt(1)), eq(y, mt(3)))

    q = var()
    (res,) = run(1, q, walko(walk_rel, add_1_mt, q))

    # The easiest way to check whether or not two arbitrary TF meta graphs are
    # (structurally) equivalent is to confirm that they unify.  This avoids
    # uninteresting differences in node names, uninferred type information,
    # etc.
    with enable_lvar_defaults("names", "node_attrs"):
        assert unify(res.eval_obj, mt(3) + mt(2)) is not False


@run_in_graph_mode
def test_commutativity():
    with enable_lvar_defaults("names"):
        add_1_mt = mt(1) + mt(2)
        add_2_mt = mt(2) + mt(1)

    q = var()
    res = run(0, q, commutative(add_1_mt.base_operator))
    assert res is not False

    res = run(0, q, eq_comm(add_1_mt, add_2_mt))
    assert res is not False

    with enable_lvar_defaults("names"):
        add_pattern_mt = mt(2) + q

    res = run(0, q, eq_comm(add_1_mt, add_pattern_mt))
    assert res[0] == add_1_mt.base_arguments[0]


@run_in_graph_mode
def test_commutativity_tfp():

    with tf.Graph().as_default():
        mu_tf = tf.compat.v1.placeholder(tf.float32, name="mu", shape=tf.TensorShape([None]))
        tau_tf = tf.compat.v1.placeholder(tf.float32, name="tau", shape=tf.TensorShape([None]))

        normal_tfp = tfd.normal.Normal(mu_tf, tau_tf)

        value_tf = tf.compat.v1.placeholder(tf.float32, name="value", shape=tf.TensorShape([None]))

        normal_log_lik = normal_tfp.log_prob(value_tf)

    normal_log_lik_opt = normalize_tf_graph(normal_log_lik)

    with enable_lvar_defaults("names", "node_attrs"):
        tfp_normal_pattern_mt = mt_normal_log_prob(var(), var(), var())

    normal_log_lik_mt = mt(normal_log_lik)
    normal_log_lik_opt_mt = mt(normal_log_lik_opt)

    # Our pattern is the form of an unnormalized TFP normal PDF.
    assert run(0, True, eq(normal_log_lik_mt, tfp_normal_pattern_mt)) == (True,)
    # Our pattern should *not* match the Grappler-optimized graph, because
    # Grappler will reorder terms (e.g. the log + constant
    # variance/normalization term)
    assert run(0, True, eq(normal_log_lik_opt_mt, tfp_normal_pattern_mt)) == ()

    # XXX: `eq_comm` is, unfortunately, order sensitive!  LHS should be ground.
    assert run(0, True, eq_comm(normal_log_lik_mt, tfp_normal_pattern_mt)) == (True,)
    assert run(0, True, eq_comm(normal_log_lik_opt_mt, tfp_normal_pattern_mt)) == (True,)
