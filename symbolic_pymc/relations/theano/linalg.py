import toolz

from operator import itemgetter, attrgetter

from theano.tensor.nlinalg import QRFull

from unification import var

from kanren import eq
from kanren.core import lall
from kanren.goals import not_equalo, conso

from ...theano.meta import mt
from ...unify import etuple, etuplize
from .. import buildo


mt.nlinalg.qr_full = mt(QRFull("reduced"))
owner_inputs = attrgetter("owner.inputs")
normal_get_size = toolz.compose(itemgetter(2), owner_inputs)
normal_get_rng = toolz.compose(itemgetter(3), owner_inputs)


def update_name_suffix(x, old_x, suffix):
    new_name = old_x.name + suffix
    x.name = new_name
    return x


def normal_normal_regression(Y, X, beta, Y_args_tail=None, beta_args=None):
    """Produce a relation for a normal-normal regression of the form `Y ~ N(X * beta, sd**2)`."""
    Y_args_tail = Y_args_tail or var()
    Y_args = var()
    beta_args = beta_args or var()
    Y_mean_lv = var()
    dot_args_lv = var()

    res = (
        lall,
        # `Y` is a `NormalRV`
        (buildo, mt.NormalRV, Y_args, Y),
        # `beta` is also a `NormalRV`
        (buildo, mt.NormalRV, beta_args, beta),
        # Obtain its mean parameter and remaining args
        (conso, Y_mean_lv, Y_args_tail, Y_args),
        (eq, dot_args_lv, etuple(X, beta)),
        # Relate it to a dot product of `X` and `beta`
        (buildo, mt.dot, dot_args_lv, Y_mean_lv),
    )

    return res


def normal_qr_transform(in_expr, out_expr):
    """Produce a relation for normal-normal regression and its QR-reduced form.

    TODO XXX: This isn't entirely correct (e.g. it needs to also
    transform the variance terms), but it demonstrates all the requisite
    functionality for this kind of model reformulation.

    """
    y_lv, Y_lv, X_lv, beta_lv = var(), var(), var(), var()
    Y_args_lv, beta_args_lv = var(), var()
    QR_lv, Q_lv, R_lv = var(), var(), var()
    beta_til_lv, beta_new_lv = var(), var()
    beta_mean_lv, beta_sd_lv = var(), var()
    beta_size_lv, beta_rng_lv = var(), var()
    Y_new_lv = var()
    X_op_lv = var()

    in_expr = etuplize(in_expr)

    res = (
        lall,
        # Only applies to regression models on observed RVs
        (eq, in_expr, etuple(mt.observed, y_lv, Y_lv)),
        # Relate the model components
        (normal_normal_regression, Y_lv, X_lv, beta_lv, Y_args_lv, beta_args_lv),
        # Let's not do all this to an already QR-reduce graph;
        # otherwise, we'll loop forever!
        (buildo, X_op_lv, var(), X_lv),
        # XXX: This type of dis-equality goal isn't the best,
        # but it will definitely work for now.
        (not_equalo, (mt.nlinalg.qr_full, X_op_lv), True),
        # Relate terms for the QR decomposition
        (eq, QR_lv, etuple(mt.nlinalg.qr_full, X_lv)),
        (eq, Q_lv, etuple(itemgetter(0), QR_lv)),
        (eq, R_lv, etuple(itemgetter(1), QR_lv)),
        # The new `beta_tilde`
        (eq, beta_args_lv, (beta_mean_lv, beta_sd_lv, beta_size_lv, beta_rng_lv)),
        (
            eq,
            beta_til_lv,
            etuple(
                mt.NormalRV,
                # Use these `tt.[ones|zeros]_like` functions to preserve the
                # correct shape (and a valid `tt.dot`).
                etuple(mt.zeros_like, beta_mean_lv),
                etuple(mt.ones_like, beta_sd_lv),
                beta_size_lv,
                beta_rng_lv,
            ),
        ),
        # Relate the new and old coeffs
        (eq, beta_new_lv, etuple(mt.dot, etuple(mt.nlinalg.matrix_inverse, R_lv), beta_til_lv)),
        # Use the relation the other way to produce the new/transformed
        # observation distribution
        (normal_normal_regression, Y_new_lv, Q_lv, beta_til_lv, Y_args_lv),
        (
            eq,
            out_expr,
            [
                (
                    in_expr,
                    etuple(mt.observed, y_lv, etuple(update_name_suffix, Y_new_lv, Y_lv, "")),
                ),
                (beta_lv, beta_new_lv),
            ],
        ),
    )
    return res
