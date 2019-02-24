import toolz

from operator import itemgetter, attrgetter

from theano.tensor.nlinalg import QRFull

from unification import var

from kanren import eq
from kanren.core import lall
from kanren.goals import heado, not_equalo
from kanren.assoccomm import buildo

from ..meta import mt
from ..unify import etuple, tuple_expression


mt.nlinalg.qr_full = mt(QRFull('reduced'))
owner_inputs = attrgetter('owner.inputs')
normal_get_size = toolz.compose(itemgetter(2), owner_inputs)
normal_get_rng = toolz.compose(itemgetter(3), owner_inputs)


def update_name_suffix(x, old_x, suffix):
    new_name = old_x.name + suffix
    x.name = new_name
    return x


def normal_normal_regression(Y, X, beta, sd=None, size=None, rng=None):
    """Relation for a normal-normal regression, i.e.

        Y ~ N(X * beta, )

    """
    mu_lv = var()
    sd = sd or var()
    size = size or var()
    rng = rng or var()

    res = (lall,
           (eq, Y, etuple(mt.NormalRV, mu_lv, sd, size, rng)),
           (eq, mu_lv, etuple(mt.dot, X, beta)),
           (heado, mt.NormalRV, beta),
    )

    return res


def normal_qr_transform(in_expr, out_expr):
    """Relation for normal-normal regression and its QR-reduced form.
    """
    y_lv, Y_lv, X_lv, beta_lv = var(), var(), var(), var()
    Y_sd_lv, Y_size_lv, Y_rng_lv = var(), var(), var()
    QR_lv, Q_lv, R_lv = var(), var(), var()
    beta_til_lv, beta_new_lv = var(), var()
    Y_new_lv = var()
    X_op_lv = var()

    in_expr = tuple_expression(in_expr)

    res = (
        lall,
        # Only applies to regression models on observed RVs
        (eq, in_expr, etuple(mt.observed, y_lv, Y_lv)),
        # Relate the model components
        (normal_normal_regression,
         Y_lv, X_lv, beta_lv,
         Y_sd_lv, Y_size_lv, Y_rng_lv),
        # Let's not do this to an already QR-reduce graph
        (buildo, X_op_lv, var(), X_lv),
        # XXX: This type of dis-equality goal isn't the best,
        # but it will definitely work for now.
        (not_equalo, (mt.nlinalg.qr_full, X_op_lv), True),
        # Relate terms for the QR decomposition
        (eq, QR_lv, etuple(mt.nlinalg.qr_full, X_lv)),
        (eq, Q_lv, etuple(itemgetter(0), QR_lv)),
        (eq, R_lv, etuple(itemgetter(1), QR_lv)),
        # Relate the transformed coeffs
        (eq, beta_til_lv,
         etuple(mt.NormalRV, 0., 1.,
                etuple(normal_get_size, beta_lv),
                etuple(normal_get_rng, beta_lv))),
        # Relate the new and old coeffs
        (eq, beta_new_lv,
         etuple(mt.dot,
                etuple(mt.nlinalg.matrix_inverse, R_lv),
                beta_til_lv)),
        # Use the relation the other way to produce the new/transformed
        # observation distribution
        (normal_normal_regression,
         Y_new_lv, Q_lv, beta_til_lv,
         Y_sd_lv, Y_size_lv, Y_rng_lv),
        (eq, out_expr,
         [
             (in_expr,
              etuple(mt.observed, y_lv, Y_new_lv)),
             (beta_lv,
              etuple(update_name_suffix,
                     beta_new_lv,
                     beta_lv,
                     '_tilde'))
         ])
    )
    return res
