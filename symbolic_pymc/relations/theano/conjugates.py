import theano

from unification import var
from kanren.facts import fact

from .. import conjugate
from ...unify import etuple
from ...theano.meta import mt


mt.namespaces += [theano.tensor.nlinalg]


def _create_normal_normal_goals():
    """Produce a relation representing Bayes theorem for a multivariate normal prior mean with a normal observation model.

    NOTE: This unifies with meta graph objects directly and not their etuple
    forms, so use it on a meta graphs if you want it to work.

    TODO: This implementation is a little too restrictive in that it limits the
    conjugate update to only random variables attached to explicitly defined
    observations (i.e. via the `observed` `Op`).

    TODO: Lift univariate normals to multivariates so that this update can be
    applied to them, as well?  Seems lame to remake this just for the
    univariate cases, especially when they can be easily and completely
    embedded in multivariate spaces.

    """
    #
    # Create the pattern/form of the prior normal distribution
    #
    beta_name_lv = var("beta_name")
    beta_size_lv = var("beta_size")
    beta_rng_lv = var("beta_rng")
    a_lv = var("a")
    R_lv = var("R")
    beta_prior_mt = mt.MvNormalRV(a_lv, R_lv, size=beta_size_lv, rng=beta_rng_lv, name=beta_name_lv)

    y_name_lv = var("y_name")
    y_size_lv = var("y_size")
    y_rng_lv = var("y_rng")
    F_t_lv = var("f")
    V_lv = var("V")
    E_y_mt = mt.dot(F_t_lv, beta_prior_mt)
    Y_mt = mt.MvNormalRV(E_y_mt, V_lv, size=y_size_lv, rng=y_rng_lv, name=y_name_lv)

    # The variable specifying the fixed sample value of the random variable
    # given by `Y_mt`
    obs_sample_mt = var("obs_sample")

    Y_obs_mt = mt.observed(obs_sample_mt, Y_mt)

    #
    # Create tuple-form expressions that construct the posterior
    #
    e_expr = mt.sub(obs_sample_mt, mt.dot(F_t_lv, a_lv))
    F_expr = etuple(mt.transpose, F_t_lv)
    R_F_expr = etuple(mt.dot, R_lv, F_expr)
    Q_expr = etuple(mt.add, V_lv, etuple(mt.dot, F_t_lv, R_F_expr))
    A_expr = etuple(mt.dot, R_F_expr, etuple(mt.matrix_inverse, Q_expr))
    # m = C \left(F V^{-1} y + R^{-1} a\right)
    m_expr = etuple(mt.add, a_lv, etuple(mt.dot, A_expr, e_expr))
    # C = \left(R^{-1} + F V^{-1} F^{\top}\right)^{-1}
    # TODO: We could use the naive posterior forms and apply identities, like
    # Woodbury's, in another set of "simplification" relations.
    # In some cases, this might make the patterns simpler and more broadly
    # applicable.
    C_expr = etuple(
        mt.sub, R_lv, etuple(mt.dot, etuple(mt.dot, A_expr, Q_expr), etuple(mt.transpose, A_expr))
    )

    norm_posterior_exprs = etuple(mt.MvNormalRV, m_expr, C_expr, y_size_lv, y_rng_lv)

    return (Y_obs_mt, norm_posterior_exprs)


def _create_normal_wishart_goals():  # pragma: no cover
    """TODO."""
    # Create the pattern/form of the prior normal distribution
    Sigma_name_lv = var("Sigma_name")
    Sigma_size_lv = var("Sigma_size")
    Sigma_rng_lv = var("Sigma_rng")
    V_lv = var("V")
    n_lv = var("n")
    Sigma_prior_mt = mt.WishartRV(V_lv, n_lv, Sigma_size_lv, Sigma_rng_lv, name=Sigma_name_lv)

    y_name_lv = var("y_name")
    y_size_lv = var("y_size")
    y_rng_lv = var("y_rng")
    V_lv = var("V")
    f_mt = var("f")
    Y_mt = mt.MvNormalRV(f_mt, V_lv, y_size_lv, y_rng_lv, name=y_name_lv)

    y_mt = var("y")
    Y_obs_mt = mt.observed(y_mt, Y_mt)

    n_post_mt = etuple(mt.add, n_lv, etuple(mt.Shape, Y_obs_mt))

    # wishart_posterior_exprs = etuple(mt.MvStudentTRV,
    #                                  m_expr, C_expr,
    #                                  y_size_lv, y_rng_lv)

    # return (Sigma_prior_mt, wishart_posterior_exprs)


norm_norm_prior_post = _create_normal_normal_goals()
fact(
    conjugate,
    # An unconjugated observation backed by an MvNormal likelihood with MvNormal prior mean
    norm_norm_prior_post[0],
    # The corresponding conjugated distribution
    norm_norm_prior_post[1],
)
