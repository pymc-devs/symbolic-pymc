from theano.tensor.nlinalg import matrix_inverse

from unification import var
from kanren import conde, eq
from kanren.facts import Relation, fact

from .. import (MvNormalRV, observed)
from ..meta import mt
from ..utils import mt_type_params
from . import conjugate


# The prior distribution
prior_dist_mt = var('prior_dist')

# The observation distribution
obs_dist_mt = var('obs_dist')

# The variable specifying the fixed sample value of the random variable
# given by `Y_mt`
obs_sample_mt = var('obs_sample')

# Make the observation relationship explicit in the graph.
obs_mt = mt.observed(obs_sample_mt, obs_dist_mt)


conde_clauses = tuple()


def create_normal_normal_goals():
    """Bayes theorem for normal prior mean with a normal observation model.

    TODO: This implementation is a little too restrictive in that it limits the
    conjugate update to only random variables attached to explicitly defined
    observations (i.e. via the `observed` `Op`).

    Right now it serves as a way to pull in a distinct tensor carrying
    observation values, since the tensor corresponding to a sample from the
    observation distribution (i.e. `Y_mt`) cannot be a shared or constant
    tensor with a user-set value.

    We can simply add the non-observation pattern, let it do the Bayes update,
    then follow that up with an relation/optimization that exchanges the
    `RandomVariable` with its associated observed tensor (e.g. swap `Y_mt` with
    `y_mt` once all the conjugate updates are done).

    This would have to be guided in some way (e.g. to only update
    `RandomVariables` that are also `FunctionGraph` outputs--i.e. the ones we
    might want to sample).

    TODO: Lift univariate normals to multivariates so that this update can be
    applied to them, as well?  Seems lame to remake this just for the
    univariate cases, especially when they can be easily and completely
    embedded in multivariate spaces.
    """
    # Create the pattern/form of the prior normal distribution
    beta_name_lv = var('beta_name')
    beta_size_lv = var('beta_size')
    beta_rng_lv = var('beta_rng')
    a_lv = var('a')
    R_lv = var('R')
    beta_prior_mt = mt.MvNormalRV(a_lv, R_lv,
                                  size=beta_size_lv,
                                  rng=beta_rng_lv,
                                  name=beta_name_lv)
    # beta_type_lvars = mt_type_params(beta_prior_mt)

    y_name_lv = var('y_name')
    y_size_lv = var('y_size')
    y_rng_lv = var('y_rng')
    F_t_lv = var('f')
    V_lv = var('V')
    E_y_mt = mt.dot(F_t_lv, beta_prior_mt)
    Y_mt = mt.MvNormalRV(E_y_mt, V_lv,
                         size=y_size_lv,
                         rng=y_rng_lv,
                         name=y_name_lv)

    Y_obs_mt = mt.observed(obs_sample_mt, Y_mt)

    # Create tuple-form expressions for the posterior
    e_expr = mt.sub(Y_obs_mt, mt.dot(F_t_lv, a_lv))
    F_expr = (mt.transpose, F_t_lv)
    R_F_expr = (mt.dot, R_lv, F_expr)
    Q_expr = (mt.add,
              V_lv,
              (mt.dot,
               F_t_lv,
               R_F_expr))
    A_expr = (mt.dot, R_F_expr, (mt.matrix_inverse, Q_expr))
    # m = C \left(F V^{-1} y + R^{-1} a\right)
    m_expr = (mt.add, a_lv, (mt.dot, A_expr, e_expr))
    # C = \left(R^{-1} + F V^{-1} F^{\top}\right)^{-1}
    # TODO: We could use the naive posterior forms and apply identities, like
    # Woodbury's, in another set of "simplification" relations.
    # In some cases, this might make the patterns simpler and more broadly
    # applicable.
    C_expr = (mt.sub,
              R_lv,
              (mt.dot,
               (mt.dot, A_expr, Q_expr),
                  (mt.transpose, A_expr)))

    norm_posterior_exprs = (mt.MvNormalRV,
                            m_expr, C_expr,
                            y_size_lv, y_rng_lv)

    fact(conjugate,
         # MvNormal likelihood, MvNormal prior mean
         Y_obs_mt, norm_posterior_exprs)

    return ((eq, prior_dist_mt, beta_prior_mt),
            # This should unify `Y_mt` and `obs_dist_mt`.
            (eq, obs_mt, Y_obs_mt))


conde_clauses += (create_normal_normal_goals(),)


def create_normal_wishart_goals():
    """
    TODO
    """
    # Create the pattern/form of the prior normal distribution
    Sigma_name_lv = var('Sigma_name')
    Sigma_size_lv = var('Sigma_size')
    Sigma_rng_lv = var('Sigma_rng')
    V_lv = var('V')
    n_lv = var('n')
    Sigma_prior_mt = mt.WishartRV(V_lv, n_lv,
                                  Sigma_size_lv, Sigma_rng_lv,
                                  name=Sigma_name_lv)
    Sigma_type_lvars = mt_type_params(Sigma_prior_mt)

    y_name_lv = var('y_name')
    y_size_lv = var('y_size')
    y_rng_lv = var('y_rng')
    V_lv = var('V')
    f_mt = var('f')
    Y_mt = mt.MvNormalRV(f_mt, V_lv,
                         y_size_lv, y_rng_lv,
                         name=y_name_lv)

    y_mt = var('y')
    Y_obs_mt = mt.observed(y_mt, Y_mt)

    n_post_mt = (mt.add, n_lv, (mt.Shape, Y_obs_mt))

    # wishart_posterior_exprs = (mt.MvStudentTRV,
    #                            m_expr, C_expr,
    #                            y_size_lv, y_rng_lv)

    # fact(conjugates,
    #      Y_obs_mt, wishart_posterior_exprs)
    pass


def conjugate_posteriors(x, y):
    """A goal relating conjugate priors and their posterior forms.

    This goal unifies `y` with a tuple-form expression for a dictionary
    specifying the graph's node replacements.
    Those replacements map prior random variable terms to their posteriors
    forms.  All other terms depending on those terms essentially become
    posterior predictives.
    """
    z = var()

    # First, find a basic conjugate structure match.
    goals = ((conjugate, x, z),)

    # Second, each conjugate case might have its own special conditions.
    goals += ((conde,) + conde_clauses,)

    # Third, connect the discovered pieces and produce the necessary output.
    # TODO: We could have a "reifiable" goal that makes sure the output is
    # a valid base/non-meta object.
    goals += ((eq, y,
               (dict,
                [
                    # Replace observation with one that doesn't link to
                    # the integrated one
                    (x, (mt.observed, obs_sample_mt, None)),
                    # (Y_mt, None),
                    # Replace the prior with the posterior
                    (prior_dist_mt, z),
                ])),)

    # This conde is just a lame way to form the conjunction
    # TODO: Use one of the *all* functions.
    res = (conde, goals)
    return res
