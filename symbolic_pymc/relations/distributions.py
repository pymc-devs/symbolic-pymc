"""Relations pertaining to probability distributions."""
from unification import var
from kanren import conde, eq
from kanren.facts import fact

from . import constant_neq, concat
from ..meta import mt
from ..unify import etuple

from kanren.facts import Relation

derived_dist = Relation("derived_dist")
stable_dist = Relation("stable_dist")
generalized_gamma_dist = Relation("generalized_gamma_dist")

uniform_mt = mt.UniformRV(var(), var(), size=var(), rng=var(), name=var())
normal_mt = mt.NormalRV(var(), var(), size=var(), rng=var(), name=var())
cauchy_mt = mt.CauchyRV(var(), var(), size=var(), rng=var(), name=var())
halfcauchy_mt = mt.HalfCauchyRV(var(), var(), size=var(), rng=var(), name=var())
gamma_mt = mt.GammaRV(var(), var(), size=var(), rng=var(), name=var())
exponential_mt = mt.ExponentialRV(var(), size=var(), rng=var(), name=var())

# TODO: Add constraints for different variations of this.  Also, consider a
# check for exact equality of the two dists, or simply normalize/canonicalize
# the graph first.
fact(
    derived_dist,
    mt.true_div(
        mt.NormalRV(0.0, 1.0, size=var("_ratio_norm_size"), rng=var("_ratio_norm_rng"), name=var()),
        mt.NormalRV(0.0, 1.0, size=var("_ratio_norm_size"), rng=var(), name=var()),
    ),
    mt.CauchyRV(0.0, 1.0, size=var("_ratio_norm_size"), rng=var("_ratio_norm_rng")),
)

# TODO:
# fact(stable_dist,
#      normal_mt, ('StableRV',
#                  2., 0.,
#                  normal_mt.owner.inputs[1],
#                  normal_mt.owner.inputs[1]))
# fact(stable_dist,
#      cauchy_mt, ('StableRV',
#                  1., 0.,
#                  cauchy_mt.owner.inputs[1],
#                  cauchy_mt.owner.inputs[1]))

# TODO: Weibull, Gamma, Exponential, Half-normal
# fact(generalized_gamma_dist,
#      None,
#      None)


def scale_loc_transform(in_expr, out_expr):
    """Create relations for lifting and sinking scale and location parameters of distributions.

    I.e. f(a + b*x) -> a + b * f(x)

    For example, `in_expr`: f(a + b*x) == `out_expr`: a + b * f(x).

    TODO: Match larger distribution families and perform transforms from there.

    """
    # Scale and location transform expression "pattern" for a Normal term.
    n_name_lv = normal_mt.name
    n_mean_lv, n_sd_lv, n_size_lv, n_rng_lv = normal_mt.owner.inputs
    offset_name_mt = var()
    rct_norm_offset_mt = etuple(
        mt.add,
        n_mean_lv,
        etuple(
            mt.mul,
            n_sd_lv,
            mt.NormalRV(0.0, 1.0, size=n_size_lv, rng=n_rng_lv, name=offset_name_mt),
        ),
    )

    # Scale and location transform expression "pattern" for a Cauchy term.
    c_name_lv = cauchy_mt.name
    c_mean_lv, c_beta_lv, c_size_lv, c_rng_lv = cauchy_mt.owner.inputs
    rct_cauchy_offset_mt = etuple(
        mt.add,
        c_mean_lv,
        etuple(
            mt.mul,
            c_beta_lv,
            mt.CauchyRV(0.0, 1.0, size=c_size_lv, rng=c_rng_lv, name=offset_name_mt),
        ),
    )

    # u_name_lv = uniform_mt.name
    # u_a_lv, u_b_lv, u_size_lv, u_rng_lv = uniform_mt.owner.inputs
    # rct_uniform_scale_mt = etuple(
    #     mt.mul,
    #     u_b_lv,
    #     mt.UniformRV(0.0, 1.0, size=u_size_lv, rng=u_rng_lv, name=offset_name_mt),
    # )
    # rct_uniform_loc_mt = etuple(mt.add, u_c_lv,
    #                             mt.UniformRV(u_a_lv, u_b_lv,
    #                                          size=u_size_lv,
    #                                          rng=u_rng_lv,
    #                                          name=offset_name_mt))

    # XXX: PyMC3 rescaling issue doesn't allow us to take the more general
    # approach, which involves separate scale and location rewrites.

    # f(a + X) -> a + f(X)
    # norm_mean_name_mt = var()
    # rct_norm_mean_mt = etuple(mt.add, mean_lv,
    #                           mt.NormalRV(0., sd_lv,
    #                                       size=norm_size_lv,
    #                                       rng=norm_rng_lv,
    #                                       name=norm_mean_name_mt))
    #
    # f(a * X) -> a * f(X)
    # norm_sd_name_mt = var()
    # rct_norm_sd_mt = etuple(mt.mul, n_sd_lv,
    #                         mt.NormalRV(n_mean_lv, 1.,
    #                                     size=n_size_lv,
    #                                     rng=n_rng_lv,
    #                                     name=n_name_lv))

    rels = (
        conde,
        [
            (eq, in_expr, normal_mt),
            (
                conde,
                [
                    (constant_neq, n_sd_lv, 1),
                    (constant_neq, n_mean_lv, 0),
                    (eq, out_expr, rct_norm_offset_mt),
                    (concat, [n_name_lv, "_offset"], offset_name_mt),
                ],
                # [(constant_neq, mean_lv, 0),
                #  (eq, out_expr, rct_norm_mean_mt),
                #  (concat, [norm_name_lv, "_rmean"], norm_mean_name_mt)],
                # [(constant_neq, sd_lv, 1),
                #  (eq, out_expr, rct_norm_sd_mt),
                #  (concat, [norm_name_lv, "_rsd"], norm_sd_name_mt)],
            ),
        ],
        [
            (eq, in_expr, cauchy_mt),
            (
                conde,
                [
                    (constant_neq, c_beta_lv, 1),
                    # TODO: Add a positivity constraint for the scale.
                    (constant_neq, c_mean_lv, 0),
                    (eq, out_expr, rct_cauchy_offset_mt),
                    (concat, [c_name_lv, "_offset"], offset_name_mt),
                ],
            ),
        ],
        # TODO
        # [(eq, in_expr, uniform_mt),
        #  (conde,
        #   [(constant_eq, u_a_lv, 0),
        #    (eq, out_expr, rct_uniform_scale_mt),
        #    (concat, [u_name_lv, "_scale"], offset_name_mt)],
        #  )],
    )

    return rels
