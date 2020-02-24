import textwrap

import theano.tensor as tt

from symbolic_pymc.theano.random_variables import NormalRV, observed
from symbolic_pymc.theano.printing import tt_pprint, tt_tprint


def test_notex_print():

    tt_normalrv_noname_expr = tt.scalar("b") * NormalRV(tt.scalar("\\mu"), tt.scalar("\\sigma"))
    expected = textwrap.dedent(
        r"""
    b in R, \mu in R, \sigma in R
    a ~ N(\mu, \sigma**2) in R
    (b * a)
    """
    )
    assert tt_pprint(tt_normalrv_noname_expr) == expected.strip()

    # Make sure the constant shape is show in values and not symbols.
    tt_normalrv_name_expr = tt.scalar("b") * NormalRV(
        tt.scalar("\\mu"), tt.scalar("\\sigma"), size=[2, 1], name="X"
    )
    expected = textwrap.dedent(
        r"""
    b in R, \mu in R, \sigma in R
    X ~ N(\mu, \sigma**2) in R**(2 x 1)
    (b * X)
    """
    )
    assert tt_pprint(tt_normalrv_name_expr) == expected.strip()

    tt_2_normalrv_noname_expr = tt.matrix("M") * NormalRV(
        tt.scalar("\\mu_2"), tt.scalar("\\sigma_2")
    )
    tt_2_normalrv_noname_expr *= tt.scalar("b") * NormalRV(
        tt_2_normalrv_noname_expr, tt.scalar("\\sigma")
    ) + tt.scalar("c")
    expected = textwrap.dedent(
        r"""
    M in R**(N^M_0 x N^M_1), \mu_2 in R, \sigma_2 in R
    b in R, \sigma in R, c in R
    a ~ N(\mu_2, \sigma_2**2) in R, d ~ N((M * a), \sigma**2) in R**(N^d_0 x N^d_1)
    ((M * a) * ((b * d) + c))
    """
    )
    assert tt_pprint(tt_2_normalrv_noname_expr) == expected.strip()

    expected = textwrap.dedent(
        r"""
    b in Z, c in Z, M in R**(N^M_0 x N^M_1)
    M[b, c]
    """
    )
    # TODO: "c" should be "1".
    assert (
        tt_pprint(tt.matrix("M")[tt.iscalar("a"), tt.constant(1, dtype="int")]) == expected.strip()
    )

    expected = textwrap.dedent(
        r"""
    M in R**(N^M_0 x N^M_1)
    M[1]
    """
    )
    assert tt_pprint(tt.matrix("M")[1]) == expected.strip()

    expected = textwrap.dedent(
        r"""
    M in N**(N^M_0)
    M[2:4:0]
    """
    )
    assert tt_pprint(tt.vector("M", dtype="uint32")[0:4:2]) == expected.strip()

    norm_rv = NormalRV(tt.scalar("\\mu"), tt.scalar("\\sigma"))
    rv_obs = observed(tt.constant(1.0, dtype=norm_rv.dtype), norm_rv)

    expected = textwrap.dedent(
        r"""
    \mu in R, \sigma in R
    a ~ N(\mu, \sigma**2) in R
    a = 1.0
        """
    )
    assert tt_pprint(rv_obs) == expected.strip()


def test_tex_print():

    tt_normalrv_noname_expr = tt.scalar("b") * NormalRV(tt.scalar("\\mu"), tt.scalar("\\sigma"))
    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
      b \in \mathbb{R}, \,\mu \in \mathbb{R}, \,\sigma \in \mathbb{R}
      \\
      a \sim \operatorname{N}\left(\mu, {\sigma}^{2}\right)\,  \in \mathbb{R}
      \end{gathered}
      \\
      (b \odot a)
    \end{equation}
    """
    )
    assert tt_tprint(tt_normalrv_noname_expr) == expected.strip()

    tt_normalrv_name_expr = tt.scalar("b") * NormalRV(
        tt.scalar("\\mu"), tt.scalar("\\sigma"), size=[2, 1], name="X"
    )
    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
      b \in \mathbb{R}, \,\mu \in \mathbb{R}, \,\sigma \in \mathbb{R}
      \\
      X \sim \operatorname{N}\left(\mu, {\sigma}^{2}\right)\,  \in \mathbb{R}^{2 \times 1}
      \end{gathered}
      \\
      (b \odot X)
    \end{equation}
    """
    )
    assert tt_tprint(tt_normalrv_name_expr) == expected.strip()

    tt_2_normalrv_noname_expr = tt.matrix("M") * NormalRV(
        tt.scalar("\\mu_2"), tt.scalar("\\sigma_2")
    )
    tt_2_normalrv_noname_expr *= tt.scalar("b") * NormalRV(
        tt_2_normalrv_noname_expr, tt.scalar("\\sigma")
    ) + tt.scalar("c")
    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
      M \in \mathbb{R}^{N^{M}_{0} \times N^{M}_{1}}
      \\
      \mu_2 \in \mathbb{R}, \,\sigma_2 \in \mathbb{R}
      \\
      b \in \mathbb{R}, \,\sigma \in \mathbb{R}, \,c \in \mathbb{R}
      \\
      a \sim \operatorname{N}\left(\mu_2, {\sigma_2}^{2}\right)\,  \in \mathbb{R}
      \\
      d \sim \operatorname{N}\left((M \odot a), {\sigma}^{2}\right)\,  \in \mathbb{R}^{N^{d}_{0} \times N^{d}_{1}}
      \end{gathered}
      \\
      ((M \odot a) \odot ((b \odot d) + c))
    \end{equation}
    """
    )
    assert tt_tprint(tt_2_normalrv_noname_expr) == expected.strip()

    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
      b \in \mathbb{Z}, \,c \in \mathbb{Z}, \,M \in \mathbb{R}^{N^{M}_{0} \times N^{M}_{1}}
      \end{gathered}
      \\
      M\left[b, \,c\right]
    \end{equation}
    """
    )
    # TODO: "c" should be "1".
    assert (
        tt_tprint(tt.matrix("M")[tt.iscalar("a"), tt.constant(1, dtype="int")]) == expected.strip()
    )

    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
      M \in \mathbb{R}^{N^{M}_{0} \times N^{M}_{1}}
      \end{gathered}
      \\
      M\left[1\right]
    \end{equation}
    """
    )
    assert tt_tprint(tt.matrix("M")[1]) == expected.strip()

    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
      M \in \mathbb{N}^{N^{M}_{0}}
      \end{gathered}
      \\
      M\left[2:4:0\right]
    \end{equation}
    """
    )
    assert tt_tprint(tt.vector("M", dtype="uint32")[0:4:2]) == expected.strip()

    norm_rv = NormalRV(tt.scalar("\\mu"), tt.scalar("\\sigma"))
    rv_obs = observed(tt.constant(1.0, dtype=norm_rv.dtype), norm_rv)

    expected = textwrap.dedent(
        r"""
    \begin{equation}
      \begin{gathered}
      \mu \in \mathbb{R}, \,\sigma \in \mathbb{R}
      \\
      a \sim \operatorname{N}\left(\mu, {\sigma}^{2}\right)\,  \in \mathbb{R}
      \end{gathered}
      \\
      a = 1.0
    \end{equation}
        """
    )
    assert tt_tprint(rv_obs) == expected.strip()
