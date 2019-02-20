import theano.tensor as tt

from symbolic_pymc import NormalRV
from symbolic_pymc.printing import tt_pprint


def test_notex_print():

    tt_normalrv_noname_expr = tt.scalar('b') * NormalRV(tt.scalar('\\mu'),
                                                        tt.scalar('\\sigma'))
    expected = 'b in R, \\mu in R, \\sigma in R\na ~ N(\\mu, \\sigma**2) in R\n(b * a)'
    assert tt_pprint(tt_normalrv_noname_expr) == expected

    # Make sure the constant shape is show in values and not symbols.
    tt_normalrv_name_expr = tt.scalar('b') * NormalRV(tt.scalar('\\mu'),
                                                      tt.scalar('\\sigma'),
                                                      size=[2, 1], name='X')
    expected = 'b in R, \\mu in R, \\sigma in R\nX ~ N(\\mu, \\sigma**2) in R**(2 x 1)\n(b * X)'
    assert tt_pprint(tt_normalrv_name_expr) == expected

    tt_2_normalrv_noname_expr = tt.matrix('M') * NormalRV(tt.scalar('\\mu_2'),
                                                          tt.scalar('\\sigma_2'))
    tt_2_normalrv_noname_expr *= (tt.scalar('b') *
                                  NormalRV(tt_2_normalrv_noname_expr,
                                           tt.scalar('\\sigma')) +
                                  tt.scalar('c'))
    expected = 'M in R**(N^M_0 x N^M_1), \\mu_2 in R, \\sigma_2 in R\nb in R, \\sigma in R, c in R\na ~ N(\\mu_2, \\sigma_2**2) in R, d ~ N((M * a), \\sigma**2) in R**(N^d_0 x N^d_1)\n((M * a) * ((b * d) + c))'
    assert tt_pprint(tt_2_normalrv_noname_expr) == expected
