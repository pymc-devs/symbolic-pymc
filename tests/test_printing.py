import theano.tensor as tt

from symbolic_pymc import *
from symbolic_pymc.printing import *


def test_notex_print():
    tt.config.compute_test_value = 'ignore'

    tt_normalrv_name_expr = tt.scalar('b') * NormalRV(tt.scalar('\\mu'),
                                                      tt.scalar('\\sigma'),
                                                      size=[2, 1], name='X')
    assert tt_pprint(tt_normalrv_name_expr) == "b in R\n\\mu in R\n\\sigma in R\nX ~ N(\\mu, \\sigma**2),  X in R**(2 x N^X_1)\n(b * X)"

    tt_normalrv_noname_expr = tt.scalar('b') * NormalRV(tt.scalar('\\mu'),
                                                        tt.scalar('\\sigma'))
    assert tt_pprint(tt_normalrv_noname_expr) == "b in R\n\\mu in R\n\\sigma in R\na ~ N(\\mu, \\sigma**2),  a in R\n(b * a)"

    tt_2_normalrv_noname_expr = tt.matrix('M') * NormalRV(tt.scalar('\\mu_2'),
                                                          tt.scalar('\\sigma_2'))
    tt_2_normalrv_noname_expr *= (tt.scalar('b') *
                                  NormalRV(tt_2_normalrv_noname_expr,
                                           tt.scalar('\\sigma')) +
                                  tt.scalar('c'))
    assert tt_pprint(tt_2_normalrv_noname_expr) == 'M in R**(N^M_0 x N^M_1)\n\\mu_2 in R\n\\sigma_2 in R\na ~ N(\\mu_2, \\sigma_2**2),  a in R\nb in R\n\\sigma in R\nd ~ N((M * a), \\sigma**2),  d in R**(N^d_0 x N^d_1)\nc in R\n((M * a) * ((b * d) + c))'
