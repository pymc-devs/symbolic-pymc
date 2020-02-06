=======================================
Compute Symbolic Closed-form Posteriors
=======================================

    :Author: Brandon T. Willard
    :Date: 2019-11-24

.. code-block:: python
    :name: compute-symbolic-posterior

    import numpy as np

    import theano
    import theano.tensor as tt

    import pymc3 as pm

    from functools import partial

    from unification import var

    from kanren import run
    from kanren.graph import reduceo, walko

    from symbolic_pymc.theano.printing import tt_pprint
    from symbolic_pymc.theano.pymc3 import model_graph

    from symbolic_pymc.relations.theano.conjugates import conjugate

    theano.config.cxx = ''
    theano.config.compute_test_value = 'ignore'

    a_tt = tt.vector('a')
    R_tt = tt.matrix('R')
    F_t_tt = tt.matrix('F')
    V_tt = tt.matrix('V')

    a_tt.tag.test_value = np.r_[1., 0.]
    R_tt.tag.test_value = np.diag([10., 10.])
    F_t_tt.tag.test_value = np.c_[-2., 1.]
    V_tt.tag.test_value = np.diag([0.5])

    y_tt = tt.as_tensor_variable(np.r_[-3.])
    y_tt.name = 'y'

    with pm.Model() as model:

        # A normal prior
        beta_rv = pm.MvNormal('beta', a_tt, R_tt, shape=(2,))

        # An observed random variable using the prior as a regression parameter
        E_y_rv = F_t_tt.dot(beta_rv)
        Y_rv = pm.MvNormal('Y', E_y_rv, V_tt, observed=y_tt)

    # Create a graph for the model
    fgraph = model_graph(model, output_vars=[Y_rv])


    def conjugate_graph(graph):
        """Apply conjugate relations throughout a graph."""

        def fixedp_conjugate_walko(x, y):
            return reduceo(partial(walko, conjugate), x, y)

        expr_graph, = run(1, var('q'),
                          fixedp_conjugate_walko(graph, var('q')))

        fgraph_opt = expr_graph.eval_obj
        fgraph_opt_tt = fgraph_opt.reify()
        return fgraph_opt_tt


    fgraph_conj = conjugate_graph(fgraph.outputs[0])

Before
------

.. code-block:: python
    :name: posterior-before-print

    >>> print(tt_pprint(fgraph))
    F in R**(N^F_0 x N^F_1), a in R**(N^a_0), R in R**(N^R_0 x N^R_1)
    V in R**(N^V_0 x N^V_1)
    beta ~ N(a, R) in R**(N^beta_0), Y ~ N((F * beta), V) in R**(N^Y_0)
    Y = [-3.]

After
-----

.. code-block:: python
    :name: posterior-after-print

    >>> print(tt_pprint(fgraph_conj))
    a in R**(N^a_0), R in R**(N^R_0 x N^R_1), F in R**(N^F_0 x N^F_1)
    c in R**(N^c_0 x N^c_1), d in R**(N^d_0 x N^d_1)
    V in R**(N^V_0 x N^V_1), e in R**(N^e_0 x N^e_1)
    b ~ N((a + (((R * F.T) * c) * ([-3.] - (F * a)))), (R - ((((R * F.T) * d) * (V + (F * (R * F.T)))) * ((R * F.T) * e).T))) in R**(N^b_0)
    b
