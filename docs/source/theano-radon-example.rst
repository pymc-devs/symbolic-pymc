=====================================
Automatic Re-centering and Re-scaling
=====================================

    :Author: Brandon T. Willard
    :Date: 2019-11-24

Using \ ``symbolic_pymc``\  we can automate the PyMC3 model
transformation in `"Why hierarchical models are awesome, tricky, and Bayesian" <https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/>`_
and improve sample chain quality.

.. code-block:: python
    :name: recenter-radon-model

    import numpy as np
    import pandas as pd

    import pymc3 as pm

    import theano
    import theano.tensor as tt

    from functools import partial

    from unification import var

    from kanren import run
    from kanren.graph import reduceo

    from symbolic_pymc.theano.meta import mt
    from symbolic_pymc.theano.pymc3 import model_graph, graph_model
    from symbolic_pymc.theano.utils import canonicalize

    from symbolic_pymc.relations.theano import non_obs_walko
    from symbolic_pymc.relations.theano.distributions import scale_loc_transform


    tt.config.compute_test_value = 'ignore'

    data = pd.read_csv('https://github.com/pymc-devs/pymc3/raw/master/pymc3/examples/data/radon.csv')
    data['log_radon'] = data['log_radon'].astype(theano.config.floatX)
    county_names = data.county.unique()
    county_idx = data.county_code.values

    n_counties = len(data.county.unique())

    with pm.Model() as model_centered:
        mu_a = pm.Normal('mu_a', mu=0., sd=100**2)
        sigma_a = pm.HalfCauchy('sigma_a', 5)
        mu_b = pm.Normal('mu_b', mu=0., sd=100**2)
        sigma_b = pm.HalfCauchy('sigma_b', 5)
        a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=n_counties)
        b = pm.Normal('b', mu=mu_b, sd=sigma_b, shape=n_counties)
        eps = pm.HalfCauchy('eps', 5)
        radon_est = a[county_idx] + b[county_idx] * data.floor.values
        radon_like = pm.Normal('radon_like', mu=radon_est, sd=eps,
                               observed=data.log_radon)

    # Convert the PyMC3 graph into a symbolic-pymc graph
    fgraph = model_graph(model_centered)
    # Perform a set of standard algebraic simplifications
    fgraph = canonicalize(fgraph, in_place=False)


    def reparam_graph(graph):
        """Apply re-parameterization relations throughout a graph."""

        graph_mt = mt(graph)

        def scale_loc_fixedp_applyo(x, y):
            return reduceo(partial(non_obs_walko, scale_loc_transform), x, y)

        q = var()
        expr_graph = run(0, q,
                         # Apply our transforms to unobserved RVs only
                         scale_loc_fixedp_applyo(graph_mt, q))

        expr_graph = expr_graph[0]
        opt_graph_tt = expr_graph.reify()

        # PyMC3 needs names for each RV
        opt_graph_tt.owner.inputs[1].name = 'Y_new'

        return opt_graph_tt


    fgraph_reparam = reparam_graph(fgraph.outputs[0])

    # Convert the symbolic-pymc graph into a PyMC3 graph so that we can sample it
    model_recentered = graph_model(fgraph_reparam)

    np.random.seed(123)

    with model_centered:
        centered_trace = pm.sample(draws=5000, tune=1000, cores=4)[1000:]

    with model_recentered:
        recentered_trace = pm.sample(draws=5000, tune=1000, cores=4)[1000:]

Before
------

.. code-block:: python
    :name: before-recenter-plot

    >>> pm.traceplot(centered_trace, varnames=['sigma_b'])

.. _fig:original_model_trace:

.. figure:: _static/centered_trace.png
    :width: 800px
    :align: center
    :figclass: align-center


    Original model trace results.

After
-----

.. code-block:: python
    :name: after-recenter-plot

    >>> pm.traceplot(recentered_trace, varnames=['sigma_b'])

.. _fig:transformed_model_trace:

.. figure:: _static/recentered_trace.png
    :width: 800px
    :align: center
    :figclass: align-center


    Transformed model trace results.
