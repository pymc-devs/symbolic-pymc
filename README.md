# Symbolic PyMC

[![Build Status](https://travis-ci.org/pymc-devs/symbolic-pymc.svg?branch=master)](https://travis-ci.org/pymc-devs/symbolic-pymc) [![Coverage Status](https://coveralls.io/repos/github/pymc-devs/symbolic-pymc/badge.svg?branch=master)](https://coveralls.io/github/pymc-devs/symbolic-pymc?branch=master)


Symbolic PyMC provides tools for the symbolic manipulation of [PyMC](https://github.com/pymc-devs) models and their underlying computational graphs.  It enables graph manipulations in the relational DSL [miniKanren](http://minikanren.org/)&mdash;via the [`kanren`](https://github.com/logpy/logpy) package&mdash;by way of meta classes and S-expression forms of a graph.

This work stems from a series of articles starting [here](https://brandonwillard.github.io/a-role-for-symbolic-computation-in-the-general-estimation-of-statistical-models.html).

*This package is currently in alpha, so expect large-scale changes at any time!*

## Features

### General

* Full [miniKanren](http://minikanren.org/) integration for relational graph/model manipulation.
  - Perform simple and robust "search and replace" over arbitrary graphs (e.g. Python builtin collections, AST, tensor algebra graphs, etc.)
  - Create and compose relations with explicit high-level statistical/mathematical meaning and functionality, such as "`X` is a normal scale mixture with mixing distribution `Y`", and automatically "solve" for components (i.e. `X` and `Y`) that satisfy a relation.
  - Apply non-trivial conditions&mdash;as relations&mdash;to produce sophisticated graph manipulations (e.g. search for normal scale mixtures and scale a term in the mixing distribution).
  - Integrate standard Python operations into relations (e.g. use a symbolic math library to compute an inverse-Laplace transform to determine if a distribution is a scale mixture&mdash;and find its mixing distribution).
* Convert graphs to an S-expression-like tuple-based form and perform manipulations at the syntax level.
* Pre-built relations for symbolic closed-form posteriors and standard statistical model reformulations.

### [Theano](https://github.com/Theano/Theano)

* A more robust Theano `Op` for representing random variables
* Conversion of PyMC3 models into sample-able Theano graphs representing all random variable inter-dependencies
* A LaTeX pretty printer that displays shape information and distributions in mathematical notation

### [TensorFlow](https://github.com/tensorflow/tensorflow)

* TensorFlow graph support
* [In progress] PyMC4 model conversion

## Installation

The package name is `symbolic_pymc` and it can be installed with `pip` directly from GitHub
```shell
$ pip install git+https://github.com/pymc-devs/symbolic-pymc
```
or after cloning the repo (and then installing with `pip`).

## Examples
### Compute Symbolic Closed-form Posteriors

```python
import numpy as np

import theano
import theano.tensor as tt

import pymc3 as pm

from unification import var

from kanren import run
from kanren.core import lallgreedy
from kanren.goals import condeseq

from symbolic_pymc.theano.printing import tt_pprint
from symbolic_pymc.theano.pymc3 import model_graph
from symbolic_pymc.theano.utils import optimize_graph, canonicalize
from symbolic_pymc.theano.opt import KanrenRelationSub

from symbolic_pymc.relations.graph import graph_applyo
from symbolic_pymc.relations.theano.conjugates import conjugate, conde_clauses

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
    beta_rv = pm.MvNormal('\\beta', a_tt, R_tt, shape=(2,))

    # An observed random variable using the prior as a regression parameter
    E_y_rv = F_t_tt.dot(beta_rv)
    Y_rv = pm.MvNormal('Y', E_y_rv, V_tt, observed=y_tt)

# Create a graph for the model
fgraph = model_graph(model, output_vars=[beta_rv, Y_rv])


# Create and apply the conjugate posterior relation
def conjugateo(x, y):
    """State a conjugate relationship between terms (with some conditions)."""
    return (lallgreedy,
            (conjugate, x, y),
            (condeseq, conde_clauses))


def conjugate_dists(graph):
    """Automatically apply closed-form conjugates in a graph."""
    expr_graph, = run(1, var('q'),
                      (graph_applyo, conjugateo,
                       graph, var('q')))

    fgraph_opt = expr_graph.eval_obj
    fgraph_opt_tt = fgraph_opt.reify()
    return fgraph_opt_tt


fgraph_opt = conjugate_dists(fgraph.outputs[1])
```

```
>>> print(tt_pprint(fgraph_opt))
a in R**(N^a_0), R in R**(N^R_0 x N^R_1), F in R**(N^F_0 x N^F_1)
c in R**(N^c_0 x N^c_1), V in R**(N^V_0 x N^V_1)
d in R**(N^d_0 x N^d_1), e in R**(N^e_0 x N^e_1)
\beta ~ N(a, R) in R**(N^\beta_0), Y ~ N((F * \beta), V) in R**(N^Y_0)
b ~ N((a + (((R * F.T) * c) * (Y = [-3.] - (F * a)))), (R - ((((R * F.T) * d) * (V + (F * (R * F.T)))) * ((R * F.T) * e).T))) in R**(N^b_0)
b
```

### Automatic Re-centering and Re-scaling

We can automate the PyMC3 model recentering and rescaling in ["Why hierarchical models are awesome, tricky, and Bayesian"](https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/) and improve sample chain quality:

```python
import pandas as pd
from symbolic_pymc.theano.utils import get_rv_observation
from symbolic_pymc.relations.theano.distributions import scale_loc_transform

# Skip compilation temporarily
_cxx_config = theano.config.cxx
theano.config.cxx = ''

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


fgraph = model_graph(model_centered)
fgraph = canonicalize(fgraph, in_place=False)

posterior_opt = EquilibriumOptimizer(
    [KanrenRelationSub(scale_loc_transform,
                       node_filter=get_rv_observation)],
    max_use_ratio=10)

fgraph_opt = optimize_graph(fgraph, posterior_opt, return_graph=True)
fgraph_opt = canonicalize(fgraph_opt, in_place=False)

theano.config.cxx = _cxx_config
model_recentered = graph_model(fgraph_opt)

np.random.seed(123)

with model_centered:
    centered_trace = pm.sample(draws=5000, tune=1000, njobs=2)[1000:]

with model_recentered:
    recentered_trace = pm.sample(draws=5000, tune=1000, njobs=2)[1000:]
```
### Convert a PyMC3 model to a Theano graph

```python
mu_X = tt.scalar('\\mu_X')
sd_X = tt.scalar('\\sigma_X')
mu_Y = tt.scalar('\\mu_Y')
sd_Y = tt.scalar('\\sigma_Y')
mu_X.tag.test_value = np.array(0., dtype=tt.config.floatX)
sd_X.tag.test_value = np.array(1., dtype=tt.config.floatX)
mu_Y.tag.test_value = np.array(1., dtype=tt.config.floatX)
sd_Y.tag.test_value = np.array(0.5, dtype=tt.config.floatX)

with pm.Model() as model:
    X_rv = pm.Normal('X', mu_X, sd=sd_X)
    Y_rv = pm.Normal('Y', mu_Y, sd=sd_Y)
    Z_rv = pm.Normal('Z',
                     X_rv + Y_rv,
                     sd=sd_X + sd_Y,
                     observed=10.)

fgraph = model_graph(model)
fgraph = canonicalize(fgraph)
```

```
>>> from theano.printing import debugprint as tt_dprint
>>> tt_dprint(fgraph)
<symbolic_pymc.Observed object at 0x7f61b61385f8> [id A] ''   5
 |TensorConstant{10.0} [id B]
 |normal_rv.1 [id C] 'Z'   4
   |Elemwise{add,no_inplace} [id D] ''   3
   | |normal_rv.1 [id E] 'X'   2
   | | |\\mu_X [id F]
   | | |\\sigma_X [id G]
   | | |TensorConstant{[]} [id H]
   | | |<RandomStateType> [id I]
   | |normal_rv.1 [id J] 'Y'   1
   |   |\\mu_Y [id K]
   |   |\\sigma_Y [id L]
   |   |TensorConstant{[]} [id H]
   |   |<RandomStateType> [id I]
   |Elemwise{add,no_inplace} [id M] ''   0
   | |\\sigma_X [id G]
   | |\\sigma_Y [id L]
   |TensorConstant{[]} [id H]
   |<RandomStateType> [id I]

```
### Mathematical Notation Pretty Printing

```
>>> from symbolic_pymc.theano.printing import tt_pprint
>>> print(tt_pprint(fgraph))
\\mu_X in R, \\sigma_X in R, \\mu_Y in R, \\sigma_Y in R
X ~ N(\\mu_X, \\sigma_X**2) in R, Y ~ N(\\mu_Y, \\sigma_Y**2) in R
Z ~ N((X + Y), (\\sigma_X + \\sigma_Y)**2) in R
Z = 10.0
```

```
>>> from symbolic_pymc.theano.printing import tt_tprint
>>> print(tt_tprint(fgraph))
```
produces
```latex
\begin{equation}
  \begin{gathered}
  \mu_X \in \mathbb{R}, \,\sigma_X \in \mathbb{R}
  \\
  \mu_Y \in \mathbb{R}, \,\sigma_Y \in \mathbb{R}
  \\
  X \sim \operatorname{N}\left(\mu_X, {\sigma_X}^{2}\right)\,  \in \mathbb{R}
  \\
  Y \sim \operatorname{N}\left(\mu_Y, {\sigma_Y}^{2}\right)\,  \in \mathbb{R}
  \\
  Z \sim \operatorname{N}\left((X + Y), {(\sigma_X + \sigma_Y)}^{2}\right)\,  \in \mathbb{R}
  \end{gathered}
  \\
  Z = 10.0
\end{equation}
```
