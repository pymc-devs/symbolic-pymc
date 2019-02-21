# Symbolic PyMC

A Python package with tools for the symbolic manipulation of [Theano](https://github.com/Theano) graphs and [PyMC3](https://github.com/pymc-devs/pymc3) models.

*This package is currently in alpha, so expect large-scale changes at any time!*
(As of now, it only serves as a central source for some loosely connected symbolic math involving Theano and PyMC3.)

These tools are designed to help automate the mathematics used in statistics and probability theory, especially the kind employed to produce specialized MCMC routines.

The project's secondary purpose is to abstract math-level symbolic work away from the specifics of Theano.  To do this, it implements graph manipulations in miniKanren--via the [`kanren`](https://github.com/logpy/logpy) package--and works exclusively with meta objects.

This work stems from a series of articles starting [here](https://brandonwillard.github.io/a-role-for-symbolic-computation-in-the-general-estimation-of-statistical-models.html).

## Features

* A more robust Theano `Op` for representing random variables
* Full miniKanren integration
* Conversion of PyMC3 models into sample-able Theano graphs representing all random variable inter-dependencies
* A LaTeX pretty printer that displays shape information and distributions in mathematical notation
* (In progress) Theano optimizations for symbolic closed-form posteriors and other probability theory-based model reformulations

## Installation

The package name is `symbolic_pymc` and it can be installed with `pip` directly from GitHub
```shell
$ pip install git+https://github.com/pymc-devs/symbolic-pymc
```
or after cloning the repo (and then installing with `pip`).

## Examples

### Convert a PyMC3 model to a Theano graph

```python
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm

from symbolic_pymc.pymc3 import model_graph
from symbolic_pymc.utils import canonicalize

theano.config.cxx = ''

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
>>> from symbolic_pymc.printing import tt_pprint
>>> print(tt_pprint(fgraph))
\\mu_X in R, \\sigma_X in R, \\mu_Y in R, \\sigma_Y in R
X ~ N(\\mu_X, \\sigma_X**2) in R, Y ~ N(\\mu_Y, \\sigma_Y**2) in R
Z ~ N((X + Y), (\\sigma_X + \\sigma_Y)**2) in R
Z = 10.0
```

```
>>> from symbolic_pymc.printing import tt_tprint
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

### Compute Symbolic Closed-form Posteriors

```python
from theano.gof.opt import EquilibriumOptimizer

from symbolic_pymc.utils import optimize_graph
from symbolic_pymc.printing import tt_pprint
from symbolic_pymc.opt import KanrenRelationSub
from symbolic_pymc.relations.conjugates import conjugate_posteriors

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

# Create and apply the conjugate posterior optimizer
posterior_opt = EquilibriumOptimizer(
    [KanrenRelationSub(conjugate_posteriors)],
    max_use_ratio=10)

fgraph_opt = optimize_graph(fgraph, posterior_opt)
fgraph_opt = canonicalize(fgraph_opt)
```

```
>>> print(tt_pprint(fgraph_opt))
a in R**(N^a_0), R in R**(N^R_0 x N^R_1), F in R**(N^F_0 x N^F_1)
c in R**(N^c_0 x N^c_1), V in R**(N^V_0 x N^V_1)
b ~ N((a + (((R * F.T) * c) * ([-3.] - (F * a)))), (R - ((((R * F.T) * c) * (V + (F * (R * F.T)))) * (c.T * (F * R.T))))) in R**(N^b_0)
b
[-3.]
```

### Automatic Re-centering and Re-scaling


Using the problem in https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/, we can automate the model reformulations that improve sample chain quality:

```python
import pandas as pd
from symbolic_pymc.utils import get_rv_observation
from symbolic_pymc.relations.distributions import scale_loc_transform

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
