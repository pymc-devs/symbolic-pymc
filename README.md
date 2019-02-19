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


## Usage

### Convert a PyMC3 model to a Theano graph

```python
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm

from symbolic_pymc.pymc3 import model_graph

theano.config.cxx = ''

mu_X = tt.scalar('mu_X')
sd_X = tt.scalar('sd_X')
mu_Y = tt.scalar('mu_Y')
sd_Y = tt.scalar('sd_Y')
mu_X.tag.test_value = np.array(0., dtype=tt.config.floatX)
sd_X.tag.test_value = np.array(1., dtype=tt.config.floatX)
mu_Y.tag.test_value = np.array(1., dtype=tt.config.floatX)
sd_Y.tag.test_value = np.array(0.5, dtype=tt.config.floatX)

with pm.Model() as model:
    X_rv = pm.Normal('X_rv', mu_X, sd=sd_X)
    Y_rv = pm.Normal('Y_rv', mu_Y, sd=sd_Y)
    Z_rv = pm.Normal('Z_rv',
                     X_rv + Y_rv,
                     sd=sd_X + sd_Y,
                     observed=10.)

fgraph = model_graph(model)
fgraph = canonicalize(fgraph)
```
```python
>>> from theano.printing import debugprint as tt_dprint
>>> tt_dprint(fgraph)
<symbolic_pymc.Observed object at 0x7f0e50556a90> [id A] ''
 |TensorConstant{10.0} [id B]
 |normal_rv.1 [id C] 'Z_rv'
   |Elemwise{add,no_inplace} [id D] ''
   | |normal_rv.1 [id E] 'X_rv'
   | | |mu_X [id F]
   | | |sd_X [id G]
   | | |TensorConstant{[]} [id H]
   | | |<RandomStateType> [id I]
   | |normal_rv.1 [id J] 'Y_rv'
   |   |mu_Y [id K]
   |   |sd_Y [id L]
   |   |TensorConstant{[]} [id H]
   |   |<RandomStateType> [id I]
   |Elemwise{add,no_inplace} [id M] ''
   | |sd_X [id G]
   | |sd_Y [id L]
   |TensorConstant{[]} [id H]
   |<RandomStateType> [id I]

```
### Mathematical Notation Pretty Printing

```python
>>> from symbolic_pymc.printing import tt_pprint
>>> print(tt_pprint(fgraph))
mu_X in R
sd_X in R
X_rv ~ N(mu_X, sd_X**2),  X_rv in R
mu_Y in R
sd_Y in R
Y_rv ~ N(mu_Y, sd_Y**2),  Y_rv in R
Z_rv ~ N((X_rv + Y_rv), (sd_X + sd_Y)**2),  Z_rv in R
Z_rv = 10.0
```

```python
>>> from symbolic_pymc.printing import tt_tprint
>>> print(tt_tprint(fgraph))
```
produces
```latex
\begin{equation}
  \begin{gathered}
  mu_X \in \mathbb{R}
  \\
  sd_X \in \mathbb{R}
  \\
  X_rv \sim \operatorname{N}\left(mu_X, {sd_X}^{2}\right), \quad X_rv \in \mathbb{R}
  \\
  mu_Y \in \mathbb{R}
  \\
  sd_Y \in \mathbb{R}
  \\
  Y_rv \sim \operatorname{N}\left(mu_Y, {sd_Y}^{2}\right), \quad Y_rv \in \mathbb{R}
  \\
  Z_rv \sim \operatorname{N}\left((X_rv + Y_rv), {(sd_X + sd_Y)}^{2}\right), \quad Z_rv \in \mathbb{R}
  \end{gathered}
  \\
  Z_rv = 10.0
\end{equation}
```

### Compute Symbolic Closed-form Posteriors

```python
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm

from theano.gof.opt import EquilibriumOptimizer

from symbolic_pymc.pymc3 import model_graph
from symbolic_pymc.utils import optimize_graph, canonicalize
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

fgraph_opt = optimize_graph(fgraph, posterior_opt, return_graph=True)
fgraph_opt = canonicalize(fgraph_opt)
```

```python
>>> print(tt_pprint(fgraph_opt[0]))
a in R**(N^a_0)
R in R**(N^R_0 x N^R_1)
F in R**(N^F_0 x N^F_1)
c in R**(N^c_0 x N^c_1)
V in R**(N^V_0 x N^V_1)
b ~ N((a + (((R * F.T) * c) * ([-3.] - (F * a)))), (R - ((((R * F.T) * c) * (V + (F * (R * F.T)))) * (c.T * (F * R.T))))),  b in R**(N^b_0)
b
```

## Installation

The package name is `symbolic_pymc` and it can be installed with `pip` directly from
GitHub, or after cloning the repo.
