# Symbolic PyMC

[![Build Status](https://travis-ci.org/pymc-devs/symbolic-pymc.svg?branch=master)](https://travis-ci.org/pymc-devs/symbolic-pymc) [![Coverage Status](https://coveralls.io/repos/github/pymc-devs/symbolic-pymc/badge.svg?branch=master)](https://coveralls.io/github/pymc-devs/symbolic-pymc?branch=master)


[Symbolic PyMC](https://pymc-devs.github.io/symbolic-pymc) provides tools for the symbolic manipulation of [PyMC](https://github.com/pymc-devs) models and their underlying computational graphs in [Theano](https://github.com/Theano/Theano) and [TensorFlow](https://github.com/tensorflow/tensorflow).  It enables graph manipulations in the relational DSL [miniKanren](http://minikanren.org/)&mdash;via the [`kanren`](https://github.com/logpy/logpy) package&mdash;by way of meta classes and S-expression forms of a graph.

This work stems from a series of articles starting [here](https://brandonwillard.github.io/a-role-for-symbolic-computation-in-the-general-estimation-of-statistical-models.html).  Documentation and examples for Symbolic PyMC are available [here](https://pymc-devs.github.io/symbolic-pymc).

*This package is currently in alpha, so expect large-scale changes at any time!*

## Features

### General

* Support for [Theano](https://github.com/Theano/Theano) and [TensorFlow](https://github.com/tensorflow/tensorflow) graphs
  - [Unification and reification](https://github.com/mrocklin/unification) for all components of a graph
  - A more robust Theano `Op` for representing random variables
  - Conversion of PyMC3 models into sample-able Theano graphs representing all random variable inter-dependencies
  - A Theano LaTeX pretty printer that displays shape information and distributions in mathematical notation
  - Simple text-based TensorFlow graph print-outs
* Full [miniKanren](http://minikanren.org/) integration for relational graph/model manipulation.
  - Perform simple and robust "search and replace" over arbitrary graphs (e.g. Python builtin collections, AST, tensor algebra graphs, etc.)
  - Create and compose relations with explicit high-level statistical/mathematical meaning and functionality, such as "`X` is a normal scale mixture with mixing distribution `Y`", and automatically "solve" for components (i.e. `X` and `Y`) that satisfy a relation
  - Apply non-trivial conditions&mdash;as relations&mdash;to produce sophisticated graph manipulations (e.g. search for normal scale mixtures and scale a term in the mixing distribution)
  - Integrate standard Python operations into relations (e.g. use a symbolic math library to compute an inverse-Laplace transform to determine if a distribution is a scale mixture&mdash;and find its mixing distribution)
* Convert graphs to an S-expression-like tuple-based form and perform manipulations at the syntax level
* Pre-built example relations for graph traversal, fixed-points, symbolic closed-form posteriors, and standard statistical model reformulations

## Installation

The package name is `symbolic_pymc` and it can be installed with `pip` directly from GitHub
```shell
$ pip install git+https://github.com/pymc-devs/symbolic-pymc
```
or after cloning the repo (and then installing with `pip`).
