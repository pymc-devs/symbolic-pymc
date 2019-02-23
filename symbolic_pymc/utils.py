import theano
import theano.tensor as tt

import numpy as np

import symbolic_pymc as sp

from collections import OrderedDict

from unification.utils import transitive_get as walk

from theano.gof import (FunctionGraph as tt_FunctionGraph, Query)
from theano.gof.graph import (inputs as tt_inputs, clone_get_equiv,
                              io_toposort)
from theano.compile import optdb


canonicalize_opt = optdb.query(Query(include=['canonicalize']))


def _check_eq(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b


def get_rv_observation(node):
    """Return a `RandomVariable` node's corresponding `Observed` node,
    or `None`.
    """
    if not getattr(node, 'fgraph', None):
        raise ValueError('Node does not belong to a `FunctionGraph`')

    if isinstance(node.op, sp.rv.RandomVariable):
        fgraph = node.fgraph
        for o, i in node.default_output().clients:
            if o == 'output':
                o = fgraph.outputs[i].owner

            if isinstance(o.op, sp.Observed):
                return o
    return None


def replace_input_nodes(inputs, outputs, replacements=None,
                        memo=None, clone_inputs=True):
    """Recreate a graph, replacing input variables according to a given map.

    This is helpful if you want to replace the variable dependencies of
    an existing variable according to a `clone_get_equiv` map and/or
    replacement variables that already exist within a `FunctionGraph`.

    The latter is especially annoying, because you can't simply make a
    `FunctionGraph` for the variable to be adjusted and then use that to
    perform the replacement; if the variables to be replaced are already in a
    `FunctionGraph` any such replacement will err-out saying "...these
    variables are already owned by another graph..."

    Parameters
    ==========
    inputs: list
        List of input nodes.
    outputs: list
        List of output nodes.  Everything between `inputs` and these `outputs`
        is the graph under consideration.
    replacements: dict (optional)
        A dictionary mapping existing nodes to their new ones.
        These values in this map will be used instead of newly generated
        clones.  This dict is not altered.
    memo: dict (optional)
        A dictionary to update with the initial `replacements` and maps from
        any old-to-new nodes arising from an actual replacement.
        It serves the same role as `replacements`, but it is updated
        as elements are cloned.
    clone_inputs: bool (optional)
        If enabled, clone all the input nodes that aren't mapped in
        `replacements`.  These cloned nodes are mapped in `memo`, as well.

    Results
    =======
    out: memo
    """
    if memo is None:
        memo = {}
    if replacements is not None:
        memo.update(replacements)
    for apply in io_toposort(inputs, outputs):

        walked_inputs = []
        for i in apply.inputs:
            if clone_inputs:
                # TODO: What if all the inputs are in the memo?
                walked_inputs.append(memo.setdefault(i, i.clone()))
            else:
                walked_inputs.append(walk(i, memo))

        if any(w != i for w, i in zip(apply.inputs, walked_inputs)):
            new_apply = apply.clone_with_new_inputs(walked_inputs)

            memo.setdefault(apply, new_apply)
            for output, new_output in zip(apply.outputs, new_apply.outputs):
                memo.setdefault(output, new_output)
    return memo


def meta_parts_unequal(x, y, pdb=False):
    """Traverse meta objects and return the first pair of elements
    that are not equal.
    """
    res = None
    if type(x) != type(y):
        print('unequal types')
        res = (x, y)
    elif isinstance(x, sp.meta.MetaSymbol):
        if x.base != y.base:
            print('unequal bases')
            res = (x.base, y.base)
        else:
            for a, b in zip(x.rands(), y.rands()):
                z = meta_parts_unequal(a, b, pdb=pdb)
                if z is not None:
                    res = z
                    break
    elif isinstance(x, (tuple, list)):
        for a, b in zip(x, y):
            z = meta_parts_unequal(a, b, pdb=pdb)
            if z is not None:
                res = z
                break
    elif not _check_eq(x, y):
        res = (x, y)

    if res is not None:
        if pdb:
            import pdb; pdb.set_trace()
        return res


def expand_meta(x, tt_print=tt.pprint):
    """Produce a dictionary representation of a meta object."""
    if isinstance(x, sp.meta.MetaSymbol):
        return OrderedDict([('rator', x.base),
                            ('rands', tuple(expand_meta(p)
                                            for p in x.rands())),
                            ('obj', expand_meta(getattr(x, 'obj', None)))])
    elif tt_print and isinstance(x, theano.gof.op.Op):
        return x.name
    elif tt_print and isinstance(x, theano.gof.graph.Variable):
        return tt_print(x)
    else:
        return x


def graph_equal(x, y):
    """Compare elements in a Theano graph using their object properties and not
    just identity.
    """
    try:
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            return (len(x) == len(y) and
                    all(sp.mt(xx) == sp.mt(yy)
                        for xx, yy in zip(x, y)))
        return sp.mt(x) == sp.mt(y)
    except ValueError:
        return False


def mt_type_params(x):
    return {'ttype': x.type, 'index': x.index, 'name': x.name}


def optimize_graph(x, optimization, return_graph=None, in_place=False):
    """Apply an optimization to either the graph formed by a Theano variable or
    an existing graph and return the resulting optimized graph.

    When given an existing `FunctionGraph`, the optimization is performed
    without side-effects (i.e. won't change the given graph).
    """
    if not isinstance(x, tt_FunctionGraph):
        inputs = tt_inputs([x])
        outputs = [x]
        model_memo = clone_get_equiv(inputs, outputs,
                                     copy_orphans=False)
        cloned_inputs = [model_memo[i] for i in inputs
                         if not isinstance(i, tt.Constant)]
        cloned_outputs = [model_memo[i] for i in outputs]

        x_graph = sp.opt.FunctionGraph(cloned_inputs, cloned_outputs,
                                       clone=False)
        x_graph.memo = model_memo

        if return_graph is None:
            return_graph = False
    else:
        x_graph = x

        if return_graph is None:
            return_graph = True

    x_graph_opt = x_graph if in_place else x_graph.clone()
    _ = optimization.optimize(x_graph_opt)

    if return_graph:
        res = x_graph_opt
    else:
        res = x_graph_opt.outputs
        if len(res) == 1:
            res, = res
    return res


def canonicalize(x, **kwargs):
    """Canonicalize a Theano variable and/or graph.
    """
    return optimize_graph(x, canonicalize_opt, **kwargs)
