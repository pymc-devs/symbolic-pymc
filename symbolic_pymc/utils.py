import theano
import theano.tensor as tt

from collections import OrderedDict

from theano.gof import FunctionGraph
from theano.gof.graph import inputs as tt_inputs, clone_get_equiv

from .meta import MetaSymbol


to_meta = MetaSymbol.from_obj


def parts_unequal(x, y):
    if type(x) != type(y):
        print('unequal types')
        return x, y
    elif isinstance(x, MetaSymbol):
        if x.base != y.base:
            print('unequal bases')
            return x.base, y.base
        for a, b in zip(x.rands(), y.rands()):
            z = parts_unequal(a, b)
            if z:
                return z
    elif x != y:
        return x, y


def expand_meta(x, tt_print=tt.pprint):
    """Produce a dictionary representation of a meta object."""
    if isinstance(x, MetaSymbol):
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
                    all(MetaSymbol.from_obj(xx) == MetaSymbol.from_obj(yy)
                        for xx, yy in zip(x, y)))
        return MetaSymbol.from_obj(x) == MetaSymbol.from_obj(y)
    except ValueError:
        return False


def mt_type_params(x):
    return {'ttype': x.type, 'index': x.index, 'name': x.name}


def optimize_graph(x, optimization):
    """Apply an optimization to either the graph formed by a Theano variable or
    an existing graph and return the resulting optimized graph.

    When given an existing `FunctionGraph`, the optimization is performed
    without side-effects (i.e. won't change the given graph).
    """
    if not isinstance(x, FunctionGraph):
        inputs = tt_inputs([x])
        outputs = [x]
        model_memo = clone_get_equiv(inputs, outputs,
                                     copy_orphans=False)
        cloned_inputs = [model_memo[i] for i in inputs]
        cloned_outputs = [model_memo[i] for i in outputs]

        x_graph = FunctionGraph(cloned_inputs, cloned_outputs, clone=False)
        x_graph.memo = model_memo
    else:
        x_graph = x

    x_graph_opt = x_graph.clone()
    optimization.optimize(x_graph_opt)
    return x_graph_opt.outputs[0]
