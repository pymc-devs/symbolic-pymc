import types

import theano
import theano.tensor as tt

from functools import wraps
from unification import var, variables

from kanren import run
from kanren.core import evalt

from theano.gof.opt import LocalOptimizer

from .meta import MetaSymbol
from .unify import reify_all_terms


def reify_meta(x):

    # Evaluate tuple-form expressions
    res = reify_all_terms(x)

    # Create base objects from the resulting meta object
    if isinstance(res, MetaSymbol):
        res = res.reify()

    if MetaSymbol.is_meta(res) or isinstance(res, (list, tuple)):
        raise ValueError(
            "Kanren results not fully reifiable: {}".format(res))

    return res


class FunctionGraph(theano.gof.fg.FunctionGraph):
    """A version of `FunctionGraph` that knows not to merge
    non-deterministic `Op`s.

    TODO: Add a check to `MergeFeature.process_node` and submit
    a PR to Theano.
    """

    def attach_feature(self, feature):
        if isinstance(feature, theano.gof.opt.MergeFeature):
            _process_node = feature.process_node

            @wraps(feature.process_node)
            def _f(self, fgraph, node):
                if getattr(node.op, 'nondeterministic', False):
                    return
                return _process_node(fgraph, node)

            feature.process_node = types.MethodType(_f, feature)

        return super().attach_feature(feature)

    def clone(self, *args, **kwargs):
        res = super().clone(*args, **kwargs)
        res.__class__ = type(self)
        return res


class KanrenRelationSub(LocalOptimizer):
    """A local optimizer that uses miniKanren goals to match and replace
    terms in a Theano `FunctionGraph`.


    TODO: Only uses *one* miniKanren `run` result (chosen by a configurable
    filter function).  We might want an option to produce multiple graphs, but
    I imagine that would involve an entirely different optimizer type.
    """
    reentrant = True

    def __init__(self, kanren_relation, relation_lvars=None,
                 results_filter=lambda x: next(iter(x), None)):
        """
        Parameters
        ==========
        kanren_relation: kanren.Relation or goal
            The miniKanren relation store or goal to use.  Custom goals should
            take an input and output argument, respectively.
        relation_lvars: Iterable
            A collection of terms to be considered logic variables by miniKanren
            (i.e. Theano terms used as "unknowns" in `kanren_relation`).
        results_filter: function
            A function that returns a single result from a stream of
            miniKanren results.  The default function returns the first result.
        """
        self.kanren_relation = kanren_relation
        self.relation_lvars = relation_lvars or []
        self.results_filter = results_filter
        super().__init__()

    def adjust_outputs(self, node, new_node, old_node=None):
        """Handle (some) nodes with multiple outputs by returning a list with
        the appropriate length and containing the new node (at the correct
        index if `default_output` is available and correct, or 0--and it
        happens to be the correct one).

        TODO: We should be able to get the correct index from the something
        like `node.outputs.index(old_node)`, but we don't exactly have
        `old_node` unless the miniKanren results give it to us.
        """
        res = list(node.outputs)
        try:
            new_node_idx = res.index(old_node)
        except ValueError:
            # Guesstimate it
            new_node_idx = getattr(node.op, 'default_output', 0) or 0

        res[new_node_idx] = new_node
        return res

    def transform(self, node):
        if not isinstance(node, tt.Apply):
            return False

        input_expr = node.default_output()

        with variables(*self.relation_lvars):
            q = var()
            kanren_results = run(1, q, (self.kanren_relation, input_expr, q))

        chosen_res = self.results_filter(kanren_results)

        if chosen_res:
            # Turn the meta objects and tuple-form expressions into Theano
            # objects.
            if isinstance(chosen_res, tuple) and chosen_res[0] == dict:
                # We got a dictionary of replacements.
                new_node = {k.obj: reify_meta(v)
                            for k, v in evalt(chosen_res).items()}

                assert all(k in node.fgraph.variables
                           for k in new_node)
            else:
                new_node = self.adjust_outputs(node, reify_meta(chosen_res))

            return new_node
        else:
            return False
