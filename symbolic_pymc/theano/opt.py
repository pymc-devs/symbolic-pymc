import types

import theano
import theano.tensor as tt

from functools import wraps

from theano.gof.opt import LocalOptimizer, local_optimizer
from theano.scan_module.scan_op import Scan
from theano.scan_module.scan_utils import scan_args as ScanArgs

from unification import var, variables

from kanren import run

from etuples.core import ExpressionTuple

from .meta import MetaSymbol
from .ops import RandomVariable


def eval_and_reify_meta(x):
    """Get Theano objects from combinations of `etuple`s and meta objects."""
    res = x

    # Create base objects from the resulting meta object
    if isinstance(res, ExpressionTuple):
        res = res.eval_obj

    if isinstance(res, MetaSymbol):
        res = res.reify()

    if MetaSymbol.is_meta(res):
        raise ValueError("Kanren results not fully reifiable: {}".format(res))

    return res


class FunctionGraph(theano.gof.fg.FunctionGraph):
    """A version of `FunctionGraph` that knows not to merge non-deterministic `Op`s.

    It also performs common operations for input cloning and keeps a map of
    original/cloned nodes, making the graph much easier to use.

    """

    def __init__(
        self,
        inputs,
        outputs,
        features=None,
        clone=True,
        memo=None,
        update_mapping=None,
        copy_inputs=True,
        copy_orphans=None,
    ):

        if clone:
            if copy_orphans is None:
                copy_orphans = copy_inputs

            self.memo = theano.gof.graph.clone_get_equiv(
                inputs, outputs, copy_inputs, copy_orphans, memo
            )

            inputs = [self.memo[i] for i in inputs]
            outputs = [self.memo[o] for o in outputs]

        super().__init__(inputs, outputs, features=features, clone=False, update_mapping=None)

    def attach_feature(self, feature):
        if isinstance(feature, theano.gof.opt.MergeFeature):
            _process_node = feature.process_node

            @wraps(feature.process_node)
            def _f(self, fgraph, node):
                if getattr(node.op, "nondeterministic", False):
                    return
                return _process_node(fgraph, node)

            feature.process_node = types.MethodType(_f, feature)

        return super().attach_feature(feature)

    def replace(self, r, new_r, reason=None, verbose=None, remove_dup_inputs=True):
        """See `theano.gof.fg.FunctionGraph.replace`.

        The original `FunctionGraph.replace` will not replace the actual
        input list.  This one will.

        """
        super().replace(r, new_r, reason=reason, verbose=verbose)

        if r in self.inputs:
            # TODO: Is there a reason to do this in-place instead?
            # Is anyone supposed to hold a reference to the original inputs
            # list?

            # Remove duplicate inputs, if any.
            if remove_dup_inputs and new_r in self.inputs:
                self.inputs.remove(new_r)

            assert r not in self.variables

            new_inputs = [new_r if i == r else i for i in self.inputs]
            self.inputs = new_inputs

            # TODO: Inputs-changed callback?

            assert r not in self.inputs

    def clone_get_equiv(self, *args, **kwargs):
        fg, var_map = super().clone_get_equiv(*args, **kwargs)
        fg.__class__ = self.__class__
        return fg, var_map


class KanrenRelationSub(LocalOptimizer):
    """A local optimizer that uses miniKanren goals to match and replace terms in a Theano `FunctionGraph`.

    TODO: Only uses *one* miniKanren `run` result (chosen by a configurable
    filter function).  We might want an option to produce multiple graphs, but
    I imagine that would involve an entirely different optimizer type.

    """

    reentrant = True

    def __init__(
        self,
        kanren_relation,
        relation_lvars=None,
        results_filter=lambda x: next(x, None),
        node_filter=lambda x: False,
    ):
        """Create a `KanrenRelationSub`.

        Parameters
        ----------
        kanren_relation: kanren.Relation or goal
            The miniKanren relation store or goal to use.  Custom goals should
            take an input and output argument, respectively.
        relation_lvars: Iterable
            A collection of terms to be considered logic variables by miniKanren
            (i.e. Theano terms used as "unknowns" in `kanren_relation`).
        results_filter: function
            A function that returns a single result from a stream of
            miniKanren results.  The default function returns the first result.
        node_filter: function
            A function taking a single node as an argument that returns `True`
            when the node should be skipped.
        """
        self.kanren_relation = kanren_relation
        self.relation_lvars = relation_lvars or []
        self.results_filter = results_filter
        self.node_filter = node_filter
        super().__init__()

    def adjust_outputs(self, node, new_node, old_node=None):
        """Make adjustments for multiple outputs.

        This handles (some) nodes with multiple outputs by returning a list
        with the appropriate length and containing the new node (at the correct
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
            new_node_idx = getattr(node.op, "default_output", 0) or 0

        res[new_node_idx] = new_node
        return res

    def transform(self, node):
        if not isinstance(node, tt.Apply):
            return False

        if self.node_filter(node):
            return False

        try:
            input_expr = node.default_output()
        except AttributeError:
            input_expr = node.outputs

        with variables(*self.relation_lvars):
            q = var()
            kanren_results = run(None, q, self.kanren_relation(input_expr, q))

        chosen_res = self.results_filter(kanren_results)

        if chosen_res:
            if isinstance(chosen_res, ExpressionTuple):
                chosen_res = eval_and_reify_meta(chosen_res)

            if isinstance(chosen_res, dict):
                chosen_res = list(chosen_res.items())

            if isinstance(chosen_res, list):
                # We got a dictionary of replacements
                new_node = {eval_and_reify_meta(k): eval_and_reify_meta(v) for k, v in chosen_res}

                assert all(k in node.fgraph.variables for k in new_node.keys())
            elif isinstance(chosen_res, tt.Variable):
                # Attempt to automatically format the output for multi-output
                # `Apply` nodes.
                new_node = self.adjust_outputs(node, eval_and_reify_meta(chosen_res))
            else:
                raise ValueError(
                    "Unsupported FunctionGraph replacement variable type: {chosen_res}"
                )

            return new_node
        else:
            return False


@local_optimizer([Scan])
def push_out_rvs_from_scan(node):
    """Push `RandomVariable`s out of `Scan` nodes.

    When `RandomVariable`s are created within the inner-graph of a `Scan` and
    are not output to the outer-graph, we "push" them out of the inner-graph.
    This helps us produce an outer-graph in which all the relevant `RandomVariable`s
    are accessible (e.g. for constructing a log-likelihood graph).
    """
    scan_args = ScanArgs(node.inputs, node.outputs, node.op.inputs, node.op.outputs, node.op.info)

    # Find the un-output `RandomVariable`s created in the inner-graph
    clients = {}
    local_fgraph_topo = theano.gof.graph.io_toposort(
        scan_args.inner_inputs, scan_args.inner_outputs, clients=clients
    )
    unpushed_inner_rvs = []
    for n in local_fgraph_topo:
        if isinstance(n.op, RandomVariable):
            unpushed_inner_rvs.extend([c for c in clients[n] if c not in scan_args.inner_outputs])

    if len(unpushed_inner_rvs) == 0:
        return False

    # Add the new outputs to the inner and outer graphs
    scan_args.inner_out_nit_sot.extend(unpushed_inner_rvs)

    assert len(scan_args.outer_in_nit_sot) > 0, "No outer-graph inputs are nit-sots!"

    # Just like `theano.scan`, we simply copy/repeat the existing nit-sot
    # outer-graph input value, which represents the actual size of the output
    # tensors.  Apparently, the value needs to be duplicated for all nit-sots.
    # FYI: This is what increments the nit-sot values in `scan_args.info`, as
    # well.
    # TODO: Can we just use `scan_args.n_steps`?
    scan_args.outer_in_nit_sot.extend(scan_args.outer_in_nit_sot[0:1] * len(unpushed_inner_rvs))

    op = Scan(scan_args.inner_inputs, scan_args.inner_outputs, scan_args.info)
    outputs = list(op(*scan_args.outer_inputs))

    # Return only the replacements for the original `node.outputs`
    new_inner_out_idx = [scan_args.inner_outputs.index(i) for i in unpushed_inner_rvs]
    _ = [outputs.pop(op.var_mappings["outer_out_from_inner_out"][i]) for i in new_inner_out_idx]

    return dict(zip(node.outputs, outputs))
