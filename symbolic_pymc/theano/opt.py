import types

import theano
import theano.tensor as tt

from functools import wraps
from unittest.mock import patch
from collections import namedtuple, OrderedDict

from theano.gof.opt import LocalOptimizer, local_optimizer
from theano.gof.graph import inputs as tt_inputs
from theano.scan_module.scan_op import Scan
from theano.scan_module.scan_utils import scan_args

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


def safe_index(lst, x):
    try:
        return lst.index(x)
    except ValueError:
        return None


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
                )  # pragma: no cover

            return new_node
        else:
            return False


FieldInfo = namedtuple("FieldInfo", ("name", "agg_name", "index", "inner_index", "agg_index"))


class ScanArgs(scan_args):
    """An improved version of `theano.scan_module.scan_utils`."""

    default_filter = lambda x: x.startswith("inner_") or x.startswith("outer_")
    nested_list_fields = ("inner_in_mit_mot", "inner_in_mit_sot", "inner_out_mit_mot")

    def __init__(self, *args, **kwargs):
        # Prevent unnecessary and counter-productive cloning.
        # If you want to clone the inner graph, do it before you call this!
        with patch(
            "theano.scan_module.scan_utils.reconstruct_graph",
            side_effect=lambda x, y, z=None: [x, y],
        ):
            super().__init__(*args, **kwargs)

    @staticmethod
    def from_node(node):
        if not isinstance(node.op, Scan):
            raise TypeError("{} is not a Scan node".format(node))
        return ScanArgs(node.inputs, node.outputs, node.op.inputs, node.op.outputs, node.op.info)

    @classmethod
    def create_empty(cls):
        info = OrderedDict(
            [
                ("n_seqs", 0),
                ("n_mit_mot", 0),
                ("n_mit_sot", 0),
                ("tap_array", []),
                ("n_sit_sot", 0),
                ("n_nit_sot", 0),
                ("n_shared_outs", 0),
                ("n_mit_mot_outs", 0),
                ("mit_mot_out_slices", []),
                ("truncate_gradient", -1),
                ("name", None),
                ("mode", None),
                ("destroy_map", OrderedDict()),
                ("gpua", False),
                ("as_while", False),
                ("profile", False),
                ("allow_gc", False),
            ]
        )
        res = cls([1], [], [], [], info)
        res.n_steps = None
        return res

    @property
    def n_nit_sot(self):
        # This is just a hack that allows us to use `Scan.get_oinp_iinp_iout_oout_mappings`
        return self.info["n_nit_sot"]

    @property
    def inputs(self):
        # This is just a hack that allows us to use `Scan.get_oinp_iinp_iout_oout_mappings`
        return self.inner_inputs

    @property
    def n_mit_mot(self):
        # This is just a hack that allows us to use `Scan.get_oinp_iinp_iout_oout_mappings`
        return self.info["n_mit_mot"]

    @property
    def var_mappings(self):
        return Scan.get_oinp_iinp_iout_oout_mappings(self)

    @property
    def field_names(self):
        res = ["mit_mot_out_slices", "mit_mot_in_slices", "mit_sot_in_slices"]
        res.extend(
            [
                attr
                for attr in self.__dict__
                if attr.startswith("inner_in")
                or attr.startswith("inner_out")
                or attr.startswith("outer_in")
                or attr.startswith("outer_out")
                or attr == "n_steps"
            ]
        )
        return res

    def get_alt_field(self, var_info, alt_prefix):
        """Get the alternate input/output field for a given element of `ScanArgs`.

        For example, if `var_info` is in `ScanArgs.outer_out_sit_sot`, then
        `get_alt_field(var_info, "inner_out")` returns the element corresponding
        `var_info` in `ScanArgs.inner_out_sit_sot`.

        Parameters
        ----------
        var_info: TensorVariable or FieldInfo
            The element for which we want the alternate
        alt_prefix: str
            The string prefix for the alternate field type.  It can be one of
            the following: "inner_out", "inner_in", "outer_in", and "outer_out".

        Outputs
        -------
        TensorVariable
        Returns the alternate variable.

        """
        if not isinstance(var_info, FieldInfo):
            var_info = self.find_among_fields(var_info)

        alt_type = var_info.name[(var_info.name.index("_", 6) + 1) :]
        alt_var = getattr(self, "inner_out_{}".format(alt_type))[var_info.index]
        return alt_var

    def find_among_fields(self, i, field_filter=default_filter):
        """Find the type and indices of the field containing a given element.

        NOTE: This only returns the *first* field containing the given element.

        Parameters
        ----------
        i: theano.gof.graph.Variable
            The element to find among this object's fields.
        field_filter: function
            A function passed to `filter` that determines which fields to
            consider.  It must take a string field name and return a truthy
            value.

        Returns
        -------
        A tuple of length 4 containing the field name string, the first index,
        the second index (for nested lists), and the "major" index (i.e. the
        index within the aggregate lists like `self.inner_inputs`,
        `self.outer_outputs`, etc.), or a triple of `None` when no match is
        found.

        """

        field_names = filter(field_filter, self.field_names)

        for field_name in field_names:
            lst = getattr(self, field_name)

            field_prefix = field_name[:8]
            if field_prefix.endswith("in"):
                agg_field_name = "{}puts".format(field_prefix)
            else:
                agg_field_name = "{}tputs".format(field_prefix)

            agg_list = getattr(self, agg_field_name)

            if field_name in self.nested_list_fields:
                for n, sub_lst in enumerate(lst):
                    idx = safe_index(sub_lst, i)
                    if idx is not None:
                        agg_idx = safe_index(agg_list, i)
                        return FieldInfo(field_name, agg_field_name, n, idx, agg_idx)
            else:
                idx = safe_index(lst, i)
                if idx is not None:
                    agg_idx = safe_index(agg_list, i)
                    return FieldInfo(field_name, agg_field_name, idx, None, agg_idx)

        return None

    def _remove_from_fields(self, i, field_filter=default_filter):

        field_info = self.find_among_fields(i, field_filter=field_filter)

        if field_info is None:
            return None

        if field_info.inner_index is not None:
            getattr(self, field_info.name)[field_info.index].remove(i)
        else:
            getattr(self, field_info.name).remove(i)

        return field_info

    def get_dependent_nodes(self, i, seen=None):

        if seen is None:
            seen = {i}
        else:
            seen.add(i)

        var_mappings = self.var_mappings

        field_info = self.find_among_fields(i)

        if field_info is None:
            raise ValueError("{} not found among fields.".format(i))

        # Find the `var_mappings` key suffix that matches the field/set of
        # arguments containing our source node
        if field_info.name[:8].endswith("_in"):
            map_key_suffix = "{}p".format(field_info.name[:8])
        else:
            map_key_suffix = field_info.name[:9]

        dependent_nodes = set()
        for k, v in var_mappings.items():

            if not k.endswith(map_key_suffix):
                continue

            dependent_idx = v[field_info.agg_index]
            dependent_idx = dependent_idx if isinstance(dependent_idx, list) else [dependent_idx]

            # Get the `ScanArgs` field name for the aggregate list property
            # corresponding to these dependent argument types (i.e. either
            # "outer_inputs", "inner_inputs", "inner_outputs", or
            # "outer_outputs").
            # To do this, we need to parse the "shared" prefix of the
            # current `var_mappings` key and append the missing parts so that
            # it either forms `"*_inputs"` or `"*_outputs"`.
            to_agg_field_prefix = k[:9]
            if to_agg_field_prefix.endswith("p"):
                to_agg_field_name = "{}uts".format(to_agg_field_prefix)
            else:
                to_agg_field_name = "{}puts".format(to_agg_field_prefix)

            to_agg_field = getattr(self, to_agg_field_name)

            for d_id in dependent_idx:
                if d_id < 0:
                    continue

                dependent_var = to_agg_field[d_id]

                if dependent_var not in seen:
                    dependent_nodes.add(dependent_var)

        if field_info.name.startswith("inner_in"):
            # If starting from an inner-input, then we need to find any
            # inner-outputs that depend on it.
            for out_n in self.inner_outputs:
                if i in tt_inputs([out_n]):
                    if out_n not in seen:
                        dependent_nodes.add(out_n)

        for n in tuple(dependent_nodes):
            if n in seen:
                continue
            sub_dependent_nodes = self.get_dependent_nodes(n, seen=seen)
            dependent_nodes |= sub_dependent_nodes
            seen |= sub_dependent_nodes

        return dependent_nodes

    def remove_from_fields(self, i, rm_dependents=True):

        if rm_dependents:
            vars_to_remove = self.get_dependent_nodes(i) | {i}
        else:
            vars_to_remove = {i}

        rm_info = []
        for v in vars_to_remove:
            dependent_rm_info = self._remove_from_fields(v)
            rm_info.append((v, dependent_rm_info))

        return rm_info

    def __str__(self):
        inner_arg_strs = [
            "\t{}={}".format(p, getattr(self, p))
            for p in self.field_names
            if p.startswith("outer_in") or p == "n_steps"
        ]
        inner_arg_strs += [
            "\t{}={}".format(p, getattr(self, p))
            for p in self.field_names
            if p.startswith("inner_in")
        ]
        inner_arg_strs += [
            "\tmit_mot_in_slices={}".format(self.mit_mot_in_slices),
            "\tmit_sot_in_slices={}".format(self.mit_sot_in_slices),
        ]
        inner_arg_strs += [
            "\t{}={}".format(p, getattr(self, p))
            for p in self.field_names
            if p.startswith("inner_out")
        ]
        inner_arg_strs += [
            "\tmit_mot_out_slices={}".format(self.mit_mot_out_slices),
        ]
        inner_arg_strs += [
            "\t{}={}".format(p, getattr(self, p))
            for p in self.field_names
            if p.startswith("outer_out")
        ]
        res = "ScanArgs(\n{})".format(",\n".join(inner_arg_strs))
        return res

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        for field_name in self.field_names:
            if not hasattr(other, field_name) or getattr(self, field_name) != getattr(
                other, field_name
            ):
                return False

        return True


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
