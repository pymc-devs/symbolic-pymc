import string
import textwrap

import theano
import theano.tensor as tt

from copy import copy
from collections import OrderedDict
from functools import reduce

from theano import gof

from sympy import Array as SympyArray
from sympy.printing import latex as sympy_latex

from .opt import FunctionGraph
from .ops import RandomVariable
from .random_variables import Observed, NormalRV


class RandomVariablePrinter(object):
    """
    Pretty print random variables.

    `Op`s are able to specify their ascii and LaTeX formats via a "print_name"
    property.  `Op.print_name` should be a tuple or list that specifies the
    plain text/ascii and LaTeX name, respectively.

    Also, distribution parameters can be formatted distinctly by overriding
    the `RandomVariablePrinter.process_param` method.

    """

    def __init__(self, name=None):
        """
        Create a `RandomVariablePrinter`.

        Parameters
        ----------
        name: str (optional)
            A fixed name to use for the random variables printed by this
            printer.  If not specified, use `RandomVariable.name`.

        """
        self.name = name

    def process_param(self, idx, sform, pstate):
        """
        Perform special per-parameter post-formatting.

        This can be used, for instance, to change a std. dev. into a variance.

        Parameters
        ----------
        idx: int
            The index value of the parameter.
        sform: str
            The pre-formatted string form of the parameter.
        pstate: object
            The printer state.

        """
        return sform

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]

        pprinter = pstate.pprinter
        node = output.owner

        if node is None or not isinstance(node.op, RandomVariable):
            raise TypeError(
                "Function %s cannot represent a variable that is "
                "not the result of a RandomVariable operation" % self.name
            )

        op_name = self.name or getattr(node.op, "print_name", None)
        op_name = op_name or getattr(node.op, "name", None)

        if op_name is None:
            raise ValueError(f"Could not find a name for {node.op}")

        # Allow `Op`s to specify their ascii and LaTeX formats (in a tuple/list
        # with that order).
        output_latex = getattr(pstate, "latex", False)
        if isinstance(op_name, (tuple, list)):
            op_name = op_name[int(output_latex)]
        elif output_latex:
            op_name = "\\operatorname{%s}" % op_name

        preamble_dict = getattr(pstate, "preamble_dict", {})
        new_precedence = -1000
        try:
            old_precedence = getattr(pstate, "precedence", None)
            pstate.precedence = new_precedence

            # Get the symbol name string from another pprinter.
            # We create a dummy variable with no `owner`, so that
            # the pprinter will format it like a standard variable.
            dummy_out = output.clone()
            dummy_out.owner = None
            # Use this to get shape information down the line.
            dummy_out.orig_var = output

            var_name = pprinter.process(dummy_out, pstate)

            if output_latex:
                dist_format = "%s \\sim %s\\left(%s\\right)"
            else:
                dist_format = "%s ~ %s(%s)"

            # Get the shape info for our dummy symbol, if available,
            # and append it to the distribution definition.
            if "shape_strings" in preamble_dict:
                shape_info_str = preamble_dict["shape_strings"].pop(dummy_out)
                shape_info_str = shape_info_str.lstrip(var_name)
                if output_latex:
                    dist_format += "\\, {}".format(shape_info_str)
                else:
                    dist_format += shape_info_str

            dist_params = node.inputs[:-2]
            formatted_params = [
                self.process_param(i, pprinter.process(p, pstate), pstate)
                for i, p in enumerate(dist_params)
            ]

            dist_def_str = dist_format % (var_name, op_name, ", ".join(formatted_params))
        finally:
            pstate.precedence = old_precedence

        # All subsequent calls will use the variable name and
        # not the distribution definition.
        pstate.memo[output] = var_name

        if preamble_dict:
            rv_strings = preamble_dict.setdefault("rv_strings", [])
            rv_strings.append(dist_def_str)
            return var_name
        else:
            return dist_def_str


class GenericSubtensorPrinter(object):
    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print Subtensor.")

        output_latex = getattr(pstate, "latex", False)

        inputs = list(r.owner.inputs)
        obj = inputs.pop(0)
        idxs = getattr(r.owner.op, "idx_list", inputs)
        sidxs = []
        old_precedence = getattr(pstate, "precedence", None)
        try:
            pstate.precedence = -1000

            for entry in idxs:
                if isinstance(entry, slice):
                    s_parts = [""] * 2
                    if entry.start is not None:
                        s_parts[0] = entry.start

                    if entry.stop is not None:
                        s_parts[1] = entry.stop

                    if entry.step is not None:
                        s_parts.append(entry.stop)

                    sidxs.append(":".join(s_parts))
                else:
                    sidxs.append(pstate.pprinter.process(inputs.pop()))

            if output_latex:
                idx_str = ", \\,".join(sidxs)
            else:
                idx_str = ", ".join(sidxs)
        finally:
            pstate.precedence = old_precedence

        try:
            pstate.precedence = 1000
            sub = pstate.pprinter.process(obj, pstate)
        finally:
            pstate.precedence = old_precedence

        if output_latex:
            return "%s\\left[%s\\right]" % (sub, idx_str)
        else:
            return "%s[%s]" % (sub, idx_str)


class VariableWithShapePrinter(object):
    """Print variable shape info in the preamble.

    Also uses readable character names for un-named variables.

    Constant arrays are only printed when their size is below a threshold
    set by `max_line_width * max_line_height`

    """

    available_names = OrderedDict.fromkeys(string.ascii_letters)
    default_printer = theano.printing.default_printer
    max_line_width = 40
    max_line_height = 20

    @classmethod
    def process(cls, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]

        using_latex = getattr(pstate, "latex", False)
        # Crude--but effective--means of stopping print-outs for large
        # arrays.
        constant = isinstance(output, tt.TensorConstant)
        too_large = constant and (output.data.size > cls.max_line_width * cls.max_line_height)

        if constant and not too_large:
            # Print constants that aren't too large
            if using_latex and output.ndim > 0:
                out_name = sympy_latex(SympyArray(output.data))
            else:
                out_name = str(output.data)
        elif isinstance(output, tt.TensorVariable) or constant:
            # Process name and shape

            # Attempt to get the original variable, in case this is a cloned
            # `RandomVariable` output; otherwise, we won't get any shape
            # information from the `FunctionGraph`.
            var = getattr(output, "orig_var", output)

            out_name = cls.process_variable_name(var, pstate)

            shape_info = cls.process_shape_info(var, pstate)

            shape_strings = pstate.preamble_dict.setdefault("shape_strings", OrderedDict())
            shape_strings[output] = shape_info
        else:
            raise TypeError(f"Type {type(output)} not handled by variable printer")

        pstate.memo[output] = out_name
        return out_name

    @classmethod
    def process_variable_name(cls, output, pstate):
        """
        Take a variable name from the available ones.

        This function also initializes the available names by removing
        all the manually specified names within the `FunctionGraph`
        being printed (if available). Doing so removes the potential for
        name collisions.

        """
        if output in pstate.memo:
            return pstate.memo[output]

        available_names = getattr(pstate, "available_names", None)
        if available_names is None:
            # Initialize this state's available names
            available_names = copy(cls.available_names)
            fgraph = getattr(output, "fgraph", None)
            if fgraph:
                # Remove known names in the graph.
                _ = [available_names.pop(v.name, None) for v in fgraph.variables]
            setattr(pstate, "available_names", available_names)

        if output.name:
            # Observed an existing name; remove it.
            out_name = output.name
            available_names.pop(out_name, None)
        else:
            # Take an unused name.
            out_name, _ = available_names.popitem(last=False)

        pstate.memo[output] = out_name
        return out_name

    @classmethod
    def process_shape_info(cls, output, pstate):
        using_latex = getattr(pstate, "latex", False)

        if output.dtype in tt.int_dtypes:
            sspace_char = "Z"
        elif output.dtype in tt.uint_dtypes:
            sspace_char = "N"
        elif output.dtype in tt.float_dtypes:
            sspace_char = "R"
        else:
            sspace_char = "?"

        fgraph = getattr(output, "fgraph", None)
        shape_feature = None
        if fgraph:
            if not hasattr(fgraph, "shape_feature"):
                fgraph.attach_feature(tt.opt.ShapeFeature())
            shape_feature = fgraph.shape_feature

        shape_dims = []
        for i in range(output.ndim):
            s_i_out = None
            if using_latex:
                s_i_pat = "N^{%s}" + ("_{%s}" % i)
            else:
                s_i_pat = "N^%s" + ("_%s" % i)
            if shape_feature:
                new_precedence = -1000
                try:
                    old_precedence = getattr(pstate, "precedence", None)
                    pstate.precedence = new_precedence
                    _s_i_out = shape_feature.get_shape(output, i)

                    if not isinstance(_s_i_out, (tt.Constant, tt.TensorVariable)):
                        s_i_out = pstate.pprinter.process(_s_i_out, pstate)
                    else:
                        s_i_out = str(tt.get_scalar_constant_value(_s_i_out))

                except (KeyError, IndexError, ValueError, tt.NotScalarConstantError):
                    # Ugh, most of these exception types are just for
                    # `get_scalar_constant_value`!
                    # TODO: The design of that function contract could use some
                    # serious reconsideration.
                    pass
                finally:
                    pstate.precedence = old_precedence

            if not s_i_out:
                s_i_out = cls.process_variable_name(output, pstate)
                s_i_out = s_i_pat % s_i_out

            shape_dims += [s_i_out]

        shape_info = cls.process_variable_name(output, pstate)
        if using_latex:
            shape_info += " \\in \\mathbb{%s}" % sspace_char
            shape_dims = " \\times ".join(shape_dims)
            if shape_dims:
                shape_info += "^{%s}" % shape_dims
        else:
            shape_info += " in %s" % sspace_char
            shape_dims = " x ".join(shape_dims)
            if shape_dims:
                shape_info += "**(%s)" % shape_dims

        return shape_info


class PreamblePPrinter(theano.printing.PPrinter):
    r"""Pretty printer that displays a preamble.

    Preambles are put into an `OrderedDict` of categories (determined by
    printers that use the preamble).  The order can be set by preempting the
    category names within an `OrderedDict` passed to the constructor via
    the `preamble_dict` keyword.
    The lines accumulated in each category are comma-separated up to a fixed
    length given by `PreamblePPrinter.max_preamble_width`, after which a
    newline is appended and process repeats.

    Example
    -------
    >>> import theano.tensor as tt
    >>> from symbolic_pymc import NormalRV
    >>> from symbolic_pymc.printing import tt_pprint
    >>> X_rv = NormalRV(tt.scalar('\\mu'), tt.scalar('\\sigma'), name='X')
    >>> print(tt_pprint(X_rv))
    \\mu in R
    \\sigma in R
    X ~ N(\\mu, \\sigma**2),  X in R
    X

    XXX: Not thread-safe!

    """

    max_preamble_width = 40

    def __init__(self, *args, pstate_defaults=None, preamble_dict=None, **kwargs):
        """Create a `PreamblePPrinter`.

        Parameters
        ----------
        pstate_defaults: dict (optional)
            Default printer state parameters.
        preamble_dict: OrderedDict (optional)
            Default preamble dictionary.  Use this to pre-set the print-out
            ordering of preamble categories/keys.
        """
        super().__init__(*args, **kwargs)
        self.pstate_defaults = pstate_defaults or {}
        self.pstate_defaults.setdefault(
            "preamble_dict", OrderedDict() if preamble_dict is None else preamble_dict
        )
        self.printers_dict = dict(tt.pprint.printers_dict)
        self.printers = copy(tt.pprint.printers)
        self._pstate = None

    def create_state(self, pstate):
        if pstate is None:
            pstate = theano.printing.PrinterState(
                pprinter=self, **{k: copy(v) for k, v in self.pstate_defaults.items()}
            )
        elif isinstance(pstate, dict):
            pstate.update({k: copy(v) for k, v in self.pstate_defaults.items()})
            pstate = theano.printing.PrinterState(pprinter=self, **pstate)

        # FIXME: Good old fashioned circular references...
        # We're doing this so that `self.process` will be called correctly
        # accross all code.  (I'm lookin' about you, `DimShufflePrinter`; get
        # your act together.)
        pstate.pprinter._pstate = pstate

        return pstate

    def process(self, r, pstate=None):
        pstate = self._pstate
        assert pstate
        return super().process(r, pstate)

    def process_graph(self, inputs, outputs, updates=None, display_inputs=False):
        raise NotImplementedError()

    def __call__(self, *args, latex_env="equation", latex_label=None):
        in_vars = args[0]

        pstate = next(iter(args[1:]), None)
        if isinstance(pstate, (theano.printing.PrinterState, dict)):
            pstate = self.create_state(args[1])
        elif pstate is None:
            pstate = self.create_state(None)

        # This pretty printer needs more information about shapes and inputs,
        # which it gets from a `FunctionGraph`.
        fgraph = None
        out_vars = None
        if isinstance(in_vars, gof.fg.FunctionGraph):
            # We were given a `FunctionGraph` to start with; let's make sure
            # it has the shape information we need.
            fgraph = in_vars
            if not hasattr(fgraph, "shape_feature"):
                shape_feature = tt.opt.ShapeFeature()
                fgraph.attach_feature(shape_feature)
            in_vars = fgraph.inputs
            out_vars = fgraph.outputs
        elif not isinstance(in_vars, (tuple, list)):
            in_vars = [in_vars]

        if fgraph is None:
            fgraphs = [getattr(v, "fgraph", None) for v in in_vars]

            # Check that every input's `FunctionGraph` is present and the same.
            # If one of those isn't true, then we'll create one `FunctionGraph`
            # for all of the inputs.
            if all(filter(lambda x: isinstance(x, FunctionGraph), fgraphs)) and reduce(
                lambda x, y: x == y and y, fgraphs
            ):
                # Just take the first one
                fgraph = next(fgraphs)
                in_vars = fgraph.inputs
                out_vars = fgraph.outputs
            else:
                # Create a new `FunctionGraph`
                fgraph = FunctionGraph(
                    [o for o in gof.graph.inputs(in_vars) if not isinstance(o, tt.Constant)],
                    in_vars,
                    features=[tt.opt.ShapeFeature()],
                    clone=True,
                )
                in_vars = [fgraph.memo[i] for i in in_vars]
                out_vars = fgraph.outputs

        # TODO: How should this be formatted to better designate
        # the output numbers (in LaTeX, as well)?
        body_strs = []
        for v in out_vars:
            body_strs += [super().__call__(v, pstate)]

        latex_out = getattr(pstate, "latex", False)

        comma_str = ", \\," if latex_out else ", "
        newline_str = "\n\\\\\n" if latex_out else "\n"

        # Let's join all the preamble categories, but split within
        # categories when the joined line is too long.
        preamble_lines = []
        for v in pstate.preamble_dict.values():
            if isinstance(v, dict):
                v = list(v.values())

            assert isinstance(v, list)

            v_new = []
            c_len = l_idx = 0
            for l in v:
                if len(v_new) <= l_idx:
                    c_len = self.max_preamble_width * l_idx
                    v_new.append([l])
                else:
                    v_new[l_idx].append(l)
                c_len += len(l)
                l_idx += int(c_len // self.max_preamble_width > l_idx)

            preamble_lines.append(newline_str.join(comma_str.join(z) for z in v_new))

        if preamble_lines and latex_out:
            preamble_body = newline_str.join(preamble_lines)
            preamble_str = "\\begin{gathered}\n%s\n\\end{gathered}"
            preamble_str = preamble_str % (preamble_body)
            res = newline_str.join([preamble_str] + body_strs)
        else:
            res = newline_str.join(preamble_lines + body_strs)

        if latex_out and latex_env:
            label_out = f"\\label{{{latex_label}}}\n" if latex_label else ""
            res = textwrap.indent(res, "  ")
            res = f"\\begin{{{latex_env}}}\n" f"{res}\n" f"{label_out}" f"\\end{{{latex_env}}}"

        return res


tt_pprint = PreamblePPrinter()

# The order here is important!
tt_pprint.printers.insert(
    0, (lambda pstate, r: isinstance(r, tt.Variable), VariableWithShapePrinter)
)
tt_pprint.printers.insert(
    0,
    (lambda pstate, r: r.owner and isinstance(r.owner.op, RandomVariable), RandomVariablePrinter()),
)


class ObservationPrinter(object):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]

        pprinter = pstate.pprinter
        node = output.owner

        if node is None or not isinstance(node.op, Observed):
            raise TypeError(f"Node Op is not of type `Observed`: {node.op}")

        val = node.inputs[0]
        rv = node.inputs[1] if len(node.inputs) > 1 else None
        new_precedence = -1000
        try:
            old_precedence = getattr(pstate, "precedence", None)
            pstate.precedence = new_precedence

            val_name = pprinter.process(val, pstate)

            if rv:
                rv_name = pprinter.process(rv, pstate)
                out_name = f"{rv_name} = {val_name}"
            else:
                out_name = val_name
        finally:
            pstate.precedence = old_precedence

        pstate.memo[output] = out_name
        return out_name


tt_pprint.assign(Observed, ObservationPrinter())


class NormalRVPrinter(RandomVariablePrinter):
    def __init__(self):
        super().__init__("N")

    def process_param(self, idx, sform, pstate):
        if idx == 1:
            if getattr(pstate, "latex", False):
                return f"{{{sform}}}^{{2}}"
            else:
                return f"{sform}**2"
        else:
            return sform


tt_pprint.assign(NormalRV, NormalRVPrinter())
tt_pprint.assign(tt.basic._dot, theano.printing.OperatorPrinter("*", -1, "left"))

subtensor_printer = GenericSubtensorPrinter()
tt_pprint.assign(tt.Subtensor, subtensor_printer)
tt_pprint.assign(tt.AdvancedSubtensor1, subtensor_printer)
# TODO: Might need to use `isinstance` for this.
tt_pprint.assign(tt.BaseAdvancedSubtensor, subtensor_printer)

tt_tprint = PreamblePPrinter(pstate_defaults={"latex": True})
tt_tprint.printers = copy(tt_pprint.printers)
tt_tprint.printers_dict = dict(tt_pprint.printers_dict)

tt_tprint.assign(tt.basic._dot, theano.printing.OperatorPrinter("\\;", -1, "left"))
tt_tprint.assign(tt.mul, theano.printing.OperatorPrinter("\\odot", -1, "either"))
tt_tprint.assign(tt.true_div, theano.printing.PatternPrinter(("\\frac{%(0)s}{%(1)s}", -1000)))
tt_tprint.assign(tt.pow, theano.printing.PatternPrinter(("{%(0)s}^{%(1)s}", -1000)))
