import string
import textwrap

import theano
import theano.tensor as tt

from copy import copy
from collections import OrderedDict

from sympy import Array as SympyArray
from sympy.printing import latex as sympy_latex

from . import *
from .rv import RandomVariable


class RandomVariablePrinter:
    """Pretty print random variables.
    """

    def __init__(self, name=None):
        """
        Parameters
        ==========
        name: str (optional)
            A fixed name to use for the random variables printed by this
            printer.  If not specified, use `RandomVariable.name`.
        """
        self.name = name

    def process_param(self, idx, sform, pstate):
        """Special per-parameter post-formatting.

        This can be used, for instance, to change a std. dev. into a variance.

        Parameters
        ==========
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
            raise TypeError("function %s cannot represent a variable that is "
                            "not the result of a RandomVariable operation" %
                            self.name)

        new_precedence = -1000
        try:
            old_precedence = getattr(pstate, 'precedence', None)
            pstate.precedence = new_precedence
            out_name = VariableWithShapePrinter.process_variable_name(
                output, pstate)
            shape_info_str = VariableWithShapePrinter.process_shape_info(
                output, pstate)
            if getattr(pstate, 'latex', False):
                dist_format = "%s \\sim \\operatorname{%s}\\left(%s\\right)"
                dist_format += ', \\quad {}'.format(shape_info_str)
            else:
                dist_format = "%s ~ %s(%s)"
                dist_format += ',  {}'.format(shape_info_str)

            op_name = self.name or node.op.name
            dist_params = node.inputs[:-2]
            formatted_params = [
                self.process_param(i, pprinter.process(p, pstate), pstate)
                for i, p in enumerate(dist_params)
            ]

            dist_params_r = dist_format % (out_name,
                                           op_name,
                                           ", ".join(formatted_params))
        finally:
            pstate.precedence = old_precedence

        pstate.preamble_lines += [dist_params_r]
        pstate.memo[output] = out_name

        return out_name


class VariableWithShapePrinter:
    """Print variable shape info in the preamble and use readable character
    names for unamed variables.
    """
    available_names = OrderedDict.fromkeys(string.ascii_letters)
    default_printer = theano.printing.default_printer

    @classmethod
    def process(cls, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]

        using_latex = getattr(pstate, 'latex', False)

        if isinstance(output, tt.gof.Constant):
            if output.ndim > 0 and using_latex:
                out_name = sympy_latex(SympyArray(output.data))
            else:
                out_name = str(output.data)
        elif isinstance(output, tt.TensorVariable):
            # Process name and shape
            out_name = cls.process_variable_name(output, pstate)
            shape_info = cls.process_shape_info(output, pstate)
            pstate.preamble_lines += [shape_info]
        elif output.name:
            out_name = output.name
        else:
            out_name = cls.default_printer.process(output, pstate)

        pstate.memo[output] = out_name
        return out_name

    @classmethod
    def process_shape_name(cls, output, pstate):
        shape_of_var = output.owner.inputs[0]
        shape_names = pstate.memo.setdefault('shape_names', {})
        out_name = shape_names.setdefault(
            shape_of_var, cls.process_variable_name(output, pstate))
        return out_name

    @classmethod
    def process_variable_name(cls, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]

        available_names = getattr(pstate, 'available_names', None)
        if available_names is None:
            # Initialize this state's available names
            available_names = copy(cls.available_names)
            fgraph = getattr(output, 'fgraph', None)
            if fgraph:
                # Remove known names in the graph.
                _ = [available_names.pop(v.name, None)
                     for v in fgraph.variables]
            setattr(pstate, 'available_names', available_names)

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
        using_latex = getattr(pstate, 'latex', False)

        if output.dtype in tt.int_dtypes:
            sspace_char = 'Z'
        elif output.dtype in tt.uint_dtypes:
            sspace_char = 'N'
        elif output.dtype in tt.float_dtypes:
            sspace_char = 'R'
        else:
            sspace_char = '?'

        fgraph = getattr(output, 'fgraph', None)
        shape_feature = None
        if fgraph:
            if not hasattr(fgraph, 'shape_feature'):
                fgraph.attach_feature(tt.opt.ShapeFeature())
            shape_feature = fgraph.shape_feature

        shape_dims = []
        for i in range(output.ndim):
            s_i_out = None
            if using_latex:
                s_i_pat = 'N^{%s}' + ('_{%s}' % i)
            else:
                s_i_pat = 'N^%s' + ('_%s' % i)
            if shape_feature:
                new_precedence = -1000
                try:
                    old_precedence = getattr(pstate, 'precedence', None)
                    pstate.precedence = new_precedence
                    _s_i_out = shape_feature.get_shape(output, i)
                    if _s_i_out.owner:
                        if (isinstance(_s_i_out.owner.op, tt.Subtensor) and
                            all(isinstance(i, tt.Constant)
                                for i in _s_i_out.owner.inputs)):
                            s_i_out = str(_s_i_out.owner.inputs[0].data[
                                _s_i_out.owner.inputs[1].data])
                        elif not isinstance(_s_i_out, tt.TensorVariable):
                            s_i_out = pstate.pprinter.process(_s_i_out, pstate)
                except KeyError:
                    pass
                finally:
                    pstate.precedence = old_precedence

            if not s_i_out:
                s_i_out = cls.process_variable_name(output, pstate)
                s_i_out = s_i_pat % s_i_out

            shape_dims += [s_i_out]

        shape_info = cls.process_variable_name(output, pstate)
        if using_latex:
            shape_info += ' \\in \\mathbb{%s}' % sspace_char
            shape_dims = ' \\times '.join(shape_dims)
            if shape_dims:
                shape_info += '^{%s}' % shape_dims
        else:
            shape_info += ' in %s' % sspace_char
            shape_dims = ' x '.join(shape_dims)
            if shape_dims:
                shape_info += '**(%s)' % shape_dims

        return shape_info


class PreamblePPrinter(theano.printing.PPrinter):
    """Pretty printer that displays a preamble.

    For example,

        X ~ N(\\mu, \\sigma)
        (b * X)

    XXX: Not thread-safe!
    """
    def __init__(self, *args, pstate_defaults=None, **kwargs):
        """
        Parameters
        ==========
        pstate_defaults: dict (optional)
            Default printer state parameters.
        """
        super().__init__(*args, **kwargs)
        self.pstate_defaults = pstate_defaults or {}
        self.printers_dict = dict(tt.pprint.printers_dict)
        self.printers = copy(tt.pprint.printers)
        self._pstate = None

    def create_state(self, pstate):
        # FIXME: Find all the user-defined node names and make the tag
        # generator aware of them.
        if pstate is None:
            pstate = theano.printing.PrinterState(
                pprinter=self,
                preamble_lines=[],
                **self.pstate_defaults)
        elif isinstance(pstate, dict):
            pstate.setdefault('preamble_lines', [])
            pstate.update(self.pstate_defaults)
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

    def process_graph(self, inputs, outputs, updates=None,
                      display_inputs=False):
        raise NotImplemented()

    def __call__(self, *args, latex_env='equation', latex_label=None):
        var = args[0]
        pstate = next(iter(args[1:]), None)
        if isinstance(pstate, (theano.printing.PrinterState, dict)):
            pstate = self.create_state(args[1])
        elif pstate is None:
            pstate = self.create_state(None)
        # else:
        #     # XXX: The graph processing doesn't pass around the printer state!
        #     # TODO: We'll have to copy the code and fix it...
        #     raise NotImplemented('No preambles for graph printing, yet.')

        # This pretty printer needs more information about shapes and inputs,
        # which it gets from a `FunctionGraph`.  Create one, if `var` isn't
        # already assigned one.
        fgraph = getattr(var, 'fgraph', None)
        if not fgraph:
            fgraph = tt.gof.fg.FunctionGraph(
                tt.gof.graph.inputs([var]), [var])
            var = fgraph.outputs[0]

            # Use this to get better shape info
            shape_feature = tt.opt.ShapeFeature()
            fgraph.attach_feature(shape_feature)

        body_str = super().__call__(var, pstate)

        latex_out = getattr(pstate, 'latex', False)
        if pstate.preamble_lines and latex_out:
            preamble_str = "\n\\\\\n".join(pstate.preamble_lines)
            preamble_str = "\\begin{gathered}\n%s\n\\end{gathered}" % (preamble_str)
            res = "\n\\\\\n".join([preamble_str, body_str])
        else:
            res = "\n".join(pstate.preamble_lines + [body_str])

        if latex_out and latex_env:
            label_out = f'\\label{{{latex_label}}}\n' if latex_label else ''
            res = textwrap.indent(res, '\t\t')
            res = (f"\\begin{{{latex_env}}}\n"
                   f"{res}\n"
                   f"{label_out}"
                   f"\\end{{{latex_env}}}")

        return res


tt_pprint = PreamblePPrinter()

tt_pprint.assign(lambda pstate, r: True, VariableWithShapePrinter)
tt_pprint.assign(UniformRV, RandomVariablePrinter('U'))
tt_pprint.assign(GammaRV, RandomVariablePrinter('Gamma'))
tt_pprint.assign(ExponentialRV, RandomVariablePrinter('Exp'))


class NormalRVPrinter(RandomVariablePrinter):
    def __init__(self):
        super().__init__('N')

    def process_param(self, idx, sform, pstate):
        if idx == 1:
            if getattr(pstate, 'latex', False):
                return f'{{{sform}}}^{{2}}'
            else:
                return f'{sform}**2'
        else:
            return sform


tt_pprint.assign(NormalRV, NormalRVPrinter())
tt_pprint.assign(MvNormalRV, NormalRVPrinter())

tt_pprint.assign(DirichletRV, RandomVariablePrinter('Dir'))
tt_pprint.assign(PoissonRV, RandomVariablePrinter('Pois'))
tt_pprint.assign(CauchyRV, RandomVariablePrinter('C'))
tt_pprint.assign(MultinomialRV, RandomVariablePrinter('MN'))

tt_tex_pprint = PreamblePPrinter(pstate_defaults={'latex': True})
tt_tex_pprint.printers = copy(tt_pprint.printers)
tt_tex_pprint.printers_dict = dict(tt_pprint.printers_dict)
tt_tex_pprint.assign(tt.mul, theano.printing.OperatorPrinter('\\odot', -1, 'either'))
tt_tex_pprint.assign(tt.true_div, theano.printing.PatternPrinter(('\\frac{%(0)s}{%(1)s}', -1000)))
tt_tex_pprint.assign(tt.pow, theano.printing.PatternPrinter(('{%(0)s}^{%(1)s}', -1000)))
