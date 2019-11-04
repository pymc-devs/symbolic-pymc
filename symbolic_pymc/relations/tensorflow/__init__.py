from unification import var

from kanren.facts import fact
from kanren.assoccomm import commutative, associative

from ...tensorflow.meta import mt, TFlowMetaOperator


# TODO: We could use `mt.*.op_def.obj.is_commutative` to capture
# more/all cases.
fact(commutative, TFlowMetaOperator(mt.AddV2.op_def, var()))
fact(commutative, TFlowMetaOperator(mt.AddN.op_def, var()))
fact(commutative, TFlowMetaOperator(mt.Mul.op_def, var()))

fact(associative, TFlowMetaOperator(mt.AddN.op_def, var()))
fact(associative, TFlowMetaOperator(mt.AddV2.op_def, var()))
