import pytest

from cons import car, cdr
from cons.core import ConsError

from etuples import rator, rands
from etuples.core import ExpressionTuple

from unification import var, unify, reify

from symbolic_pymc.meta import MetaVariable, MetaOp


class SomeOp(object):
    def __repr__(self):
        return "<SomeOp>"


class SomeType(object):
    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2

    def __repr__(self):
        return f"SomeType({self.field1}, {self.field2})"

    def __str__(self):
        return f"SomeType<{self.field1}, {self.field2}>"


class SomeMetaOp(MetaOp):
    __slots__ = ()
    base = SomeOp

    def output_meta_types(self):
        return [SomeMetaVariable]

    def __call__(self, *args, **kwargs):
        return SomeMetaVariable(*args, **kwargs)


class SomeOtherMetaOp(SomeMetaOp):
    pass


class SomeMetaVariable(MetaVariable):
    __slots__ = ("op", "args")
    base = SomeType

    def __init__(self, op, args, obj=None):
        super().__init__(obj)
        self.op = op
        self.args = args

    @property
    def base_operator(self):
        if type(self.op) == SomeMetaOp:
            return self.op
        else:
            raise NotImplementedError()

    @property
    def base_arguments(self):
        if len(self.args) > 0:
            return self.args
        else:
            raise NotImplementedError()


def test_unify():

    q_lv = var()

    op = SomeMetaOp()
    a_args = (1, 2)
    a = SomeMetaVariable(op, a_args, obj=SomeType(1, 2))
    b = SomeMetaVariable(op, q_lv)

    s = unify(a, b)
    assert s is not False
    assert s[q_lv] is a_args

    obj = reify(b, s)

    assert obj == a

    r_lv = var()
    b = SomeMetaVariable(op, q_lv, obj=r_lv)

    s = unify(a, b)
    assert s is not False
    assert s[r_lv] is a.obj

    assert car(a) == rator(a) == op
    assert isinstance(cdr(a), ExpressionTuple)
    assert isinstance(rands(a), ExpressionTuple)
    assert cdr(a) == rands(a) == a_args

    a = SomeMetaVariable(op, ())

    with pytest.raises(ConsError):
        cdr(a)

    with pytest.raises(ConsError):
        rands(a)

    op = SomeOtherMetaOp()
    a = SomeMetaVariable(op, ())

    with pytest.raises(ConsError):
        car(a)

    with pytest.raises(ConsError):
        rator(a)
