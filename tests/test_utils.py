import numpy as np

from unification import var

from symbolic_pymc.meta import MetaSymbol, MetaOp
from symbolic_pymc.utils import meta_diff, eq_lvar, HashableNDArray


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


class SomeMetaSymbol(MetaSymbol):
    __slots__ = ("field1", "field2", "_blah")
    base = SomeType

    def __init__(self, obj=None):
        super().__init__(obj)
        self.field1 = 1
        self.field2 = 2
        self._blah = "a"


class SomeMetaOp(MetaOp):
    __slots__ = ()
    base = SomeOp

    def output_meta_types(self):
        return [SomeMetaSymbol]

    def __call__(self, *args, **kwargs):
        return SomeMetaSymbol(*args, **kwargs)


class SomeOtherMetaSymbol(MetaSymbol):
    __slots__ = ("field1", "field2")
    base = SomeType

    def __init__(self, field1, field2, obj=None):
        super().__init__(obj)
        self.field1 = field1
        self.field2 = field2


class SomeOtherOp(object):
    def __repr__(self):
        return "<SomeOp>"


class SomeOtherMetaOp(SomeMetaOp):
    base = SomeOtherOp


def test_parts_unequal():
    s0 = SomeMetaSymbol()
    s1 = SomeOtherMetaSymbol(1, 2)

    res = meta_diff(s0, s1)
    assert res.reason == "types"
    assert res.path(s0) is s0
    assert res.objects == (s0, s1)

    res = meta_diff(s0, s1, cmp_types=False)
    assert res is None

    s2 = SomeOtherMetaSymbol(1, 3)
    res = meta_diff(s0, s2, cmp_types=False)
    assert res.path(s2) == 3
    assert res.path(s1) == 2
    assert res.reason == "ne_fn"
    assert res.objects == (2, 3)

    res = meta_diff(SomeMetaOp(), SomeMetaOp())
    assert res is None

    op1 = SomeMetaOp()
    op2 = SomeOtherMetaOp()
    res = meta_diff(op1, op2, cmp_types=False)
    assert res.path(op1) is op1
    assert res.reason == "bases"
    assert res.objects == (op1.base, op2.base)

    a = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 4)])
    b = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 5)])
    res = meta_diff(a, b)

    assert res.path(a) == 4
    assert res.path(b) == 5
    assert res.reason == "ne_fn"
    assert res.objects == (4, 5)

    a = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 4)])
    b = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 4)])
    res = meta_diff(a, b)
    assert res is None

    a = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 4), 5])
    b = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 4)])
    res = meta_diff(a, b)
    assert res is not None
    assert res.reason == "seq len"

    a = SomeOtherMetaSymbol(1, ["a", "b"])
    b = SomeOtherMetaSymbol(1, 2)
    res = meta_diff(a, b, cmp_types=False)
    assert res is not None
    assert res.reason == "ne_fn"

    a = SomeOtherMetaSymbol(1, ["a", "b"])
    b = SomeOtherMetaSymbol(1, "ab")
    res = meta_diff(a, b, cmp_types=False)
    assert res is not None

    a = SomeOtherMetaSymbol(1, {"a": 1, "b": 2})
    b = SomeOtherMetaSymbol(1, {"b": 2, "a": 1})
    res = meta_diff(a, b)
    assert res is None

    a = SomeOtherMetaSymbol(1, {"a": 1, "b": 2})
    b = SomeOtherMetaSymbol(1, {"b": 3, "a": 1})
    res = meta_diff(a, b)
    assert res.reason == "ne_fn"
    assert res.objects == (2, 3)
    assert res.path(a) == 2
    assert res.path(b) == 3

    a = SomeOtherMetaSymbol(1, {"a": 1, "b": 2})
    b = SomeOtherMetaSymbol(1, {"a": 1, "c": 2})
    res = meta_diff(a, b)
    assert res.reason == "map keys"
    assert res.path(a) == {"a": 1, "b": 2}
    assert res.objects == ([("a", 1), ("b", 2)], [("a", 1), ("c", 2)])


def test_eq_lvar():
    a = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 4)])
    b = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 4)])
    assert eq_lvar(a, b) is True

    a = SomeOtherMetaSymbol(1, [2, SomeOtherMetaSymbol(3, 4)])
    b = SomeOtherMetaSymbol(1, [2, var()])
    assert eq_lvar(a, b) is False

    a = SomeOtherMetaSymbol(1, [2, var()])
    b = SomeOtherMetaSymbol(1, [2, var()])
    assert eq_lvar(a, b) is True

    a = SomeOtherMetaSymbol(1, [2, {"a": var()}])
    b = SomeOtherMetaSymbol(1, [2, {"a": var()}])
    assert eq_lvar(a, b) is True

    a = SomeOtherMetaSymbol(1, [3, var()])
    b = SomeOtherMetaSymbol(1, [2, var()])
    assert eq_lvar(a, b) is False


def test_HashableNDArray():
    a = np.r_[[1, 2], 3]
    a_h = a.view(HashableNDArray)
    b = np.r_[[1, 2], 3]
    b_h = b.view(HashableNDArray)

    assert hash(a_h) == hash(b_h)
    assert a_h == b_h
    assert not a_h != b_h

    c = np.r_[[1, 2], 4]
    c_h = c.view(HashableNDArray)
    assert hash(a_h) != hash(c_h)
    assert a_h != c_h
