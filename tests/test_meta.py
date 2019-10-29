import pytest

import numpy as np

from symbolic_pymc.utils import HashableNDArray
from symbolic_pymc.meta import MetaSymbol, MetaOp, metatize


class SomeOp(object):
    def __repr__(self):
        return '<SomeOp>'


class SomeType(object):
    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2

    def __repr__(self):
        return f'SomeType({self.field1}, {self.field2})'

    def __str__(self):
        return f'SomeType<{self.field1}, {self.field2}>'


class SomeMetaSymbol(MetaSymbol):
    __slots__ = ('field1', 'field2', '_blah')
    base = SomeType

    def __init__(self, obj=None):
        super().__init__(obj)
        self.field1 = 1
        self.field2 = 2
        self._blah = 'a'


class SomeMetaOp(MetaOp):
    __slots__ = ()
    base = SomeOp

    def out_meta_types(self):
        return [SomeMetaSymbol]

    def __call__(self, *args, **kwargs):
        return SomeMetaSymbol(*args, **kwargs)


def test_meta():
    """Make sure hash caching and slot manipulation works."""

    some_mt = SomeMetaSymbol()

    assert some_mt.__all_slots__ == ('_obj', '_hash', '_rands', 'field1', 'field2', '_blah')
    assert some_mt.__all_props__ == ('field1', 'field2')
    assert some_mt.__props__ == ('field1', 'field2')
    assert some_mt.__volatile_slots__ == ('_obj', '_hash', '_rands', '_blah')

    assert some_mt.obj is None
    assert not hasattr(some_mt, '_hash')

    some_hash = hash(some_mt)

    assert some_mt._hash == some_hash

    assert some_mt.field1 == 1
    assert some_mt.field2 == 2

    # This assignment shouldn't change the cached values
    some_mt._blah = 'b'

    assert some_mt._hash == some_hash

    # This should
    some_mt.field1 = 10

    assert some_mt._hash is None
    assert some_mt._blah is None

    some_new_hash = hash(some_mt)

    assert some_mt._hash == some_new_hash
    assert some_new_hash != some_hash

    some_op_mt = SomeMetaOp(SomeOp())

    with pytest.raises(AttributeError):
        some_op_mt.obj = SomeOp()


def test_meta_inheritance():
    class SomeOtherType(SomeType):
        def __init__(self, field1, field2, field3):
            super().__init__(field1, field2)
            self.field3 = field3

    class SomeOtherMetaSymbol(SomeMetaSymbol):
        __slots__ = ('field3', '_bloh')
        base = SomeOtherType

        def __init__(self, obj=None):
            super().__init__(obj)
            self.field3 = 3

        def __hash__(self):
            return hash((super().__hash__(), self.field3))

    some_mt = SomeMetaSymbol()
    other_mt = SomeOtherMetaSymbol()

    assert some_mt != other_mt

    assert other_mt.__all_slots__ == ('_obj', '_hash', '_rands', 'field1', 'field2', '_blah', 'field3', '_bloh')
    assert other_mt.__all_props__ == ('field1', 'field2', 'field3')
    assert other_mt.__props__ == ('field3',)
    assert other_mt.__volatile_slots__ == ('_obj', '_hash', '_rands', '_blah', '_bloh')


def test_meta_str():

    some_mt = SomeMetaSymbol()

    assert repr(some_mt) == 'SomeMetaSymbol(1, 2)'
    assert str(some_mt) == repr(some_mt)

    some_mt = SomeMetaSymbol(SomeType(1, 2))

    assert repr(some_mt) == 'SomeMetaSymbol(1, 2, obj=SomeType(1, 2))'
    assert str(some_mt) == 'SomeMetaSymbol(1, 2)'

    some_op_mt = SomeMetaOp()
    assert repr(some_op_mt) == 'SomeMetaOp(obj=None)'

    some_op_mt = SomeMetaOp(SomeOp())
    assert repr(some_op_mt) == 'SomeMetaOp(obj=<SomeOp>)'


def test_meta_pretty():
    pretty_mod = pytest.importorskip("IPython.lib.pretty")
    from symbolic_pymc.meta import meta_repr

    some_mt = SomeMetaSymbol()

    assert pretty_mod.pretty(some_mt) == 'SomeMetaSymbol(field1=1, field2=2)'

    meta_repr.print_obj = True

    assert pretty_mod.pretty(some_mt) == 'SomeMetaSymbol(field1=1, field2=2)'

    some_mt = SomeMetaSymbol(SomeType(1, 2))

    assert pretty_mod.pretty(some_mt) == 'SomeMetaSymbol(field1=1, field2=2, obj=SomeType(1, 2))'

    meta_repr.print_obj = False

    some_mt = SomeMetaSymbol(SomeType(1, 2))
    some_mt.field1 = SomeMetaSymbol(SomeType(3, 4))
    some_mt.field1.field2 = SomeMetaSymbol(SomeType(5, 6))

    assert pretty_mod.pretty(some_mt) == 'SomeMetaSymbol(\n  field1=SomeMetaSymbol(field1=1, field2=SomeMetaSymbol(field1=1, field2=2)),\n  field2=2)'


def test_metatize():
    x_mt = metatize(np.r_[1, 2, 3])
    assert isinstance(x_mt, HashableNDArray)

    y_mt = metatize(np.r_[1, 2, 3, 4])
    assert isinstance(y_mt, HashableNDArray)

    assert x_mt != y_mt

    assert x_mt != 1
