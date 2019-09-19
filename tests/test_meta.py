from symbolic_pymc.meta import MetaSymbol


class SomeType(object):
    def __init__(self, field1, field2):
        self.field1 = field1
        self.field2 = field2


class SomeMetaSymbol(MetaSymbol):
    __slots__ = ('field1', 'field2', '_blah')
    base = SomeType

    def __init__(self, obj=None):
        super().__init__(obj)
        self.field1 = 1
        self.field2 = 2
        self._blah = 'a'


def test_meta():
    """Make sure hash caching and slot manipulation works."""

    some_mt = SomeMetaSymbol()

    assert some_mt.__all_slots__ == ('_obj', '_hash', 'field1', 'field2', '_blah')
    assert some_mt.__all_props__ == ('field1', 'field2')
    assert some_mt.__props__ == ('field1', 'field2')
    assert some_mt.__volatile_slots__ == ('_obj', '_hash', '_blah')

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

    other_mt = SomeOtherMetaSymbol()

    assert other_mt.__all_slots__ == ('_obj', '_hash', 'field1', 'field2', '_blah', 'field3', '_bloh')
    assert other_mt.__all_props__ == ('field1', 'field2', 'field3')
    assert other_mt.__props__ == ('field3',)
    assert other_mt.__volatile_slots__ == ('_obj', '_hash', '_blah', '_bloh')
