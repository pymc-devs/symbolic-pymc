import pytest

from operator import add

from kanren.term import term, operator, arguments

from symbolic_pymc.etuple import (ExpressionTuple, etuple)


def test_etuple():
    """Test basic `etuple` functionality."""
    def test_op(*args):
        return tuple(object() for i in range(sum(args)))

    e1 = etuple(test_op, 1, 2)

    assert e1._eval_obj is ExpressionTuple.null

    with pytest.raises(ValueError):
        e1.eval_obj = 1

    e1_obj = e1.eval_obj
    assert len(e1_obj) == 3
    assert all(type(o) == object for o in e1_obj)

    # Make sure we don't re-create the cached `eval_obj`
    e1_obj_2 = e1.eval_obj
    assert e1_obj == e1_obj_2

    # Confirm that evaluation is recursive
    e2 = etuple(add, (object(),), e1)

    # Make sure we didn't convert this single tuple value to
    # an `etuple`
    assert type(e2[1]) == tuple

    # Slices should be `etuple`s, though.
    assert isinstance(e2[:1], ExpressionTuple)
    assert e2[1] == e2[1:2][0]

    e2_obj = e2.eval_obj

    assert type(e2_obj) == tuple
    assert len(e2_obj) == 4
    assert all(type(o) == object for o in e2_obj)
    # Make sure that it used `e1`'s original `eval_obj`
    assert e2_obj[1:] == e1_obj

    # Confirm that any combination of `tuple`s/`etuple`s in
    # concatenation result in an `etuple`
    e_radd = (1,) + etuple(2, 3)
    assert isinstance(e_radd, ExpressionTuple)
    assert e_radd == (1, 2, 3)

    e_ladd = etuple(1, 2) + (3,)
    assert isinstance(e_ladd, ExpressionTuple)
    assert e_ladd == (1, 2, 3)


def test_etuple_term():
    """Test `etuplize` and `etuple` interaction with `term`
    """
    # Make sure that we don't lose underlying `eval_obj`s
    # when taking apart and re-creating expression tuples
    # using `kanren`'s `operator`, `arguments` and `term`
    # functions.
    e1 = etuple(add, (object(),), (object(),))
    e1_obj = e1.eval_obj

    e1_dup = (operator(e1),) + arguments(e1)

    assert isinstance(e1_dup, ExpressionTuple)
    assert e1_dup.eval_obj == e1_obj

    e1_dup_2 = term(operator(e1), arguments(e1))
    assert e1_dup_2 == e1_obj


def test_etuple_kwargs():
    """Test keyword arguments and default argument values."""
    def test_func(a, b, c=None, d='d-arg', **kwargs):
        assert isinstance(c, (type(None), int))
        return [a, b, c, d]

    e1 = etuple(test_func, 1, 2)
    assert e1.eval_obj == [1, 2, None, 'd-arg']

    # Make sure we handle variadic args properly
    def test_func2(*args, c=None, d='d-arg', **kwargs):
        assert isinstance(c, (type(None), int))
        return list(args) + [c, d]

    e11 = etuple(test_func2, 1, 2)
    assert e11.eval_obj == [1, 2, None, 'd-arg']

    e2 = etuple(test_func, 1, 2, 3)
    assert e2.eval_obj == [1, 2, 3, 'd-arg']

    e3 = etuple(test_func, 1, 2, 3, 4)
    assert e3.eval_obj == [1, 2, 3, 4]

    e4 = etuple(test_func, 1, 2, c=3)
    assert e4.eval_obj == [1, 2, 3, 'd-arg']

    e5 = etuple(test_func, 1, 2, d=3)
    assert e5.eval_obj == [1, 2, None, 3]

    e6 = etuple(test_func, 1, 2, 3, d=4)
    assert e6.eval_obj == [1, 2, 3, 4]

    # Try evaluating nested etuples
    e7 = etuple(test_func, etuple(add, 1, 0), 2,
                c=etuple(add, 1, etuple(add, 1, 1)))
    assert e7.eval_obj == [1, 2, 3, 'd-arg']

    # Try a function without an obtainable signature object
    e8 = etuple(enumerate, etuple(list, ['a', 'b', 'c', 'd']),
                start=etuple(add, 1, etuple(add, 1, 1)))
    assert list(e8.eval_obj) == [(3, 'a'), (4, 'b'), (5, 'c'), (6, 'd')]
