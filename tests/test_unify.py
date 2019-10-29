import numpy as np

# Needed to set up unify dispatch functions
import symbolic_pymc.unify

from unification import unify, var, reify

from cons import cons

from symbolic_pymc.etuple import etuple, ExpressionTuple


def test_etuple():
    et = etuple(var('a'),)
    res = reify(et, {var('a'): 1})
    assert isinstance(res, ExpressionTuple)

    et = etuple(var('a'),)
    res = unify(et, (1,))
    assert res == {var('a'): 1}

    from operator import add

    et = etuple(add, 1, 2)
    assert et.eval_obj == 3

    res = unify(et, cons(var('a'), var('b')))
    assert res == {var('a'): add,
                   var('b'): et[1:]}

    assert ((res[var('a')],) + res[var('b')])._eval_obj == 3
