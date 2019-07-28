from unification import var

from kanren import eq  # , run, lall
# from kanren.core import goaleval

from symbolic_pymc.relations import lconj, lconj_seq, ldisj, ldisj_seq, conde


def test_lconj_basics():

    res = list(lconj(eq(1, var('a')), eq(2, var('b')))({}))
    assert res == [{var('a'): 1, var('b'): 2}]

    res = list(lconj(eq(1, var('a')))({}))
    assert res == [{var('a'): 1}]

    res = list(lconj_seq([])({}))
    assert res == []

    res = list(lconj(eq(1, var('a')), eq(2, var('a')))({}))
    assert res == []

    res = list(lconj(eq(1, 2))({}))
    assert res == []

    res = list(lconj(eq(1, 1))({}))
    assert res == [{}]


def test_ldisj_basics():

    res = list(ldisj(eq(1, var('a')))({}))
    assert res == [{var('a'): 1}]

    res = list(ldisj(eq(1, 2))({}))
    assert res == []

    res = list(ldisj(eq(1, 1))({}))
    assert res == [{}]

    res = list(ldisj(eq(1, var('a')), eq(1, var('a')))({}))
    assert res == [{var('a'): 1}, {var('a'): 1}]

    res = list(ldisj(eq(1, var('a')), eq(2, var('a')))({}))
    assert res == [{var('a'): 1}, {var('a'): 2}]

    res = list(ldisj_seq([])({}))
    assert res == []


def test_conde_basics():

    res = list(conde([eq(1, var('a')), eq(2, var('b'))],
                     [eq(1, var('b')), eq(2, var('a'))])({}))
    assert res == [{var('a'): 1, var('b'): 2},
                   {var('b'): 1, var('a'): 2}]

    res = list(conde([eq(1, var('a')), eq(2, 1)],
                     [eq(1, var('b')), eq(2, var('a'))])({}))
    assert res == [{var('b'): 1, var('a'): 2}]

    res = list(conde([eq(1, var('a')),
                      conde([eq(11, var('aa'))],
                            [eq(12, var('ab'))])],
                     [eq(1, var('b')),
                      conde([eq(111, var('ba')),
                             eq(112, var('bb'))],
                            [eq(121, var('bc'))])])({}))
    assert res == [{var('a'): 1, var('aa'): 11},
                   {var('b'): 1, var('ba'): 111, var('bb'): 112},
                   {var('a'): 1, var('ab'): 12},
                   {var('b'): 1, var('bc'): 121}]

    res = list(conde([eq(1, 2)], [eq(1, 1)])({}))
    assert res == [{}]

    assert list(lconj(eq(1, 1))({})) == [{}]

    res = list(lconj(conde([eq(1, 2)], [eq(1, 1)]))({}))
    assert res == [{}]

    res = list(lconj(conde([eq(1, 2)], [eq(1, 1)]),
                     conde([eq(1, 2)], [eq(1, 1)]))({}))
    assert res == [{}]

# def test_short_circuit_lconj():
#
#     def one_bad_goal(goal_nums, good_goals=10, _eq=eq):
#         for i in goal_nums:
#             if i == good_goals:
#                 def _g(S, i=i):
#                     print('{} bad'.format(i))
#                     yield from _eq(1, 2)(S)
#
#             else:
#                 def _g(S, i=i):
#                     print('{} good'.format(i))
#                     yield from _eq(1, 1)(S)
#
#             yield _g
#
#     goal_nums = iter(range(20))
#     run(0, var('q'), lall(*one_bad_goal(goal_nums)))
#
#     # `kanren`'s `lall` will necessarily exhaust the generator.
#     next(goal_nums, None)
#
#     goal_nums = iter(range(20))
#     run(0, var('q'), lconj(one_bad_goal(goal_nums)))
#     next(goal_nums, None)
