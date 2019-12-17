import weakref

from abc import ABC, abstractmethod
from types import MappingProxyType
from collections import OrderedDict

from unification import unify, reify, Var
from unification.core import _unify, _reify


class KanrenConstraintStore(ABC):
    """A class that enforces constraints between logic variables in a miniKanren state."""

    __slots__ = ("mappings",)

    def __init__(self, mappings=None):
        # Mappings between logic variables and their constraint values
        # (e.g. the values against which they cannot be unified).
        self.mappings = mappings if mappings is not None else dict()
        # TODO: We can't use this until `Var` is a factory returning unique
        # objects/references for a given `Var.token` value.
        # self.mappings = weakref.WeakKeyDictionary(mappings)

    @abstractmethod
    def pre_check(self, state, key=None, value=None):
        """Check a key-value pair before they're added to a KanrenState."""
        raise NotImplementedError()

    @abstractmethod
    def post_check(self, new_state, key=None, value=None, old_state=None):
        """Check a key-value pair after they're added to a KanrenState."""
        raise NotImplementedError()

    @abstractmethod
    def update(self, *args, **kwargs):
        """Add a new constraint."""
        raise NotImplementedError()

    @abstractmethod
    def constraints_str(self, var):
        """Print the constraints on a logic variable."""
        raise NotImplementedError()


class KanrenState(dict):
    """A miniKanren state that holds unifications of logic variables and upholds constraints on logic variables."""

    __slots__ = ("constraints", "__weakref__")

    def __init__(self, *s, constraints=None):
        super().__init__(*s)
        self.constraints = OrderedDict(constraints or [])

    def pre_checks(self, key, value):
        return all(cstore.pre_check(self, key, value) for cstore in self.constraints.values())

    def post_checks(self, new_state, key, value):
        return all(
            cstore.post_check(new_state, key, value, old_state=self)
            for cstore in self.constraints.values()
        )

    def add_constraint(self, constraint):
        assert isinstance(constraint, KanrenConstraintStore)
        self.constraints[type(constraint)] = constraint

    def __eq__(self, other):
        if isinstance(other, KanrenState):
            return super().__eq__(other)

        # When comparing with a plain dict, disregard the constraints.
        if isinstance(other, dict):
            return super().__eq__(other)
        return False

    def __repr__(self):
        return f"KanrenState({super().__repr__()}, {self.constraints})"


class Disequality(KanrenConstraintStore):
    """A disequality constraint (i.e. two things do not unify)."""

    def post_check(self, new_state, key=None, value=None, old_state=None):
        # This implementation follows-up every addition to a `KanrenState` with
        # a consistency check against all the disequality constraints.  It's
        # not particularly scalable, but it works for now.
        return not any(
            any(new_state == unify(lvar, val, new_state) for val in vals)
            for lvar, vals in self.mappings.items()
        )

    def pre_check(self, state, key=None, value=None):
        return True

    def update(self, key, value):
        # In this case, logic variables are mapped to a set of values against
        # which they cannot unify.
        if key not in self.mappings:
            self.mappings[key] = {value}
        else:
            self.mappings[key].add(value)

    def constraints_str(self, var):
        if var in self.mappings:
            return f"=/= {self.mappings[var]}"
        else:
            return ""

    def __repr__(self):
        return ",".join([f"{k} =/= {v}" for k, v in self.mappings.items()])


def unify_KanrenState(u, v, S):
    if S.pre_checks(u, v):
        s = unify(u, v, MappingProxyType(S))
        if s is not False and S.post_checks(s, u, v):
            return KanrenState(s, constraints=S.constraints)

    return False


unify.add((object, object, KanrenState), unify_KanrenState)
unify.add(
    (object, object, MappingProxyType),
    lambda u, v, d: unify.dispatch(type(u), type(v), dict)(u, v, d),
)
_unify.add(
    (object, object, MappingProxyType),
    lambda u, v, d: _unify.dispatch(type(u), type(v), dict)(u, v, d),
)


class ConstrainedVar(Var):
    """A logic variable that tracks its own constraints.

    Currently, this is only for display/reification purposes.

    """

    __slots__ = ("_id", "token", "S", "var")

    def __new__(cls, var, S):
        obj = super().__new__(cls, var.token)
        obj.S = weakref.ref(S)
        obj.var = weakref.ref(var)
        return obj

    def __repr__(self):
        var = self.var()
        S = self.S()
        if var is not None and S is not None:
            u_constraints = ",".join([c.constraints_str(var) for c in S.constraints.values()])
            return f"{var}: {{{u_constraints}}}"


def reify_KanrenState(u, S):
    u_res = reify(u, MappingProxyType(S))
    if isinstance(u_res, Var):
        return ConstrainedVar(u_res, S)
    else:
        return u_res


_reify.add((tuple(p[0] for p in _reify.ordering if p[1] == dict), KanrenState), reify_KanrenState)
_reify.add((object, MappingProxyType), lambda u, s: _reify.dispatch(type(u), dict)(u, s))


def neq(u, v):
    """Construct a disequality goal."""

    def neq_goal(S):
        if not isinstance(S, KanrenState):
            S = KanrenState(S)

        diseq_constraint = S.constraints.setdefault(Disequality, Disequality())

        diseq_constraint.update(u, v)

        if diseq_constraint.post_check(S):
            return iter([S])
        else:
            return iter([])

    return neq_goal
