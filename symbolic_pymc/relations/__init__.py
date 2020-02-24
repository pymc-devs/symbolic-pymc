from kanren.facts import Relation

from unification import unify, reify, Var


# Hierarchical models that we recognize.
hierarchical_model = Relation("hierarchical")

# Conjugate relationships
conjugate = Relation("conjugate")


def concat(a, b, out):
    """Construct a non-relational string concatenation goal."""

    def concat_goal(S):
        nonlocal a, b, out

        a_rf, b_rf, out_rf = reify((a, b, out), S)

        if isinstance(a_rf, str) and isinstance(b_rf, str):
            S_new = unify(out_rf, a_rf + b_rf, S)

            if S_new is not False:
                yield S_new
                return
        elif isinstance(a_rf, (Var, str)) and isinstance(b_rf, (Var, str)):
            yield S

    return concat_goal
