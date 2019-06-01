from kanren.facts import Relation
from kanren.goals import goalify


# Hierarchical models that we recognize.
hierarchical_model = Relation("hierarchical")

# Conjugate relationships
conjugate = Relation("conjugate")


concat = goalify(lambda *args: "".join(args))
