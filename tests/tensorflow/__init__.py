from functools import wraps
from tensorflow.python.eager.context import graph_mode


def run_in_graph_mode(f):
    @wraps(f)
    def _f(*args, **kwargs):
        with graph_mode():
            return f()

    return _f
