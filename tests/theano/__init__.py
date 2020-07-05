import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import theano

import pymc3 as pm

from functools import wraps


theano.config.compute_test_value = "ignore"
theano.config.on_opt_error = "raise"
theano.config.mode = "FAST_COMPILE"
theano.config.cxx = ""


def requires_test_values(f):
    @wraps(f)
    def _f(*args, **kwargs):

        import theano

        last_value = theano.config.compute_test_value
        theano.config.compute_test_value = "raise"

        try:
            res = f(*args, **kwargs)
        finally:
            theano.config.compute_test_value = last_value

        return res

    return _f
