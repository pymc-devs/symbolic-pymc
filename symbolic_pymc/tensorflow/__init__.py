from tensorflow.python.framework import ops

# Needed to register generic functions
from .unify import *

ops.disable_eager_execution()
