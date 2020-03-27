# We need this so that `multipledispatch` initialization occurs
from .dispatch import *

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
