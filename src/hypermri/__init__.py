from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hypermri")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from .brukerexp import BrukerHandler
from .core import MRIExperiment
from .study import Study

__all__ = [
    "__version__",
    "MRIExperiment",
    "BrukerHandler",
    "Study",
]
