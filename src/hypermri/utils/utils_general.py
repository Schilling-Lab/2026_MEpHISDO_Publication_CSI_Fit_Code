"""
Shim for backward compatibility of general utilities.

Deprecated: This module has been split into focused modules in `hypermri.utils`.
"""

import warnings

import numpy as np

from .coordinates.grids import (
    get_xyz_grid,
)
from .time.temporal import (
    calc_sampling_time_axis,
)


# Constants
DATA_PRECISION = np.float16
DB_MSG = True

warnings.warn(
    "Importing from hypermri.utils.utils_general is deprecated. "
    "Please update your code to import from the specific modules in hypermri.utils.",
    DeprecationWarning,
    stacklevel=2,
)

# Legacy aliases
calc_sampling_time_axis_old = calc_sampling_time_axis
get_xyz_grid_new = get_xyz_grid
