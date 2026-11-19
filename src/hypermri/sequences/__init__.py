"""
Sequences module for hypermri.

Sequence classes are looked up by name from this module during smart loading
(see config/SmartLoadMeasurements.json).
"""

from .base_anatomy import BaseAnatomy
from .base_spectroscopy import BaseSpectroscopy
from .CSI import CSI
from .RARE import RARE

__all__ = [
    "BaseAnatomy",
    "BaseSpectroscopy",
    "CSI",
    "RARE",
]
