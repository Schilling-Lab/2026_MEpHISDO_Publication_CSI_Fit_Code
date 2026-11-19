"""
Standardized data types and containers for the hypermri package.

This module defines data structures used throughout the hypermri ecosystem
to ensure consistent data handling and type safety.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional



class VendorType(Enum):
    """Supported MRI vendors."""

    BRUKER = "bruker"
    SIEMENS = "siemens"
    PHILIPS = "philips"
    GE = "ge"


class SequenceCategory(Enum):
    """Categories of MRI sequences."""

    ANATOMICAL = "anatomical"
    SPECTROSCOPY = "spectroscopy"
    DIFFUSION = "diffusion"
    PERFUSION = "perfusion"
    FUNCTIONAL = "functional"


@dataclass
class ExperimentMetadata:
    """Metadata for MRI experiments."""

    id: str
    sequence_name: str
    vendor: VendorType
    category: SequenceCategory
    timestamp: Optional[str] = None
    subject_id: Optional[str] = None
    study_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


