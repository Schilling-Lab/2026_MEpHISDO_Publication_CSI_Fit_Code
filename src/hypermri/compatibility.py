"""
Backward compatibility utilities for HyperMRI.

This module provides utilities to help migrate from the old variable naming
conventions to the new standardized ones while maintaining backward compatibility.
"""

from collections.abc import MutableMapping
from typing import Any


# Common legacy mappings
COMMON_DATA_ARRAY_MAPPINGS = {
    "seq2d": "data_array",
    "reco_array": "data_array",  # Note: this is a simplification
    "reco_data": "data_array",
    "kspace_xarray": "kspace_array",
}

COMMON_PATH_MAPPINGS = {
    # Most path attributes should just use 'path'
}

LEGACY_FREQUENCY_KEY_MAP = {
    "freq_axis_hz": ("freq_range_hz", "freq_range_Hz", "freq_range_HZ"),
    "freq_axis_ppm": ("freq_range_ppm", "freq_range_PPM"),
}


def _normalize_frequency_unit(unit: str) -> str:
    unit_norm = str(unit).strip().lower()
    if unit_norm not in {"hz", "ppm"}:
        raise ValueError(
            f"Unsupported frequency unit '{unit}'. Expected one of ['Hz', 'ppm']."
        )
    return unit_norm


def _resolve_frequency_keys(unit: str) -> tuple[str, tuple[str, ...]]:
    normalized_unit = _normalize_frequency_unit(unit)
    new_key = f"freq_axis_{normalized_unit}"
    legacy_keys = LEGACY_FREQUENCY_KEY_MAP.get(new_key, ())
    return new_key, legacy_keys


def assign_frequency_axis(
    attrs: MutableMapping | None,
    freq_axis: Any,
    unit: str,
    *,
    mirror_legacy_keys: bool = True,
) -> None:
    """
    Assign a frequency axis to attrs, optionally mirroring legacy key names.

    Parameters
    ----------
    attrs : MutableMapping or None
        Attribute mapping to update.
    freq_axis : Any
        Frequency axis to assign.
    unit : {'Hz', 'ppm'}
        Frequency unit associated with the axis.
    mirror_legacy_keys : bool, optional
        If True, also store the axis under legacy key names.
    """
    if attrs is None or not isinstance(attrs, MutableMapping) or freq_axis is None:
        return

    new_key, legacy_keys = _resolve_frequency_keys(unit)
    attrs[new_key] = freq_axis

    if mirror_legacy_keys:
        for legacy_key in legacy_keys:
            attrs[legacy_key] = freq_axis


                # Optionally remove the old attribute or leave it for compatibility
                # delattr(obj, old_attr)


