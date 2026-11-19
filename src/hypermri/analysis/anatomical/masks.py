"""Mask and ROI utilities for anatomical analysis.

These helpers originate from the legacy ``utils_general`` module and are now part
of the structured anatomical analysis package.
"""

from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np

__all__ = [
    "calc_coverage",
    "apply_mask",
]


def _ensure_mask_is_6d(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        raise ValueError("mask must be provided")

    mask_array = np.asarray(mask)
    if mask_array.ndim < 1:
        raise ValueError("mask must have at least one dimension")

    if mask_array.ndim < 6:
        mask_array = np.expand_dims(mask_array, axis=0)
        while mask_array.ndim < 6:
            mask_array = np.expand_dims(mask_array, axis=-1)

    return mask_array


def calc_coverage(
    input_data: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    return_nans: bool = False,
    masks_drawn_on: str = "axial",
):
    """Compute mask coverage for a low-resolution volume against a high-res mask."""
    if input_data is None or mask is None:
        raise ValueError("Both input_data and mask must be provided")

    data_array = np.asarray(input_data)
    mask_array = _ensure_mask_is_6d(mask)

    if data_array.ndim < 6:
        raise ValueError("input_data must be at least 6D")

    if data_array.ndim == 7:
        mask_array = np.expand_dims(mask_array, axis=-1)
        mask_array = np.repeat(mask_array, data_array.shape[6], axis=-1)

    for dim_index in (0, 4, 5):
        if (
            data_array.shape[dim_index] > mask_array.shape[dim_index]
            and mask_array.shape[dim_index] == 1
        ):
            mask_array = np.repeat(
                mask_array, data_array.shape[dim_index], axis=dim_index
            )

    low_res_shape = data_array.shape[1:4]
    high_res_shape = mask_array.shape[1:4]

    output_shape = tuple(
        data_dim if mask_dim != 1 else 1
        for data_dim, mask_dim in zip(data_array.shape, mask_array.shape, strict=False)
    )
    coverage = np.zeros(output_shape, dtype=float)

    depth_scale = high_res_shape[0] / (low_res_shape[0] if low_res_shape[0] != 1 else 1)
    row_scale = high_res_shape[1] / (low_res_shape[1] if low_res_shape[1] != 1 else 1)
    col_scale = high_res_shape[2] / (low_res_shape[2] if low_res_shape[2] != 1 else 1)

    for i in range(output_shape[1]):
        for j in range(output_shape[2]):
            for k in range(output_shape[3]):
                start_depth = int(i * depth_scale) if high_res_shape[0] > 1 else 0
                end_depth = (
                    min(int((i + 1) * depth_scale), mask_array.shape[1])
                    if high_res_shape[0] > 1
                    else 1
                )

                start_row = int(j * row_scale) if high_res_shape[1] > 1 else 0
                end_row = (
                    min(int((j + 1) * row_scale), mask_array.shape[2])
                    if high_res_shape[1] > 1
                    else 1
                )

                start_col = int(k * col_scale) if high_res_shape[2] > 1 else 0
                end_col = (
                    min(int((k + 1) * col_scale), mask_array.shape[3])
                    if high_res_shape[2] > 1
                    else 1
                )

                nelems = max(
                    (end_depth - start_depth)
                    * (end_row - start_row)
                    * (end_col - start_col),
                    1,
                )

                coverage[:, i, j, k, :, :] = (
                    np.nansum(
                        mask_array[
                            :,
                            start_depth:end_depth,
                            start_row:end_row,
                            start_col:end_col,
                            :,
                            :,
                        ],
                        axis=(1, 2, 3),
                    )
                    / nelems
                )

    if return_nans:
        coverage[coverage == 0] = np.nan

    return coverage


def apply_mask(
    input_data: np.ndarray | None = None,
    mask_dict: Mapping[str, np.ndarray] | None = None,
    mask_key: str | None = None,
    mask: np.ndarray | None = None,
    return_nans: bool = False,
    avg_all_space_dims: bool = False,
    masks_drawn_on: str = "axial",
    mask_slice_ind: int | None = None,
    bitwise: bool = False,
    **kwargs,
):
    """Apply a mask to spectroscopy data and optionally average spatial dims."""
    if input_data is None:
        return False

    selected_mask = mask
    if selected_mask is None and mask_dict is not None and mask_key is not None:
        selected_mask = mask_dict.get(mask_key)
        if selected_mask is None:
            warnings.warn(
                "Cannot extract mask from mask_dict", RuntimeWarning, stacklevel=2
            )
            return False

    if selected_mask is None:
        warnings.warn("No mask provided", RuntimeWarning, stacklevel=2)
        return False

    mask_copy = _ensure_mask_is_6d(selected_mask)

    if mask_slice_ind is None:
        warnings.warn("No mask_slice_ind was given!", RuntimeWarning, stacklevel=2)
    else:
        if masks_drawn_on == "axial":
            mask_copy = mask_copy[:, [mask_slice_ind], :, :, :, :]
        elif masks_drawn_on == "coronal":
            mask_copy = mask_copy[:, :, :, [mask_slice_ind], :, :]
        else:
            mask_copy = mask_copy[:, :, [mask_slice_ind], :, :, :]

    data_array = np.asarray(input_data)
    if data_array.ndim < 6:
        from ...utils.image.dimensions import get_indices_for_strings
        from ..spectroscopy.core import make_NDspec_6Dspec

        if "provided_dims" in kwargs:
            provided_dims = get_indices_for_strings(kwargs["provided_dims"])
            data_array = make_NDspec_6Dspec(
                input_data=data_array, provided_dims=provided_dims
            )
        else:
            data_array = make_NDspec_6Dspec(input_data=data_array)

    coverage = calc_coverage(
        input_data=data_array, mask=mask_copy, return_nans=return_nans
    )

    if bitwise:
        threshold = kwargs.get("bitwise_lower_threshold", 0.5)
        coverage[coverage >= threshold] = 1
        if return_nans:
            coverage[coverage < threshold] = np.nan
        else:
            coverage[coverage < threshold] = 0.0

    masked_data = data_array * coverage

    if avg_all_space_dims:
        masked_data = np.squeeze(np.nanmean(masked_data, axis=(1, 2, 3), keepdims=True))

    return masked_data
