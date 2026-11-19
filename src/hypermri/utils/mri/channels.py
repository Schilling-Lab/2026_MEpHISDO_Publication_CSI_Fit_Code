"""Channel combination utilities for multi-channel MRI data.

This module provides general-purpose channel combination methods that work
with xarray DataArrays and properly handle dimension management.
"""

import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive channel combination.

    Parameters
    ----------
    adaptive_method : {"global", "iterative", "sliding_window"}
        The phase alignment method to use.
    weigh_reference : bool
        If True, weigh the least-squares fit by the squared magnitude.
    auto_reference : bool
        If True, automatically select the reference channel with the highest mean magnitude.
    reference_channel : int
        Index of the channel to use as phase reference (if auto_reference is False).
    window_size : int
        The size of the sliding window for the "sliding_window" method.
    return_phased_channels : bool
        If True, return the phased channels before combination.
    """

    adaptive_method: Literal["global", "iterative", "sliding_window"] = "global"
    weigh_reference: bool = True
    auto_reference: bool = True
    reference_channel: int = 0
    window_size: int = 5
    return_phased_channels: bool = False


@dataclass
class PhaseAlignedConfig:
    """Configuration for phase-aligned channel combination for spectroscopy.

    Parameters
    ----------
    raw_fid_data : np.ndarray
        Raw FID data for phase calculation. Required.
    adjust_phase : bool
        Whether to apply phase correction.
    combination_method : {"mean", "sum"}
        How to combine the aligned channels.
    phase_diff_deg : float, optional
        Explicit phase difference in degrees. If None, calculated from raw_fid_data.
    """

    raw_fid_data: np.ndarray
    adjust_phase: bool = True
    combination_method: Literal["mean", "sum"] = "mean"
    phase_diff_deg: Optional[float] = None


@dataclass
class WeightedMeanConfig:
    """Configuration for weighted mean channel combination.

    Parameters
    ----------
    weights : np.ndarray | xr.DataArray
        Weights for each point. Must be broadcastable to the data array.
    """

    weights: Union[np.ndarray, xr.DataArray]


def combine_channels(
    data_array: xr.DataArray,
    method: Literal[
        "rss", "sum", "mean", "phase_aligned", "weighted_mean", "adaptive"
    ] = "rss",
    config: Optional[
        Union[AdaptiveConfig, PhaseAlignedConfig, WeightedMeanConfig, dict]
    ] = None,
    domain: Optional[Literal["spatial", "k-space"]] = None,
    **kwargs,
) -> xr.DataArray:
    """Combine multiple receiver channels into a single channel.

    This is the main API for channel combination. It automatically handles
    dimension management, ensuring the output has a channel_idx dimension
    with size 1 (unless return_phased_channels is True in AdaptiveConfig).

    Parameters
    ----------
    data_array : xr.DataArray
        Input data with channel_idx dimension.
    method : {"rss", "sum", "mean", "phase_aligned", "weighted_mean", "adaptive"}
        Combination method:
            - "rss": Root-sum-of-squares (default, recommended for general use)
            - "sum": Simple sum across channels
            - "mean": Average across channels
            - "phase_aligned": Phase-aligned combination for 2-channel spectroscopy
            - "weighted_mean": Weighted average across channels.
            - "adaptive": Phase-aligned combination based on a reference channel using least-squares alignment.
    config : AdaptiveConfig | PhaseAlignedConfig | WeightedMeanConfig | dict, optional
        Configuration object or dictionary containing method-specific parameters.
        Recommended over passing parameters via **kwargs.
    domain : {"spatial", "k-space"}, optional
        The domain in which to perform the combination. If None, uses the
        'domain' attribute of the data_array. If the requested domain
        differs from the current domain, FFT/IFFT is performed.
    **kwargs : dict
        Legacy parameters for backward compatibility. Use 'config' instead.
        Supported keys depend on the method (e.g., 'weights' for 'weighted_mean').

    Returns
    -------
    xr.DataArray
        Data with channels combined.

    Raises
    ------
    ValueError
        If channel_idx dimension is missing, or if invalid parameters are provided.
    """
    if "channel_idx" not in data_array.dims:
        raise ValueError("Input data must have 'channel_idx' dimension")

    n_channels = data_array.sizes["channel_idx"]
    if n_channels == 1:
        # Single channel - already has channel_idx with size 1
        return data_array

    # 1. Handle Legacy Parameters
    legacy_keys = {
        "adjust_phase",
        "raw_fid_data",
        "phase_diff_deg",
        "weights",
        "reference_channel",
        "auto_reference",
        "return_phased_channels",
        "weigh_reference",
        "adaptive_method",
        "window_size",
    }

    used_legacy = set(kwargs.keys()) & legacy_keys
    if used_legacy:
        warnings.warn(
            f"Parameters {used_legacy} passed directly to combine_channels are deprecated. "
            "Please use the 'config' parameter with an appropriate configuration object instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # For simplicity in this transition, merge kwargs into config if config is a dict
        if config is None:
            config = kwargs
        elif isinstance(config, dict):
            config = {**kwargs, **config}

    # 2. Dispatch to appropriate method
    if method == "rss":
        return _combine_channels_rss(data_array)
    elif method == "sum":
        return _combine_channels_sum(data_array)
    elif method == "mean":
        return _combine_channels_mean(data_array)
    elif method == "weighted_mean":
        if config is None:
            raise ValueError(
                "config (or legacy 'weights') is required for weighted_mean"
            )

        weights = (
            config.weights if hasattr(config, "weights") else config.get("weights")
        )
        if weights is None:
            raise ValueError(
                "weights are required for weighted_mean channel combination"
            )

        return _combine_channels_weighted_mean(data_array, weights)

    elif method == "phase_aligned":
        if config is None:
            raise ValueError(
                "config (or legacy parameters) is required for phase_aligned"
            )

        if isinstance(config, dict):
            cfg = PhaseAlignedConfig(
                **{
                    k: v
                    for k, v in config.items()
                    if k in PhaseAlignedConfig.__dataclass_fields__
                }
            )
        else:
            cfg = config

        if n_channels != 2:
            raise ValueError(
                f"phase_aligned method requires exactly 2 channels, got {n_channels}"
            )

        return _combine_channels_phase_aligned(
            data_array,
            raw_fid_data=cfg.raw_fid_data,
            adjust_phase=cfg.adjust_phase,
            combination_method=cfg.combination_method,
            phase_diff_deg=cfg.phase_diff_deg,
        )

    elif method == "adaptive":
        if config is None:
            cfg = AdaptiveConfig()
        elif isinstance(config, dict):
            # Extract only valid fields for AdaptiveConfig
            valid_fields = AdaptiveConfig.__dataclass_fields__.keys()
            cfg_dict = {k: v for k, v in config.items() if k in valid_fields}
            cfg = AdaptiveConfig(**cfg_dict)
        else:
            cfg = config
        print(cfg)

        return _combine_channels_adaptive(
            data_array,
            reference_channel=cfg.reference_channel,
            auto_reference=cfg.auto_reference,
            adaptive_method=cfg.adaptive_method,
            domain=domain,
            weigh_reference=cfg.weigh_reference,
            window_size=cfg.window_size,
            return_phased_channels=cfg.return_phased_channels,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: rss, sum, mean, phase_aligned, weighted_mean, adaptive"
        )


def _combine_channels_rss(data_array: xr.DataArray) -> xr.DataArray:
    """Combine channels using root-sum-of-squares (RSS)."""
    if "channel_idx" not in data_array.dims:
        raise ValueError("Input data must have 'channel_idx' dimension")

    if data_array.sizes["channel_idx"] == 1:
        return data_array

    # Get channel axis index
    channel_axis = data_array.dims.index("channel_idx")

    # Calculate RSS: sqrt(sum(|channel|^2))
    if np.iscomplexobj(data_array.values):
        rss_values = np.sqrt(
            np.sum(np.abs(data_array.values) ** 2, axis=channel_axis, keepdims=True)
        )
    else:
        rss_values = np.sqrt(
            np.sum(data_array.values**2, axis=channel_axis, keepdims=True)
        )

    return _create_result_array(data_array, rss_values, "rss")


def _combine_channels_sum(data_array: xr.DataArray) -> xr.DataArray:
    """Combine channels by summing."""
    result_values = data_array.sum(dim="channel_idx", keepdims=True).values
    return _create_result_array(data_array, result_values, "sum")


def _combine_channels_mean(data_array: xr.DataArray) -> xr.DataArray:
    """Combine channels by averaging."""
    result_values = data_array.mean(dim="channel_idx", keepdims=True).values
    return _create_result_array(data_array, result_values, "mean")


def _combine_channels_weighted_mean(
    data_array: xr.DataArray, weights: Union[np.ndarray, xr.DataArray]
) -> xr.DataArray:
    """Combine channels by weighted averaging."""
    numerator = (data_array * weights).sum(dim="channel_idx", keepdims=True)
    ones = xr.ones_like(data_array)
    broadcasted_weights = ones * weights
    denominator = broadcasted_weights.sum(dim="channel_idx", keepdims=True)
    result_values = (numerator / denominator).values
    return _create_result_array(data_array, result_values, "weighted_mean")


def _combine_channels_phase_aligned(
    data_array: xr.DataArray,
    raw_fid_data: np.ndarray,
    adjust_phase: bool = True,
    combination_method: Literal["mean", "sum"] = "mean",
    phase_diff_deg: Optional[float] = None,
) -> xr.DataArray:
    """
    Combine 2 channels with phase alignment for spectroscopy data.

    Parameters
    ----------
    data_array : xr.DataArray
        Input data with channel_idx dimension.
    raw_fid_data : np.ndarray
        Raw FID data. First point is used for phase estimation. Order of data
        is assumed to be FID(ch1) - FID(ch2) - ...
    combination_method : {"mean", "sum"}, optional
        Combination method of the channels after data was phase aligned.
    phase_diff_deg : float
        The phase difference in degree between the 2 channels.

    """
    if "channel_idx" not in data_array.dims:
        raise ValueError("Input data must have 'channel_idx' dimension")
    if "acquisition_idx" not in data_array.dims:
        raise ValueError(
            "Input data must have 'acquisition_idx' dimension for phase-aligned combination"
        )
    if phase_diff_deg is None:
        if raw_fid_data is None:
            raise ValueError(
                "If no phase_diff_deg is provided, raw_fid_data is needed for phase estimation."
            )

    n_channels = data_array.sizes["channel_idx"]
    if n_channels != 2:
        raise ValueError(
            f"Phase-aligned combination requires exactly 2 channels, got {n_channels}"
        )

    acquisition_idx_length = data_array.sizes["acquisition_idx"]

    if phase_diff_deg is None:
        phase_diff_rad = np.angle(raw_fid_data[acquisition_idx_length]) - np.angle(
            raw_fid_data[0]
        )
        phase_diff_deg = phase_diff_rad * 180 / np.pi

    if not adjust_phase:
        phase_diff_deg = 0.0

    ch1_data = data_array.isel(channel_idx=0)
    ch2_data = data_array.isel(channel_idx=1)
    ch2_corrected = ch2_data * np.exp(-1j * np.deg2rad(phase_diff_deg))

    if combination_method == "mean":
        combined_values = ((ch1_data + ch2_corrected) / 2).values
    else:
        combined_values = (ch1_data + ch2_corrected).values

    # Add back the channel dimension for consistency in _create_result_array
    combined_values = np.expand_dims(
        combined_values, axis=data_array.dims.index("channel_idx")
    )

    result = _create_result_array(data_array, combined_values, "phase_aligned")
    result.attrs["phase_correction_deg"] = float(phase_diff_deg)
    return result


def _combine_channels_adaptive(
    data_array: xr.DataArray,
    reference_channel: int = 0,
    auto_reference: bool = True,
    domain: Optional[Literal["spatial", "k-space"]] = None,
    weigh_reference: bool = True,
    adaptive_method: Literal["global", "iterative", "sliding_window"] = "global",
    window_size: int = 5,
    return_phased_channels: bool = False,
) -> xr.DataArray:
    """
    Adaptive channel combination using phase alignment.

    Parameters
    ----------
    data_array : xr.DataArray
        Input data with channel_idx dimension.
    reference_channel : int, optional
        Index of the channel to use as phase reference. Default is 0.
    auto_reference : bool, optional
        If True, automatically select the reference channel with the highest mean magnitude. Default is True.
    domain : {"spatial", "k-space"}, optional
        The domain in which to perform the combination.
    weigh_reference : bool, optional
        If True, weigh the least-squares fit by the magnitude (only applies to 'global' and 'iterative'). Default is True.
    adaptive_method : {"global", "iterative", "sliding_window"}, optional
        The phase alignment method to use.
        - "global": single phase factor per channel using least-squares.
        - "iterative": aligns channels consecutively based on highest spatial overlap.
        - "sliding_window": applies a spatially varying phase correction. Default is "global".
    window_size : int, optional
        The size of the sliding window for the "sliding_window" method. Default is 5.
    return_phased_channels : bool, optional
        If True, return the phased channels before combination. Default is False.

    Returns
    -------
    xr.DataArray
        The combined single-channel data array, or phased multi-channel data array.
    """
    # Estimate the signal domain (k-space/spatial) of the input data
    original_domain = data_array.attrs.get("domain", "spatial")
    target_domain = domain or original_domain
    spatial_dimension = data_array.attrs.get("spatial_dimension", "2D")
    is_3d = False
    if spatial_dimension == "3D":
        is_3d = True

    working_data = data_array.copy()

    if target_domain == "k-space" and original_domain == "spatial":
        working_data = _transform_spatial_to_kspace(working_data, is_3d=is_3d)
    elif target_domain == "spatial" and original_domain == "k-space":
        working_data = _transform_kspace_to_spatial(working_data, is_3d=is_3d)

    n_channels = working_data.sizes["channel_idx"]
    n_slices = working_data.n_slices

    if is_3d:
        if auto_reference:
            dims_to_reduce = [d for d in working_data.dims if d != "channel_idx"]
            mean_magnitudes = np.abs(working_data).mean(dim=dims_to_reduce)
            reference_channel = int(mean_magnitudes.argmax(dim="channel_idx"))

        reference_scan = working_data.isel(channel_idx=reference_channel)

        vals = working_data.values
        ref_vals = reference_scan.values
        channel_axis = working_data.get_axis_num("channel_idx")

        phased_vals = np.zeros_like(vals, dtype=complex)

        if adaptive_method in ("global", "iterative"):
            vals_transposed = np.moveaxis(vals, channel_axis, -1)
            original_shape = vals_transposed.shape
            vals_flat = vals_transposed.reshape(-1, n_channels)
            ref_vals_flat = ref_vals.reshape(-1, 1)
            phased_vals_flat = np.zeros_like(vals_flat, dtype=complex)

            if adaptive_method == "global":
                for c in range(n_channels):
                    ck_flat = vals_flat[:, c : c + 1]
                    if weigh_reference:
                        pf, _, _, _ = np.linalg.lstsq(
                            ck_flat * np.abs(ck_flat),
                            ref_vals_flat * np.abs(ref_vals_flat),
                            rcond=None,
                        )
                    else:
                        pf, _, _, _ = np.linalg.lstsq(
                            ck_flat, ref_vals_flat, rcond=None
                        )

                    pf_normed = pf / np.abs(pf) if np.abs(pf) > 0 else 1.0
                    phased_vals_flat[:, c : c + 1] = ck_flat * pf_normed

            elif adaptive_method == "iterative":
                # Ensure overlap is calculated using spatial magnitudes
                if original_domain == "spatial":
                    spatial_vals = data_array.values
                else:
                    spatial_vals = _transform_kspace_to_spatial(
                        data_array, is_3d=is_3d
                    ).values

                spatial_vals_flat = np.moveaxis(spatial_vals, channel_axis, -1).reshape(
                    -1, n_channels
                )
                spatial_magnitudes = np.abs(spatial_vals_flat)
                overlap = spatial_magnitudes.T @ spatial_magnitudes

                aligned_indices = [reference_channel]
                unaligned_indices = [
                    i for i in range(n_channels) if i != reference_channel
                ]

                phased_vals_flat[:, reference_channel] = vals_flat[:, reference_channel]

                while unaligned_indices:
                    best_overlap, best_u, best_t = -1, -1, -1
                    for u in unaligned_indices:
                        for a in aligned_indices:
                            if overlap[u, a] > best_overlap:
                                best_overlap = overlap[u, a]
                                best_u = u
                                best_t = a

                    c_u = vals_flat[:, best_u : best_u + 1]
                    c_t = phased_vals_flat[:, best_t : best_t + 1]

                    if weigh_reference:
                        pf, _, _, _ = np.linalg.lstsq(
                            c_u * np.abs(c_u), c_t * np.abs(c_t), rcond=None
                        )
                    else:
                        pf, _, _, _ = np.linalg.lstsq(c_u, c_t, rcond=None)

                    pf_normed = pf / np.abs(pf) if np.abs(pf) > 0 else 1.0
                    phased_vals_flat[:, best_u : best_u + 1] = c_u * pf_normed

                    aligned_indices.append(best_u)
                    unaligned_indices.remove(best_u)

            phased_vals_transposed = phased_vals_flat.reshape(original_shape)
            phased_vals = np.moveaxis(phased_vals_transposed, -1, channel_axis)

        elif adaptive_method == "sliding_window":
            for c in range(n_channels):
                sl = [slice(None)] * vals.ndim
                sl[channel_axis] = c
                ck_vals = vals[tuple(sl)]

                cross_product = ck_vals * np.conj(ref_vals)

                smoothed_real = uniform_filter(cross_product.real, size=window_size)
                smoothed_imag = uniform_filter(cross_product.imag, size=window_size)
                smoothed_cross = smoothed_real + 1j * smoothed_imag

                pf_spatial = smoothed_cross / (np.abs(smoothed_cross) + 1e-8)
                phased_vals[tuple(sl)] = ck_vals * pf_spatial

    # --------------
    else:  # 2D
        if auto_reference:
            dims_to_reduce = [
                d for d in working_data.dims if d != "channel_idx" and d != "slice_idx"
            ]
            mean_magnitudes = np.abs(working_data).mean(dim=dims_to_reduce)
            # Get the per-slice reference channel:
            reference_channel = []
            for s in range(data_array.n_slices):
                reference_channel.append(
                    int(mean_magnitudes.isel(slice_idx=s).argmax(dim="channel_idx"))
                )

        vals = working_data.values

        channel_axis = working_data.get_axis_num("channel_idx")
        slice_axis = working_data.get_axis_num("slice_idx")

        phased_vals = np.zeros_like(vals, dtype=complex)

        if adaptive_method in ("global", "iterative"):
            vals_transposed = np.moveaxis(vals, [slice_axis, channel_axis], [-2, -1])
            original_shape = vals_transposed.shape

            vals_flat = vals_transposed.reshape(-1, n_slices, n_channels)
            phased_vals_flat = np.zeros_like(vals_flat, dtype=complex)

            for s in range(data_array.n_slices):
                # Handle auto_reference (list) vs explicit reference_channel (int)
                ref_c = (
                    reference_channel[s]
                    if isinstance(reference_channel, list)
                    else reference_channel
                )

                reference_scan = working_data.isel(slice_idx=s, channel_idx=ref_c)
                ref_vals = reference_scan.values
                ref_vals_flat = ref_vals.reshape(-1, 1)

                if adaptive_method == "global":
                    for c in range(n_channels):
                        ck_flat = vals_flat[:, s, c : c + 1]

                        if weigh_reference:
                            pf, _, _, _ = np.linalg.lstsq(
                                ck_flat * np.abs(ck_flat),
                                ref_vals_flat * np.abs(ref_vals_flat),
                                rcond=None,
                            )
                        else:
                            pf, _, _, _ = np.linalg.lstsq(
                                ck_flat, ref_vals_flat, rcond=None
                            )

                        pf_normed = pf / np.abs(pf) if np.abs(pf) > 0 else 1.0
                        phased_vals_flat[:, s, c : c + 1] = ck_flat * pf_normed

                elif adaptive_method == "iterative":
                    # Ensure overlap is calculated using spatial magnitudes
                    if original_domain == "spatial":
                        spatial_vals_da = data_array
                    else:
                        spatial_vals_da = _transform_kspace_to_spatial(
                            data_array, is_3d=is_3d
                        )

                    spatial_vals_slice = spatial_vals_da.isel(slice_idx=s)
                    current_channel_axis = spatial_vals_slice.get_axis_num(
                        "channel_idx"
                    )
                    spatial_vals = spatial_vals_slice.values

                    # move the current_channel_axis to the last dimension + flatten to be N x n_channels:
                    spatial_vals_flat = np.moveaxis(
                        spatial_vals, current_channel_axis, -1
                    ).reshape(-1, n_channels)

                    # Take magnitude
                    spatial_magnitudes = np.abs(spatial_vals_flat)
                    # Calculate the overlap of the image with itself
                    overlap = spatial_magnitudes.T @ spatial_magnitudes

                    aligned_indices = [ref_c]
                    unaligned_indices = [i for i in range(n_channels) if i != ref_c]

                    phased_vals_flat[:, s, ref_c] = vals_flat[:, s, ref_c]

                    while unaligned_indices:
                        best_overlap, best_u, best_t = -1, -1, -1
                        for u in unaligned_indices:
                            for a in aligned_indices:
                                if overlap[u, a] > best_overlap:
                                    best_overlap = overlap[u, a]
                                    best_u = u
                                    best_t = a
                        c_u = vals_flat[:, s, best_u : best_u + 1]
                        c_t = phased_vals_flat[:, s, best_t : best_t + 1]

                        if weigh_reference:
                            pf, _, _, _ = np.linalg.lstsq(
                                c_u * np.abs(c_u), c_t * np.abs(c_t), rcond=None
                            )
                        else:
                            pf, _, _, _ = np.linalg.lstsq(c_u, c_t, rcond=None)

                        pf_normed = pf / np.abs(pf) if np.abs(pf) > 0 else 1.0
                        phased_vals_flat[:, s, best_u : best_u + 1] = c_u * pf_normed

                        aligned_indices.append(best_u)
                        unaligned_indices.remove(best_u)

            # Reshape the flat data back to the intermediate transposed shape
            phased_vals_transposed = phased_vals_flat.reshape(original_shape)

            # Move axes back: [-2, -1] (slice, channel) -> [slice_axis, channel_axis]
            phased_vals = np.moveaxis(
                phased_vals_transposed, [-2, -1], [slice_axis, channel_axis]
            )

        elif adaptive_method == "sliding_window":
            # Build a reference volume dynamically slice-by-slice
            ref_vals_shape = list(vals.shape)
            ref_vals_shape.pop(channel_axis)
            ref_vals = np.zeros(ref_vals_shape, dtype=vals.dtype)

            out_slice_axis = slice_axis if slice_axis < channel_axis else slice_axis - 1

            for s in range(n_slices):
                sl_in = [slice(None)] * vals.ndim
                sl_in[slice_axis] = s

                ref_c = (
                    reference_channel[s]
                    if isinstance(reference_channel, list)
                    else reference_channel
                )
                sl_in[channel_axis] = ref_c

                sl_out = [slice(None)] * ref_vals.ndim
                sl_out[out_slice_axis] = s

                ref_vals[tuple(sl_out)] = vals[tuple(sl_in)]

            # Expand to match channel dimension for broadcasting
            ref_vals = np.expand_dims(ref_vals, axis=channel_axis)

            for c in range(n_channels):
                sl = [slice(None)] * vals.ndim
                sl[channel_axis] = c
                ck_vals = vals[tuple(sl)]

                # cross_product broadcasts against the per-slice reference correctly
                cross_product = ck_vals * np.conj(ref_vals[tuple(sl)])

                smoothed_real = uniform_filter(cross_product.real, size=window_size)
                smoothed_imag = uniform_filter(cross_product.imag, size=window_size)
                smoothed_cross = smoothed_real + 1j * smoothed_imag

                pf_spatial = smoothed_cross / (np.abs(smoothed_cross) + 1e-8)
                phased_vals[tuple(sl)] = ck_vals * pf_spatial

    # --------------
    if return_phased_channels:
        result = working_data.copy(data=phased_vals)
        result.attrs["channels_phased"] = "adaptive"
    else:
        combined_vals = np.sum(phased_vals, axis=channel_axis, keepdims=True)
        result = _create_result_array(working_data, combined_vals, "adaptive")

    result.attrs["reference_channel"] = reference_channel

    if target_domain != original_domain:
        if original_domain == "spatial":
            result = _transform_kspace_to_spatial(result, is_3d=is_3d)
        else:
            result = _transform_spatial_to_kspace(result, is_3d=is_3d)

    return result


def _transform_spatial_to_kspace(
    data: xr.DataArray, is_3d: bool = False
) -> xr.DataArray:
    """Transform DataArray from spatial to k-space domain."""
    res = data.copy()
    axes_to_transform = []
    if "read_idx" in data.dims:
        axes_to_transform.append(data.get_axis_num("read_idx"))
    if "phase_idx" in data.dims:
        axes_to_transform.append(data.get_axis_num("phase_idx"))
    if is_3d:
        if "slice_idx" in data.dims:
            axes_to_transform.append(data.get_axis_num("slice_idx"))

    if not axes_to_transform:
        return res

    vals = res.values
    for ax in axes_to_transform:
        vals = np.fft.ifftshift(
            np.fft.ifft(np.fft.ifftshift(vals, axes=ax), axis=ax), axes=ax
        )

    res.values = vals
    res.attrs["domain"] = "k-space"
    return res


def _transform_kspace_to_spatial(
    data: xr.DataArray, is_3d: bool = False
) -> xr.DataArray:
    """Transform DataArray from k-space to spatial domain."""
    res = data.copy()
    axes_to_transform = []
    if "read_idx" in data.dims:
        axes_to_transform.append(data.get_axis_num("read_idx"))
    if "phase_idx" in data.dims:
        axes_to_transform.append(data.get_axis_num("phase_idx"))
    if is_3d:
        if "slice_idx" in data.dims:
            axes_to_transform.append(data.get_axis_num("slice_idx"))

    if not axes_to_transform:
        return res

    vals = res.values
    for ax in axes_to_transform:
        vals = np.fft.fftshift(
            np.fft.fft(np.fft.fftshift(vals, axes=ax), axis=ax), axes=ax
        )

    res.values = vals
    res.attrs["domain"] = "spatial"
    return res


def _create_result_array(
    original_array: xr.DataArray, result_values: np.ndarray, method_name: str
) -> xr.DataArray:
    """Helper to create a result DataArray with consistent metadata."""
    # Create new coordinates, keeping channel_idx with size 1
    new_coords = {}
    for dim in original_array.dims:
        if dim in original_array.coords:
            coord = original_array.coords[dim]
            if dim == "channel_idx":
                new_coords[dim] = [0]
            elif "channel_idx" in getattr(coord, "dims", []):
                new_coords[dim] = coord.isel(channel_idx=0)
            else:
                new_coords[dim] = coord

    result = xr.DataArray(
        result_values,
        dims=original_array.dims,
        coords=new_coords,
        attrs=original_array.attrs.copy(),
    )
    result.attrs["channels_combined"] = method_name
    result.attrs["n_channels_combined"] = original_array.sizes["channel_idx"]
    return result
