"""
Mathematical calculations and curve fitting utilities.

This module contains basic mathematical functions extracted from utils_general.py
as part of the Phase 2 refactoring to create focused utility modules.
"""

import numpy as np

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def calc_sampling_time_axis(dwell_time_s=0, npoints=0, data_obj=None, bw_hz=0):
    """Create the sampling time axis associated with a dwell time.

    Parameters
    ----------
    dwell_time_s : float
        Dwell time between consecutive samples in seconds.
    npoints : int
        Number of points in the acquisition.
    data_obj : object, optional
        Data object to extract acquisition parameters from.
    bw_hz : float, optional
        Spectral bandwidth in Hz.

    Returns
    -------
    numpy.ndarray
        Equally spaced time axis in seconds.
    """
    if dwell_time_s <= 0:
        if bw_hz > 0:
            dwell_time_s = 1 / bw_hz
        else:
            dwell_time_s = 1e-3

    if data_obj is not None:
        if npoints == 0 or npoints is None:
            npoints = data_obj.n_acquisition - getattr(data_obj, "cut_off", 0)

    return np.linspace(0, dwell_time_s * (npoints - 1), npoints)


def part_func(input_data=None, signal_part="Magnitude"):
    """Extract a chosen component from a complex-valued signal.

    Parameters
    ----------
    input_data : array_like, optional
        Complex-valued samples from which to extract a component.
    signal_part : {"Real", "Imaginary", "Phase", "Magnitude"}, optional
        Component to extract. Any other value defaults to ``"Magnitude"``.

    Returns
    -------
    numpy.ndarray
        Requested component of ``input_data``. Returns ``None`` when
        ``input_data`` is ``None``.

    Notes
    -----
    The function silently falls back to the magnitude if an unsupported
    ``signal_part`` is provided.
    """
    if input_data is None:
        return None

    match signal_part:
        case "Real":
            return np.real(input_data)
        case "Imaginary":
            return np.imag(input_data)
        case "Phase":
            return np.angle(input_data)
        case "Magnitude":
            return np.abs(input_data)
        case _:
            return np.abs(input_data)
