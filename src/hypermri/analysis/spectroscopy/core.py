"""Core spectroscopy utilities shared across fitting workflows."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from ...utils.system.logging import debug_log, init_default_logger

DEBUG_FILE = "/tmp/analysis_spec_core.log"
DEBUG_LOG = True


def debug_log_spec_core(
    message: str = "", file_name=DEBUG_FILE, write_to_file=DEBUG_LOG
):
    """Debug logging system for spectroscopy core module."""
    debug_log(
        message=message,
        file_name=file_name,
        write_to_file=write_to_file,
    )


logger = init_default_logger(__name__, fstring="%(name)s - %(funcName)s - %(message)s ")


def get_freq_axis(
    data_obj: Any = None,
    unit: str = "ppm",
    cut_off: Optional[int] = None,
    npoints: Optional[int] = None,
    centered_at_0: bool = False,
    dwell_time_s: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Get frequency axis of given spectroscopy data_obj in unit of ppm or Hz.

    Parameters
    ----------
    data_obj: Any object

    unit : string
        Hz or ppm, Default value is ppm,

    cut_off : int
        number of points to remove from frequency axis [for stock Bruker 13c expirements]

    npoints: int
        number of points the frequency axis should have.

    Returns
    -------
    ppm_axis : np.ndarray
        Frequency axis of measurement in unit of ppm.
    """
    # Calculate frequency axis in ppm or Hz if no data_obj is passed
    # --------------------------------------------------------------
    debug_log_spec_core(
        f"get_freq_axis called with unit={unit}, cut_off={cut_off}, data_obj={data_obj}, npoints={npoints}, dwell_time_s={dwell_time_s}"
    )
    if data_obj is None:
        if dwell_time_s is None or npoints is None:
            Warning("if data_obj=None pass dwell_time_s and npoints!")
            return None
        if unit == "ppm":
            raise ValueError("if data_obj=None this can only return unit=Hz axis!")
            return None

        if isinstance(npoints, tuple) and len(npoints) == 1:
            npoints = npoints[0]

        if cut_off is None:
            cut_off = 0

        # define frequency axis in Hz
        hz_axis = np.fft.fftshift(np.fft.fftfreq(npoints, d=dwell_time_s))
        if centered_at_0 is True:
            hz_axis = center_freq_axis(freq_axis=hz_axis)
            return hz_axis
        else:
            return hz_axis
    else:
        if cut_off is None:
            cut_off = data_obj.get("cut_off", 0)

    # Calculate frequency axis in ppm or Hz if data_obj is passed:
    # ------------------------------------------------------------
    if not hasattr(data_obj, "bandwidth_hz"):
        debug_log_spec_core(
            "data_obj has no attribute bandwidth_hz, cannot calculate freq axis"
        )
        return None

    # if spectral range should be centered:
    if centered_at_0 is None:
        # usually ppm ranges should not be centered around 0 ppm:
        if unit == "ppm":
            centered_at_0 = False
        # usually Hz ranges should be centered around 0 Hz:
        elif unit == "Hz":
            centered_at_0 = False
        else:
            centered_at_0 = False

    # define center frequency:
    center_ppm = data_obj.get("center_ppm", 0)
    debug_log_spec_core(f"center_ppm={center_ppm}")

    # define bandwidth [Hz]:
    bw_hz = data_obj.bandwidth_hz
    if bw_hz is None:
        return []

    # sampling rate [s]:
    from ...utils.time.temporal import get_dwell_time_s

    # get sampling duration [s]
    dt = get_dwell_time_s(data_obj=data_obj)
    debug_log_spec_core(f"dt={dt}")

    # if no number of points was specified, use scan info, else use passed
    # number of points:
    if npoints is None:
        # number of acquisition points:
        freq_points = int(data_obj.n_acquisition - cut_off)
    else:
        freq_points = npoints

    debug_log_spec_core(f"freq_points={freq_points}")

    # define frequency axis in [Hz]
    hz_axis = np.fft.fftshift(np.fft.fftfreq(freq_points, d=dt))

    if unit.lower() == "ppm":
        # define bandwidth [ppm]:
        bw_ppm = getattr(data_obj, "bandwidth_ppm", None)
        debug_log_spec_core(f"bw_ppm={bw_ppm} ppm")

        if bw_ppm is None or bw_ppm == 0:
            debug_log_spec_core("bandwidth_ppm is None or 0, cannot calculate ppm axis")
            return None

        # converison factor:
        hz_to_ppm = bw_hz / bw_ppm

        debug_log_spec_core(f"hz_to_ppm={hz_to_ppm} Hz/ppm")

        # translate to [ppm]
        ppm_axis = hz_axis / hz_to_ppm

        debug_log_spec_core(f"ppm_axis (before centering)={ppm_axis}")

        # add offset:
        ppm_axis = ppm_axis + center_ppm

        # center around 0:
        if centered_at_0:
            ppm_axis = center_freq_axis(freq_axis=ppm_axis)
        debug_log_spec_core(f"ppm_axis (after centering)={ppm_axis}")
        return ppm_axis

    elif unit.lower() == "hz":
        # center around 0 Hz:
        if centered_at_0:
            hz_axis = center_freq_axis(freq_axis=hz_axis)

        return hz_axis

    else:
        Warning(f"unknown frequency unit {unit}")


def conv_func(
    name: str = None, lb_hz: float = 0.0, time_s: np.ndarray = None
) -> np.ndarray:
    """Define a convolution function (for linebroadening)

    Parameters
    ----------
    name: str
        Name of convolution function, e.g. "Gaussian", "Lorentzian", default is "Gaussian"
    lb_hz: float
        Linebroadening in Hz
    time_s: array
        time in seconds

    Returns
    -------

    """
    if name == "Gaussian":
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))

    elif name == "Lorentzian":
        filter = np.exp(-(lb_hz * time_s))

    elif name is None:
        # default Gaussian:
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))
    else:
        # default Gaussian:
        filter = np.exp(-((np.pi * lb_hz * time_s) ** 2) / (2 * np.log(2)))
    return filter


def multi_dim_linebroadening(
    input_data: Any = None,
    lb_hz: float = 0.0,
    fid_start: int = 0,
    fid_end: int = 0,
    dwell_time_s: float = 0,
    conv_func_name: str = "Gaussian",
    signal_domain: str = "spectral",
    axis: int = 0,
    data_obj: Any = None,
) -> Optional[np.ndarray]:
    """
    Applies Gaussian linebroadening on spectra in all voxel

    Parameters
    ----------
    data_obj : MRIExperiment
    input_data: N-D array
        echoes - z - x -y -rep - chans
    lb_hz: float
        linebroadening in Hz
    fid_start: Integer
        Start index
    fid_end: Integer
        End index
    dwell_time_s: float
        sampling time per FID point [second]
    signal_domain: string
        whether data is in spectral or temporal (FID) domain

    Returns
    -------
    linebroadend_data: N-D array

    """
    if input_data is None:
        try:
            input_data = data_obj.reco_data
        except BaseException:
            raise Exception("provide either input_data or data_obj")

    else:
        pass

    if (
        dwell_time_s == 0 and data_obj is not None
    ):  # define sampling duration to do: check
        from ...utils.time.temporal import get_dwell_time_s

        dwell_time_s = get_dwell_time_s(data_obj=data_obj)
    elif dwell_time_s == 0 and data_obj is None:
        return None
    else:
        pass

    def wrapper_apply_lb(arr):
        return apply_lb(
            input_data=arr,
            lb_hz=lb_hz,
            fid_start=fid_start,
            fid_end=fid_end,
            dwell_time_s=dwell_time_s,
            conv_func_name=conv_func_name,
            signal_domain=signal_domain,
            data_obj=data_obj,
        )

    # Apply along the first axis (axis=0)
    linebroadend_data = np.apply_along_axis(wrapper_apply_lb, axis=axis, arr=input_data)

    return linebroadend_data


def apply_lb(
    input_data: np.array = None,
    lb_hz: float = 0.0,
    fid_start: int = 0,
    fid_end: int = 0,
    dwell_time_s: float = 0.0,
    conv_func_name: str = "Gaussian",
    signal_domain: str = "spectral",
    data_obj: Any = None,
    plot: bool = False,
) -> Optional[np.ndarray]:
    """
    Apply linebroadening to the input data.

    Parameters
    ----------
    input_data: 1D-array
        1D-spectrum or FID (use signal_domain to specify).
    lb_hz: float
        linebroadening in Hz.
    fid_start: int
        At which index the FID starts.
    fid_end: int
        At which index the FID ends.
    dwell_time_s: float
        dwell time in seconds.
    conv_func_name: {"Gaussian"}, string
        Which function to use in the convolution. Options: Gaussian, Lorentzian.
    signal_domain: {"spectral", "temporal"}, string
        In which domain the input data is.
    data_obj: hypermri object
        data object from which the sampling time can be extracted.
    plot: bool
        Select if plotting for checking is wanted.

    Returns
    -------
    A sepctrum on which linebroadening has been applied. If the input data was in the temporal domain (aka FID), a
    spectrum will be returned. If the data is in the spectral domain, a linebroadened spectrum will be returned.

    Examples
    --------
    >>> from hypermri.analysis.core import apply_lb, generate_fid
    >>> from hypermri.utils.time.temporal import calc_sampling_time_axis
    >>> dt = 5e-4
    >>> npoints=1000
    >>> time_axis_s = calc_sampling_time_axis(fid_start=0, fid_end=npoints, dwell_time_s=dt)

    >>> f = np.fft.fftshift(np.fft.fftfreq(npoints, d=dt))
    >>> lb_hz=20.0
    >>> fid = generate_fid(time_axis_s=time_axis_s, T2_s=0.1, freq0_Hz=100.0, phase0_rad=0.5, dwell_time_s=dt,
    >>> npoints=npoints)
    >>> spec_lb = apply_lb(input_data=fid, lb_hz=lb_hz, dwell_time_s=dt, signal_domain="temporal")

    >>> plt.figure()
    >>> plt.subplot(1,2,1)
    >>> plt.plot(time_axis_s, np.real(fid))
    >>> plt.plot(time_axis_s, np.imag(fid))
    >>> plt.plot(time_axis_s, np.abs(fid))
    >>> plt.xlabel('t [s]')

    >>> plt.subplot(1,2,2)
    >>> plt.plot(f, np.abs(np.fft.fftshift(np.fft.fft(fid)))/np.max(np.abs(np.fft.fftshift(np.fft.fft(fid)))),
    >>> label='original FID')
    >>> plt.plot(f, np.abs(spec_lb)/np.max(np.abs(spec_lb)),label=f'linebroadened FID {lb_hz}Hz')
    >>> plt.xlabel('f [Hz]')
    >>> plt.legend()
    """
    # check if data was input:
    if input_data is None:
        raise Exception("No input_data")
    else:
        pass

    input_data = np.atleast_1d(np.squeeze(input_data))

    # apply 1D Fourier-Transform
    if signal_domain == "temporal":
        pass
    elif signal_domain == "spectral":
        input_data = np.fft.ifft(np.fft.ifftshift(input_data, axes=(0,)), axis=0)

    # check dimensions of input data:
    if input_data.ndim == 1:
        if fid_end == 0:
            fid_end = input_data.shape[0]
        elif fid_end > input_data.shape[0]:
            fid_end = input_data.shape[0]
        elif fid_end < fid_start:
            raise Exception("fid_end has to be higher than fid_start")
        else:
            pass
    else:
        raise Exception("input_data has to have dim=1")

    if (
        dwell_time_s == 0 and data_obj is not None
    ):  # define sampling duration to do: check
        from ...utils.time.temporal import get_dwell_time_s

        dwell_time_s = get_dwell_time_s(
            data_obj=data_obj
        )  # data_obj.method["PVM_SpecDwellTime"] * 2.0 / 1000.0 / 1000.0
    elif dwell_time_s == 0 and data_obj is None:
        return None
    else:
        pass

    # get time axis in seconds
    if dwell_time_s == 0:
        raise Exception("sampling duration has to be > 0")
    else:
        from ...utils.time.temporal import calc_sampling_time_axis

        time_ax = calc_sampling_time_axis(
            npoints=len(input_data), dwell_time_s=dwell_time_s
        )

    # define filter:
    filter = conv_func(name=conv_func_name, lb_hz=lb_hz, time_s=time_ax)

    fid_times_dec_exp = input_data[fid_start:fid_end] * filter
    if signal_domain == "temporal":
        return_lb = fid_times_dec_exp
    elif signal_domain == "spectral":
        return_lb = np.fft.fftshift(np.fft.fft(fid_times_dec_exp, axis=0), axes=(0,))
    else:
        return_lb = np.fft.fftshift(np.fft.fft(fid_times_dec_exp, axis=0), axes=(0,))
        warnings.warn(
            "signal_domain not recognized - returning spectrum as output - use signal_domain='spectral' or 'temporal'",
            stacklevel=2,
        )
    if plot:
        _, ax = plt.subplots(1, 1)
        ax.plot(time_ax, np.real(input_data), label="original FID")
        ax.plot(time_ax, np.real(fid_times_dec_exp), label="linebroadened FID")
        ax.set_xlabel("t [s]")
        ax.legend()

    return return_lb


def zeropad_data(
    input_data: np.ndarray = None,
    num_zp: int = 0,
    fid_start: int = 0,
    fid_end: int = 0,
    signal_domain: str = "spectral",
):
    """
    Zeropad multidimensional data along temporal domain. Return data in the set domain.

    Parameters
    ----------
    input_data: 1D-array
        1D-spectrum
    num_zp: float
        number of points to add to the fid
    signal_domain: {"spectral", "temporal"}, string
        In which domain the input data is

    Returns
    -------
    Data that has been zeropadded. If input data was in the temporal domain (aka FID), a zeropadded FID is returned. If
    the data is in the spectral domain, a zeropadded spectrum will be returned.

    """
    # check if data was input:
    if input_data is None:
        raise Exception("No input_data")
    else:
        pass

    # apply 1D Fourier-Transform
    if signal_domain == "temporal":
        pass
    elif signal_domain == "spectral":
        input_data = np.fft.ifft(np.fft.ifftshift(input_data, axes=(0,)), axis=0)

    # check dimensions of input data:
    if input_data.ndim == 1:
        if fid_end == 0:
            fid_end = input_data.shape[0]
        elif fid_end > input_data.shape[0]:
            fid_end = input_data.shape[0]
        elif fid_end < fid_start:
            raise Exception("fid_end has to be higher than fid_start")
        else:
            pass
        fid = input_data[fid_start:fid_end]
    else:
        raise Exception("input_data has to have dim=1")

    # append zeros to the FID:
    fid_zp = np.append(fid, np.zeros((num_zp, 1), dtype=type(fid)))
    fid_zp = np.pad(fid, (0, num_zp), mode="constant", constant_values=(0, 0))
    if signal_domain == "temporal":
        return fid_zp
    elif signal_domain == "spectral":
        # apply 1D Fourier-Transform
        spec_zp = np.fft.fftshift(np.fft.fft(fid_zp, axis=0), axes=(0,))
        return spec_zp


# chemical shift in ppm:
METABOLITE_CS_PPM = {
    "pyruvate": 171.076,  # Lactate - 920Hz
    "lactate": 183.35,  # determined from repeated experiments
    "alanine": 176.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "pyruvatehydrate": 179.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "fumarate": 175.4,
    "malate1": 181.7,  # https://pubs.acs.org/doi/epdf/10.1021/jacs.9b10094
    "malate4": 180.5,  # https://pubs.acs.org/doi/epdf/10.1021/jacs.9b10094
    # https://pubmed.ncbi.nlm.nih.gov/30639333/ (fits human in vivo data @
    # 3T)
    "bicarbonate": 161.0,
    "urea": 163.5,  # trust me bro
    "co2": 124.5,  # https://www.pnas.org/doi/10.1073/pnas.0706235104
    "aspartate1": 176.92,  # doi: 10.3389/fphys.2021.792769
    "aspartate4": 180.20,  # doi: 10.3389/fphys.2021.792769
}


def freq_Hz_to_ppm(
    freq_Hz: float = 0.0,
    hz_axis: Optional[np.array] = None,
    ppm_axis: Optional[np.array] = None,
    ppm_axis_flipped: bool = False,
    data_obj: Any = None,
    ppm_centered_at_0: bool = False,
) -> Optional[Union[float, np.ndarray]]:
    """
    Converts frequency values from Hz to parts per million (ppm) based on provided Hz and ppm axes.

    This function allows for the conversion of frequency values from Hertz (Hz) to parts per million (ppm),
    which is often required in spectroscopy for chemical shift calculations. The conversion relies on
    matching frequency values in Hz to their corresponding values in ppm using provided frequency axes.
    The function can handle single values, lists, or arrays of frequencies to be converted.

    Parameters
    ----------
    freq_Hz : float or ndarray, optional
        Frequency value(s) in Hz to be converted. Can be a single value, list, or ndarray. Default is 0.
    hz_axis : ndarray, optional
        The frequency axis in Hz. Required if `data_obj` is not provided. Default is None.
    ppm_axis : ndarray, optional
        The frequency axis in ppm. Required if `data_obj` is not provided. Default is None.
    ppm_axis_flipped : bool, optional
        Set to True if the ppm axis is in descending order. This affects the conversion calculation.
        Default is False.
    data_obj : MRIExperiment, optional
        An object containing the Hz and ppm axes, used if either axis is not explicitly provided.
        Default is None.
    ppm_centered_at_0 : bool, optional
        If True, centers the ppm axis at 0. This parameter is considered in the conversion calculation.
        Default is False.

    Returns
    -------
    ndarray or float
        The converted frequency values in ppm. The return type matches the type of `freq_Hz` input
        (i.e., if `freq_Hz` is a single value, a single float is returned; if it's an array, an ndarray is returned).

    Raises
    ------
    ValueError
        If neither `hz_axis` nor `ppm_axis` is provided and `data_obj` is None.

    Examples
    --------
    >>> hz_axis = np.linspace(-2000, 2000, 4000)
    >>> ppm_axis = np.linspace(10, -10, 4000)
    >>> freq_Hz_to_ppm(freq_Hz=500, hz_axis=hz_axis, ppm_axis=ppm_axis)
    2.5
    """
    if hz_axis is None or ppm_axis is None:
        if data_obj is None:
            raise ValueError("Provide either hz_axis and ppm_axis or data_obj!")
            return None
        if hz_axis is None:
            hz_axis = get_freq_axis(
                data_obj=data_obj, unit="Hz"
            )  # Assuming get_freq_axis is defined elsewhere
        if ppm_axis is None:
            ppm_axis = get_freq_axis(
                data_obj=data_obj, unit="ppm"
            )  # Assuming get_freq_axis is defined elsewhere

    if ppm_axis_flipped:
        ppm_axis = np.flip(ppm_axis)

    if ppm_centered_at_0:
        ppm_axis = center_freq_axis(
            freq_axis=ppm_axis
        )  # Assuming center_freq_axis is defined elsewhere
        hz_axis = center_freq_axis(
            freq_axis=hz_axis
        )  # Assuming center_freq_axis is defined elsewhere

    # calculate a "conversion function" using linear regression
    from scipy.stats import linregress

    f_hz_to_ppm = linregress(hz_axis, ppm_axis)

    input_was_list = isinstance(freq_Hz, list)  # Check if input is a list

    # Ensure freq_Hz is an array for vectorized operations
    freq_Hz = np.asarray(freq_Hz)

    # Convert using the linear regression results
    freq_ppm = f_hz_to_ppm.intercept + f_hz_to_ppm.slope * freq_Hz

    # Return the output in the same type as the input
    if input_was_list:
        return freq_ppm.tolist()  # Convert back to list if the input was a list
    elif isinstance(freq_Hz, np.ndarray):
        return freq_ppm  # Keep as numpy array if the input was an array
    else:
        return float(freq_ppm)  # Convert to float if the input was a scalar


def center_freq_axis(freq_axis: np.array = None) -> np.array:
    """
    Center the inout freq_axis around 0.

    Parameters
    ----------
    freq_axis : np.array
        The frequency axis to be centered.

    Returns
    -------
    np.array
        The centered frequency axis.

    """
    # calculate translation term:
    freq_axis_centered_around_0 = freq_axis
    # from 0 to max:
    freq_axis_centered_around_0 = freq_axis_centered_around_0 - np.min(
        freq_axis_centered_around_0
    )
    # center around 0:
    freq_axis_centered_around_0 = (
        freq_axis_centered_around_0 - freq_axis_centered_around_0[-1] / 2.0
    )

    return freq_axis_centered_around_0


def make_NDspec_6Dspec(input_data: np.ndarray = None, provided_dims: tuple = None):
    """
    Transforms an N-dimensional (ND) array to a 6-dimensional array, with the option to reorder the dimensions.

    Parameters
    ----------
    input_data : ndarray
        The input array to transform. Can be of any shape with dimensions N where N <= 6.
    provided_dims : tuple of ints, optional
        Specifies the new order of the original dimensions in the resulting 6D array. The length of this tuple should
        match the number of dimensions in `input_data`. Each element in the tuple represents the index of the dimension
        in the output 6D array. If `None`, the original order is preserved in the first N dimensions, and the rest are
        filled with singleton dimensions.

    Returns
    -------
    output_data : ndarray
        The resulting 6D array with possibly reordered dimensions. If the input array had fewer than 6 dimensions, the
        additional dimensions are singleton (i.e., of size 1).

    Examples
    --------
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> make_NDspec_6Dspec(a).shape
    (2, 3, 1, 1, 1, 1)

    >>> make_NDspec_6Dspec(a).shape
    (2, 3, 1, 1, 1, 1)

    >>> make_NDspec_6Dspec(a, provided_dims=(1,0)).shape
    (3, 2, 1, 1, 1, 1)

    >>> make_NDspec_6Dspec(a, provided_dims=(5,0)).shape
    (3, 1, 1, 1, 1, 2)

    >>> aa = np.arange(720).reshape((1,2,3,4,5,6))
    >>> make_NDspec_6Dspec(aa, provided_dims=(1,2,3,4,5,0)).shape
    (6, 1, 2, 3, 4, 5)
    """
    if input_data is None:
        raise ValueError("Input data cannot be None")

    # Expand input_data to 6D if it has fewer dimensions, using np.newaxis to
    # add singleton dimensions
    while np.ndim(input_data) < 6:
        input_data = np.expand_dims(input_data, axis=-1)

    # If provided_dims is None, use the existing order, otherwise calculate
    # the new order
    if provided_dims is None:
        provided_dims = range(np.ndim(input_data))

    if any(isinstance(elem, str) for elem in provided_dims):
        from ...utils.image.dimensions import get_indices_for_strings

        provided_dims = get_indices_for_strings(strings=provided_dims)

    else:
        provided_dims = list(provided_dims)

    # Calculate the positions of missing dimensions if the provided_dims does
    # not cover all 6 dimensions
    missing_dims = [dim for dim in range(6) if dim not in provided_dims]

    # Combine provided_dims with missing_dims to cover all 6 dimensions
    all_dims = provided_dims + missing_dims
    new_order = [all_dims.index(i) for i in range(len(all_dims))]

    # Reorder dimensions according to all_dims
    output_data = np.transpose(input_data, axes=new_order)

    return output_data


