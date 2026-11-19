"""
Physics-related functions for MRI calculations.

This module contains functions for:
- Magnetic field conversions (gauss_to_kHz, kHz_to_gauss, etc.)
- Gyromagnetic ratios (get_gmr, etc.)
- Nuclear physics (larmor frequency, zeeman energy, thermal polarization)
- Hyperpolarization (hyperpolarized_Mz_T1, etc.)
- Angle conversions (rad_to_deg, deg_to_rad, etc.)
- Flip angle corrections (flipangle_corr, etc.)
- Temperature conversions (celsius_to_kelvin, kelvin_to_celsius)
- RF power calculations (calc_B1_amp_G, calc_fa_deg, rfpower_from_refpower)
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..utils.system.logging import init_default_logger

# Physics constants
GAMMA_RATIOS = {
    "1H": 267522187,
    "1H (in H2O)": 267522187,
    "2H": 41066225,
    "3H": 285349779,
    "3He": 203789473,
    "7Li": 103977013,
    "13C": 67283474,
    "14N": 19337792,
    "15N": 27126030,
    "17O": 36281518,
    "19F": 251815948,
    "23Na": 70808493,
    "27Al": 69763060,
    "29Si": 53190356,
    "31P": 108394000,
    "57Fe": 8680624,
    "89Y": 13162791,
    "129Xe": 74521644,
    "195Pt": 57715900,
}


logger = init_default_logger(__name__)


def get_gmr(nucleus: str = "13c", unit: str = "MHz_T", db: bool = False):
    """
    Get the gyromagnetic ratio of nearly any nucleus.

    Parameters
    ----------
    nucleus: str
        can be 1h, 2h, 13c, ...
    unit: str
        Either MHz/T (MHz_T, "gamma_bar") or 10e6 rad/T/s (rad_Ts)

    Returns
    -------
    The gyromagnetic ratio in standard unit

    """
    # valid nuclei:
    nuclei = [
        "1H",
        "1H (in H2O)",
        "2H",
        "3H",
        "3He",
        "7Li",
        "13C",
        "14N",
        "15N",
        "17O",
        "19F",
        "23Na",
        "27Al",
        "29Si",
        "31P",
        "57Fe",
        "63Cu",
        "67Zn",
        "129Xe",
    ]

    nucleus = nucleus.lower()

    # Checking for valid nucleus and unit
    valid_nuclei = [nuc.lower() for nuc in nuclei]
    valid_units = ["rad_Ts", "MHz_T"]

    # Dictionary with gmr values (taken from wikipedia)
    # https://en.wikipedia.org/wiki/Gyromagnetic_ratio
    gmr_values = {
        "1h_rad_Ts": 267.52218744 * 1e6,
        "1h_MHz_T": 42.577478518,
        "1h_in_h2o_rad_Ts": 267.5153151 * 1e6,
        "1h_in_h2o_MHz_T": 42.57638474,
        "2h_rad_Ts": 41.065 * 1e6,
        "2h_MHz_T": 6.536,
        "3h_rad_Ts": 285.3508 * 1e6,
        "3h_MHz_T": 45.415,
        "3he_rad_Ts": -203.7894569 * 1e6,
        "3he_MHz_T": -32.43409942,
        "7li_rad_Ts": 103.962 * 1e6,
        "7li_MHz_T": 16.546,
        "13c_rad_Ts": 67.2828 * 1e6,
        "13c_MHz_T": 10.7084,
        "14n_rad_Ts": 19.331 * 1e6,
        "14n_MHz_T": 3.077,
        "15n_rad_Ts": -27.116 * 1e6,
        "15n_MHz_T": -4.316,
        "17o_rad_Ts": -36.264 * 1e6,
        "17o_MHz_T": -5.772,
        "19f_rad_Ts": 251.815 * 1e6,
        "19f_MHz_T": 40.078,
        "23na_rad_Ts": 70.761 * 1e6,
        "23na_MHz_T": 11.262,
        "27al_rad_Ts": 69.763 * 1e6,
        "27al_MHz_T": 11.103,
        "29si_rad_Ts": -53.19 * 1e6,
        "29si_MHz_T": -8.465,
        "31p_rad_Ts": 108.291 * 1e6,
        "31p_MHz_T": 17.235,
        "57fe_rad_Ts": 8.681 * 1e6,
        "57fe_MHz_T": 1.382,
        "63cu_rad_Ts": 71.118 * 1e6,
        "63cu_MHz_T": 11.319,
        "67zn_rad_Ts": 16.767 * 1e6,
        "67zn_MHz_T": 2.669,
        "129xe_rad_Ts": -73.997 * 1e6,
        "129xe_MHz_T": -11.777,
    }

    # Constructing the key for the dictionary lookup
    key = f"{nucleus.lower()}_{unit}"

    # check if passed input is valid:
    if nucleus.lower() not in valid_nuclei:
        return f"Unknown nucleus: {nucleus}. Please provide a valid nucleus."
    elif unit not in valid_units:
        return f"Unknown unit: {unit}. Please provide a valid unit (either 'rad_Ts' or 'MHz_T')."
    else:
        pass

    # print which unit is passed:
    if db:
        if unit == "MHz_T":
            print("in [MHz/T]")
        elif unit == "rad_Ts":
            print("in [10e6 rad/T/s]")
        else:
            print("unknown unit")

    # Returning the value if the key exists, otherwise return None
    return gmr_values.get(key, None)


def estimate_noise(
    input_data: np.ndarray = None,
    mask_dict: dict = None,
    mask_key: str = None,
    mask: str = None,
    masks_drawn_on="axial",
    slice_ind=None,
    signal_domain="temporal",
    component="complex",
    in_mask=False,
    plot=False,
    **kwargs,
):
    """
    Estimate the standard deviation of noise from MRI images.

    This function calculates the standard deviation of noise in MRI images
    using a specified mask or performs the calculations on all voxels if no mask is provided.
    Assumes first dimension to be in spectral dimension (to temporal=FID).
    The returned standard deviation of the noise is calculated from the temporal dimension
    to be easily usable with fitted amplitudes from utils_fitting.fit_data_pseudo_inv.

    Parameters
    ----------
    input_data : ndarray, optional
        The MRI data from which to estimate noise. The data should be a 5D
        ndarray.
    mask_dict : dict, optional
        A dictionary containing masks. A specific mask is selected using `mask_key`.
    mask_key : str, optional
        The key corresponding to the mask to be used from `mask_dict`.
    mask : ndarray, optional
        An alternative to using `mask_dict` and `mask_key`. Directly specify the mask to use.
        All voxels will be considered if no mask is provided.
    masks_drawn_on : str, default 'axial'
        The orientation of the masks.
    slice_ind : int, optional
        The specific slice index to use from `input_data`. Ignored if no mask is provided.
    signal_domain : str, optional
        input_data signal domain, can be spectral (default) or temporal. If spectral, an inverse FT is performed.
    in_mask: bool, optional
        Whether the noise is in the mask or outside the mask. Default is False.
    plot : bool, default False
        If True, plots the mask, real and imaginary components of the noise signal, and their histograms.
    kwargs : dict, optional
        Additional keyword arguments for extended functionality. Supported keywords include `return_all` to return all results.

    Returns
    -------
    std_noise : float
        The estimated standard deviation of the noise, calculated as the average of the standard
        deviations of the real and imaginary components of the signal.

    """
    from ..analysis.anatomical.masks import apply_mask
    from ..utils.mathematics.utils_fit import fit_rayleigh, rayleigh_pdf_fit

    # Validate input_data
    if input_data is None:
        raise ValueError("input_data must be provided!")

    # Process mask or use all voxels if no mask is provided
    if mask is None:
        if mask_dict is not None and mask_key is not None:
            try:
                mask = mask_dict[mask_key]
            except KeyError:
                raise ValueError(f"Cannot load mask_key {mask_key} from mask_dict!")
        else:
            # Use all voxels and skip slice operations
            mask = None
            warnings.warn(
                "No mask provided, using all voxels for noise estimation!", stacklevel=2
            )

    # Fourier transform based on signal domain
    if signal_domain == "spectral":
        input_data = np.fft.ifft(np.fft.ifftshift(input_data, axes=0), axis=0)

    # If mask is provided, use mask operations; otherwise, just flatten data
    if mask is not None:
        mask_inv = mask if in_mask else np.ones_like(mask) - mask

        # Apply mask and calculate noise
        noise = apply_mask(
            input_data,
            mask=mask_inv,
            mask_slice_ind=slice_ind,
            return_nans=True,
            masks_drawn_on=masks_drawn_on,
            bitwise=True,
        )
        noise_signal = noise[~np.isnan(noise)].flatten()
    else:
        noise_signal = input_data.flatten()

    # Binned dataset for noise distribution
    _, bins = np.histogram(np.abs(noise_signal), bins=30, density=1.0)
    # centers = (bins[:-1] + bins[1:]) / 2
    xlin = np.linspace(bins[0], bins[-1], 100)

    popt, _ = fit_rayleigh(input_data=np.abs(noise_signal), plot=False)

    # Calculate the mean and standard deviation of noise
    std_real = np.nanstd(np.real(noise_signal))
    std_imag = np.nanstd(np.imag(noise_signal))
    mean_real = np.nanmean(np.real(noise_signal))
    mean_imag = np.nanmean(np.imag(noise_signal))

    # Plotting section, if requested
    if plot:
        _, ax = plt.subplots(1, 5, tight_layout=True, figsize=(8, 3))

        if mask is not None:
            if masks_drawn_on == "axial":
                im = ax[0].imshow(
                    np.rot90(np.squeeze(mask_inv[0, slice_ind, :, :, 0, 0]))
                )
                plt.colorbar(im, ax=ax[0], label="Mask Coverage", shrink=0.3)
                ax[0].set_title("Inverse Signal Mask")
            else:
                im = ax[0].imshow(
                    np.rot90(np.squeeze(mask_inv[0, :, :, slice_ind, 0, 0]))
                )
                plt.colorbar(im, ax=ax[0], label="Mask Coverage", shrink=0.3)
                ax[0].set_title("Inverse Signal Mask")
        else:
            ax[0].text(0.5, 0.5, "No Mask Applied", ha="center", va="center")
            ax[0].set_title("All Voxels Used")

        ax[1].plot(np.real(noise_signal), color="blue")
        ax[1].plot(np.imag(noise_signal), color="red", alpha=0.5)
        ax[1].set_title("Signal Real/Imaginary")

        ax[2].hist(np.real(noise_signal), bins=30, color="blue", density=1.0)
        ax[2].hist(np.imag(noise_signal), bins=30, color="red", alpha=0.5, density=1)
        ax[2].set_title("Histogram Real/Imaginary")

        ax[3].plot(np.abs(noise_signal))
        ax[3].set_title("Signal Magnitude")

        ax[4].hist(np.abs(noise_signal), bins=30, density=1)
        ax[4].plot(xlin, rayleigh_pdf_fit(xlin, *popt), color="r")
        ax[4].axvline(np.mean(np.abs(noise_signal)), color="k")
        ax[4].axvline(popt[0], color="r")
        ax[4].axvline(popt[1], color="r")
        ax[4].set_title("Histogram Magnitude")
        plt.show()

    # Output the mean and standard deviation of noise
    print(f"Mean noise: {mean_real:.2f} + {mean_imag:.2f}*1i")
    print(f"Mean |noise|: {np.mean(np.abs(noise_signal))}")
    print(f"Standard deviation of noise: σ = {std_real:.2f} + {std_imag:.2f}*1i")
    print(f"Rayleigh Fit of |noise|: σ = {popt[1]:.2f}")

    # Determine the noise standard deviation based on the component setting
    if component == "complex":
        std_noise = (std_real + std_imag) / 2
    elif component in ["magnitude", "mag"]:
        std_noise = popt[1]
    else:
        std_noise = (std_real + std_imag) / 2

    if "return_all" in kwargs and kwargs["return_all"]:
        results = {}
        results["mean_real"] = mean_real
        results["mean_imag"] = mean_imag
        results["std_real"] = std_real
        results["std_imag"] = std_imag
        results["std_noise"] = std_noise
        results["rayleigh_fit"] = popt
        return results

    return std_noise


