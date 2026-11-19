from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from hypermri.core.experiment import MRIExperiment

from ..utils.system.logging import LOG_MODES, init_default_logger
from .base_spectroscopy import BaseSpectroscopy

# initialize logger
logger = init_default_logger(__name__)
logger.setLevel(LOG_MODES["Info"])
DEBUG_FILE = "/tmp/csi_debug.log"
DEBUG_LOG = True


class CSI(BaseSpectroscopy):
    """Chemical shift imaging sequence with spectroscopy utilities.

    The class extends :class:`~hypermri.sequences.base_spectroscopy.BaseSpectroscopy`
    with helpers for channel-aware reconstructions (e.g. :meth:`combine_channels`),
    interactive signal filtering, and voxel-wise fitting visualisations used in the
    spectroscopy tutorials.
    """

    def __init__(self, mri_exp_obj: MRIExperiment, *args, **kwargs):
        # Extract sequence-specific kwargs before calling parent constructor
        reco_options = kwargs.pop("reco_options", {})

        super().__init__(mri_exp_obj, *args, **kwargs)

        # Check if FID data is available for CSI reconstruction
        # In minimal/lazy loading modes, we allow CSI to load even without FID data
        if not hasattr(self.handler, "fid") or self.handler.fid is None:
            logger.warning(
                "CSI: No FID data available - this may limit functionality but allowing initialization to continue"
            )
        else:
            try:
                self.reconstruct(reco_type="raw", **reco_options)
                logger.info("CSI: Raw reconstruction successful")
            except Exception as e:
                logger.error(f"CSI raw reconstruction failed: {e}")
                logger.error(
                    "CSI: Continuing with initialization but no reconstruction"
                )
                # Continue - this allows CSI to load successfully even if reconstruction fails

    # @property

    @property
    def data(self):
        """
        Return the primary data for CSI, preferring broadened data if available.
        Uses standardized data access patterns.
        """
        # Check for CSI-specific broadened data first
        if hasattr(self, "seq2d_broadend") and self.seq2d_broadend is not None:
            return self.seq2d_broadend

        # Fall back to primary data array
        if self.primary_data is not None:
            return self.primary_data

        # Legacy fallback
        if hasattr(self, "_legacy_seq2d") and self._legacy_seq2d is not None:
            return self._legacy_seq2d

        return None

    @property
    def data_array(self):
        """
        Alias for primary_data to maintain backward compatibility.
        CSI raw reconstruction stores data in reco_data, which primary_data accesses correctly.
        """
        return self.primary_data

    @data_array.setter
    def data_array(self, value):
        """Setter that delegates to the mixin's data_array setter."""
        # Call the parent's data_array setter
        BaseSpectroscopy.data_array.fset(self, value)


    def combine_channels(
        self,
        input_data: xr.DataArray | None = None,
        channel_combination: str = "mean",
        adjust_phase: bool = True,
        plot: bool = False,
        config: Any = None,
    ) -> xr.DataArray:
        """Combine receive channels after optional phase alignment.

        The method measures the phase offset between the raw FID traces of two
        receive channels, corrects the second channel of ``input_data`` if
        ``adjust_phase`` is true, and combines both channels. The combined dataset is stored on the
        object as :attr:`reco_data_combined_channels` and returned for convenience.

        .. deprecated::
            This method is being refactored. For general channel combination,
            consider using :func:`hypermri.utils.mri.channels.combine_channels`.

        Parameters
        ----------
        input_data : xarray.DataArray, optional
            Spectroscopic data with ``channel_idx`` and ``acquisition_idx``
            dimensions. Defaults to :attr:`reco_data`.
        channel_combination : {"mean", "sum", "rss"}, optional
            How to combine the aligned channels. Default is "mean".
        adjust_phase : bool, optional
            If ``True`` (default) apply the measured phase difference to the second
            channel before combining. Disable when the hardware stack already
            delivers phase-matched channels.
        plot : bool, optional
            Show diagnostic figures for the raw and corrected FIDs.
        config : any, optional
            Configuration object for channel combination. If provided, overrides
            other parameters.

        Returns
        -------
        xarray.DataArray
            Combined spectroscopic data with channel_idx dimension removed.

        Raises
        ------
        ValueError
            Raised when the input is missing ``channel_idx``/``acquisition_idx``
            dimensions or does not contain two channels.
        """
        if input_data is None:
            input_data = self.reco_data

        # Check for required dimensions
        if "channel_idx" not in input_data.dims:
            raise ValueError("Input data must have 'channel_idx' dimension")
        if "acquisition_idx" not in input_data.dims:
            raise ValueError("Input data must have 'acquisition_idx' dimension")

        # Get acquisition_idx size from the data
        acquisition_idx_length = input_data.sizes["acquisition_idx"]

        # Get number of channels
        n_channels = input_data.sizes["channel_idx"]

        # Check if we have 2 channels
        if n_channels != 2:
            raise ValueError(
                f"Input data must have at 2 channels for phase correction, has {n_channels}!"
            )

        # Optional plotting of reconstruction data
        if plot:
            fig, ax = plt.subplots(2, 2, figsize=(6, 4))
            ax = ax.flatten()

            image_ch1 = input_data.isel(channel_idx=0).squeeze()
            image_ch2 = input_data.isel(channel_idx=1).squeeze()
            print(f"Channel 1 shape: {image_ch1.shape}")

            # Plot original data
            ax[0].imshow(np.squeeze(np.real(image_ch1[0, ...])))
            ax[0].set_title("Ch1 (Image))")

            # Get axis indices for FFT
            read_idx = input_data.get_axis_num("read_idx")
            phase_idx = input_data.get_axis_num("phase_idx")

            # Perform 2D IFFT for channel 1
            kspace_ch1 = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(image_ch1, axes=read_idx), axis=read_idx),
                axes=read_idx,
            )
            kspace_ch1 = np.fft.fftshift(
                np.fft.ifft(
                    np.fft.ifftshift(kspace_ch1, axes=phase_idx), axis=phase_idx
                ),
                axes=phase_idx,
            )
            ax[1].imshow(np.squeeze(np.abs(kspace_ch1[0, ...])))
            ax[1].set_title("Ch1 (k-space)")

            # Perform 2D IFFT for channel 2
            kspace_ch2 = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(image_ch2, axes=read_idx), axis=read_idx),
                axes=read_idx,
            )
            kspace_ch2 = np.fft.fftshift(
                np.fft.ifft(
                    np.fft.ifftshift(kspace_ch2, axes=phase_idx), axis=phase_idx
                ),
                axes=phase_idx,
            )
            ax[2].imshow(np.squeeze(np.abs(image_ch2[0, ...])))
            ax[2].set_title("Ch2 (image)")
            ax[3].imshow(np.squeeze(np.abs(kspace_ch2[0, ...])))
            ax[3].set_title("Ch2 (k-space)")

            plt.tight_layout()
            plt.show()

        # Process raw FID data for phase correction
        fid = self.handler.rawdatajob0

        # Optional plotting of FID data

        # Calculate phase difference between channels
        phase_diff = np.angle(fid[acquisition_idx_length]) - np.angle(fid[0])
        phase_diff_degrees = phase_diff * 180 / np.pi

        print(
            f"Phase difference between channels: {phase_diff:.4f} rad ({phase_diff_degrees:.2f}°)"
        )
        if not adjust_phase:
            phase_diff = 0.0
            print("Phase adjustment disabled, proceeding without phase correction.")

        # Reshape FID data and separate channels
        fid = np.reshape(fid, (-1, acquisition_idx_length))

        fid_ch1 = fid[::2, :]  # Even indices (channel 1)
        fid_ch2 = fid[1::2, :]  # Odd indices (channel 2)

        # Apply phase correction to channel 2
        fid_ch2_phased = fid_ch2 * np.exp(-1j * phase_diff)

        # Reconstruct phase-corrected FID data
        fid_phased = np.zeros_like(fid, dtype=np.complex64)
        fid_phased[::2, :] = fid_ch1
        fid_phased[1::2, :] = fid_ch2_phased

        # Optional plotting of phase correction results
        if plot:
            fig, ax = plt.subplots(2, 3, figsize=(8, 4))
            ax = ax.flatten()

            # Plot real parts of first acquisition_idx points for each channel
            ax[0].plot(np.real(fid[0, :]))
            ax[0].set_title("Ch1 Real FID")
            ax[1].plot(np.real(fid[1, :]))
            ax[1].set_title("Ch2 Real FID")

            ax[2].plot(np.real(fid[0, :] + fid[1, :]), ".-", label="Original")

            ax[3].plot(np.real(fid_phased[0, :]))
            ax[3].set_title("Ch1 Real FID (Phased)")
            ax[4].plot(np.real(fid_phased[1, :]))
            ax[4].set_title("Ch2 Real FID (Phased)")
            ax[5].plot(
                np.real(fid[0, :] + fid[1, :]),
                ".-",
                label="Ch1 + Ch2 Original",
                alpha=0.5,
                linewidth=4,
                color="gray",
            )
            ax[5].plot(
                np.real(fid_phased[0, :] + fid_phased[1, :]),
                ".-",
                label="Ch1 + Ch2 Phased",
            )
            ax[5].legend()
            ax[5].set_title("Comparison Combined Channels")

            plt.tight_layout()
            plt.show()
        # Use the new general-purpose channel combination utility
        from hypermri.utils.mri.channels import PhaseAlignedConfig, combine_channels

        # Get the original raw FID data (before reshaping)
        raw_fid = self.handler.rawdatajob0

        # Combine channels using phase-aligned method with the new utility
        # Note: The utility properly removes the channel_idx dimension
        if config is None:
            config = PhaseAlignedConfig(
                raw_fid_data=raw_fid,
                adjust_phase=adjust_phase,
                combination_method=(
                    channel_combination
                    if channel_combination in ["mean", "sum"]
                    else "mean"
                ),
            )

        # The phase correction is applied only here (reconstruction is linear, so
        # rotating channel 2 in image space equals rotating its raw FID)
        self.reco_data_combined_channels = combine_channels(
            data_array=input_data,
            method="phase_aligned",
            config=config,
        )

        # Apply the same channel 2 phase correction to the stored k-space. The phase
        # already applied is tracked in the attrs, so repeated calls don't stack up.
        kspace_xarray = getattr(self.handler, "kspace_xarray", None)
        if kspace_xarray is not None and "channel_idx" in kspace_xarray.dims:
            phase_deg = self.reco_data_combined_channels.attrs["phase_correction_deg"]
            delta_rad = np.deg2rad(
                phase_deg - kspace_xarray.attrs.get("phase_correction_deg", 0.0)
            )
            channel_axis = kspace_xarray.get_axis_num("channel_idx")
            channel_phase = np.ones(n_channels, dtype=complex)
            channel_phase[1] = np.exp(-1j * delta_rad)
            channel_phase = np.expand_dims(
                channel_phase,
                tuple(i for i in range(kspace_xarray.ndim) if i != channel_axis),
            )

            self.handler.kspace_xarray = kspace_xarray.copy(
                data=(kspace_xarray.values * channel_phase).astype(kspace_xarray.dtype)
            )
            self.handler.kspace_xarray.attrs["phase_correction_deg"] = phase_deg

        # Store the combination method in attributes
        combination_method = "mean" if channel_combination != "sum" else "sum"

        # The utility already adds metadata, but we'll ensure description is complete
        if "description" not in self.reco_data_combined_channels.attrs:
            phase_info = self.reco_data_combined_channels.attrs.get(
                "phase_correction_deg", 0.0
            )
            if adjust_phase:
                self.reco_data_combined_channels.attrs["description"] = (
                    f"Channels combined using '{combination_method}' after phase alignment. "
                    f"Phase correction of {phase_info:.2f}° applied to channel 2."
                )
            else:
                self.reco_data_combined_channels.attrs["description"] = (
                    f"Channels combined using '{combination_method}' without phase correction."
                )

        return self.reco_data_combined_channels


    # --> move away from CSI?


