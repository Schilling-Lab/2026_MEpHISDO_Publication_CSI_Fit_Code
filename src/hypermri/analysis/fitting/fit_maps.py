"""Voxel-wise fitting helpers for parametric MRI maps.

The module provides light-weight helper functions used by the ``hp`` xarray
accessor to run non-linear fits across volumes. Each helper accepts NumPy
arrays and returns dictionaries so that the accessor can easily assemble
``xarray`` datasets with parameter estimates.
"""


import numpy as np
import xarray as xr
from hypermri.utils.system.logging import init_default_logger

DB_MSG = True  # Change this to False to hide detailed error messages
DATA_PRECISION = np.float16


DEBUG_FILE = "/tmp/fit_maps_debug.log"
DEBUG_LOG = True


logger = init_default_logger(
    __name__, fstring="[%(levelname)s] %(name)s.%(funcName)s(): %(message)s "
)


@xr.register_dataset_accessor("hp")
class fit_accessor:
    """``xarray`` accessor that orchestrates model reconstructions."""

    def __init__(self, xarray_obj):
        """Attach the accessor to an :class:`xarray.Dataset`.

        Parameters
        ----------
        xarray_obj
            Dataset created by one of the fitting pipelines.
        """
        self._obj = xarray_obj
        self._model_reconstructors = {
            "FID": fid_model,
        }

    def reconstruct(self, voxel_coords: dict = None, **kwargs) -> xr.DataArray:
        """Reconstruct fitted signals for whole volumes or individual voxels.

        Parameters
        ----------
        voxel_coords
            Mapping of dimension names to indices that identify a single voxel.
            When ``None`` the reconstruction spans the full dataset.
        **kwargs
            Additional parameters required by the underlying model reconstructor,
            for example ``sampling_time_axis_s`` for the FID model.

        Returns
        -------
        xarray.DataArray
            Reconstructed signal curves. When ``voxel_coords`` are provided the
            array is one-dimensional across the dynamic axis; otherwise it matches
            the spatial layout of the original dataset.

        Raises
        ------
        NotImplementedError
            If the dataset advertises an unsupported model via ``model_name``.
        ValueError
            If the required auxiliary parameters are missing from the dataset
            attributes or the supplied keyword arguments.
        """
        model_name = self._obj.attrs.get("model_name")
        if not model_name or model_name not in self._model_reconstructors:
            raise NotImplementedError(
                f"Reconstruction for model '{model_name}' is not implemented or model_name is missing, available models: {list( self._model_reconstructors.keys())}"
            )

        reconstructor = self._model_reconstructors[model_name]

        # Select the data for the specific voxel if coordinates are provided
        data_to_reconstruct = (
            self._obj.isel(voxel_coords) if voxel_coords else self._obj
        )

        # --- Gather parameters from the selected dataset and kwargs ---
        params = {}
        dynamic_dim_attr = data_to_reconstruct.attrs.get("dynamic_dim")
        dynamic_dim_override = kwargs.get("dynamic_dim")

        if model_name == "FID":
            # Allow callers to override the sampling axis when reconstructing.
            sampling_time_axis = dynamic_dim_override
            if sampling_time_axis is None:
                sampling_time_axis = kwargs.get(
                    "sampling_time_axis_s",
                    kwargs.get(
                        "t", data_to_reconstruct.attrs.get("sampling_time_axis_s")
                    ),
                )

            if sampling_time_axis is None:
                raise ValueError(
                    "FID reconstruction requires `dynamic_dim` or `sampling_time_axis_s` via kwargs or dataset attrs"
                )

            # FID model parameters
            params = {
                "amplitude": data_to_reconstruct["fit_amps"],
                "freq": data_to_reconstruct["fit_freqs"],
                "T2_star": data_to_reconstruct["fit_t2s"],
                "t": np.asarray(sampling_time_axis),
                "db_mode": False,
            }

        return reconstructor(**params)


# -- FID model ---
def fid_model(
    amplitude: xr.DataArray,
    freq: xr.DataArray,
    T2_star: xr.DataArray,
    t: np.ndarray,
    db_mode: bool = False,
) -> xr.DataArray:
    """
    Calculate the Free Induction Decay (FID) signal for given frequencies and T2* relaxation times.
    """
    if db_mode:
        print("\n--- Entering fid_model ---")
        print(f"  Input 'amplitude' dims: {amplitude.dims}, shape: {amplitude.shape}")
        print(f"  Input 'freq' dims: {freq.dims}, shape: {freq.shape}")
        print(f"  Input 'T2_star' dims: {T2_star.dims}, shape: {T2_star.shape}")
        print(f"  Input 't' (numpy array) shape: {t.shape}")

    # Convert 't' (numpy array) to an xarray.DataArray with 'acquisition_idx'
    # dimension.
    t_da = xr.DataArray(t, dims=["acquisition_idx"], coords={"acquisition_idx": t})
    if db_mode:
        print(f"  Converted 't_da' dims: {t_da.dims}, shape: {t_da.shape}")

    # Explicitly broadcast 't_da' to match the dimensions of 'amplitude'.
    # This ensures 't_aligned' has all necessary dimensions for correct
    # alignment.
    t_aligned = t_da.broadcast_like(amplitude)
    if db_mode:
        print(
            f"  't_aligned' (broadcasted 't') dims: {t_aligned.dims}, shape: {t_aligned.shape}"
        )

    # Perform the FID calculation.
    # All operands are now xarray DataArrays with explicit named dimensions.
    sig = (
        amplitude
        * np.exp(1j * 2.0 * np.pi * freq * t_aligned)
        * np.exp(-t_aligned / T2_star)
    )

    if db_mode:
        print(
            f"  Output 'sig' dims: {sig.dims}, shape: {sig.shape}, dtype: {sig.dtype}"
        )
        print("--- Exiting fid_model ---\n")
    return sig
