"""
Denoising algorithms used across HyperMRI analysis workflows.

References
----------
.. [Veraart2016a] Veraart J, Novikov DS, Christiaens D, Ades-aron B, Sijbers,
       Fieremans E, 2016. Denoising of Diffusion MRI using random matrix
       theory. Neuroimage 142:394-406.
       doi: 10.1016/j.neuroimage.2016.08.016
.. [Veraart2016b] Veraart J, Fieremans E, Novikov DS. 2016. Diffusion MRI noise
       mapping using random matrix theory. Magnetic Resonance in Medicine.
       doi: 10.1002/mrm.26059.
.. [Manjon2013] Manjon JV, Coupe P, Concha L, Buades A, Collins DL (2013)
       Diffusion Weighted Image Denoising Using Overcomplete Local
       PCA. PLoS ONE 8(9): e73021.
       https://doi.org/10.1371/journal.pone.0073021
.. [Henriques2021] Henriques RN, Ianuş A, Novello L, Jovicich J, Jespersen SN, Shemesh N.
       Efficient PCA denoising of diffusion MRI data corrupted by spatially correlated
       noise.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.linalg import eigh, svd

from ...utils.system.logging import init_default_logger

logger = init_default_logger(__name__)
COORDINATE_PRECISION = np.float16  # Or np.float16, np.float64, etc.


# All code between the two lines below is copied from:
# References:
#   Henriques, R. N., et al. (2023). "Efficient PCA denoising of spatially
#   correlated redundant MRI data." Imaging Neuroscience 1: 1-26.
#   DOI: 10.1162/imag_a_00049
# Code available at:
#   git@github.com:RafaelNH/PCAdenoising.git


# ------------------------------PCAdenoising for xarrays------------------
# --------------------------------------------------------------------------------


def denoise_pca(
    input_xarray: xr.DataArray,
    components: Union[int, list, slice] = None,
    pca_method: str = "eig",
    plot: bool = False,
    db_mode: bool = False,
    spatial_dims_present: list = None,
    redundant_dims_present: list = None,
    noise_threshold: float = None,
    noise_estimation_method: str = "auto",
    treat_as_redundant: list = None,
) -> xr.DataArray:
    """
    Perform PCA-based denoising on a complex-valued xarray.DataArray.

    This function automatically identifies spatial and redundant dimensions,
    reshapes the data into a 2D matrix (pixels x repetitions), applies
    PCA denoising, and then reconstructs the data back into its original
    xarray structure, preserving all original coordinates and dimension orders.

    Parameters
    ----------
    input_xarray: xr.DataArray
        The input data with spatial and redundant dimensions.
    components: int, list, or slice, optional
        The principal components to keep for reconstruction.
        - If int: Keep the top N components (0 to N-1).
        - If list: Keep specific component indices (e.g., [0, 2, 4]).
        - If slice: Keep a range of components (e.g., slice(3, 5)).
        - If None: Automatically determine the number of components using TPCA threshold.
        Defaults to None.
    pca_method: str, optional
        The method for PCA ('eig' or 'svd'). eig' is faster, 'svd' may be more stable. Defaults to "eig".
    plot: bool, optional
        If True, displays the scree plot of eigenvalues. Defaults to False.
    db_mode: bool, optional
        If True, prints debug information. Defaults to False.
    spatial_dims_present: list, optional
        List of spatial dimensions to consider. If None, defaults to ["read_idx", "phase_idx", "slice_idx"].
        Defaults to None.
    redundant_dims_present: list, optional
        List of redundant dimensions for PCA. If None, it is automatically determined from the input data.
        Defaults to None.
    noise_threshold: float, optional
        Manual noise variance threshold for component selection. If None, noise is estimated automatically.
        Defaults to None.
    noise_estimation_method: str, optional
        Method for estimating noise variance. Defaults to 'auto'.
        Options:
        - 'auto' (corner-based with fallback),
        - 'corner' (corners only), 'last_reps' (last 10% of repetitions),
        - 'manual' (use provided noise_threshold).

    treat_as_redundant: list, optional
        List of dimension names to force-treat as redundant dimensions rather than spatial. Useful for
        spectral data where 'slice_idx' represents multiple acquisitions.
        Defaults to None.

    Returns
    -------
    pca_denoised array: xr.DataArray
        A new xarray.DataArray containing the denoised data.

    """
    # --- 1. Identify Spatial and Redundant Dimensions ---
    if db_mode:
        print("--- Step 1: Identifying Dimensions ---")

    SPATIAL_DIMS = ["read_idx", "phase_idx", "slice_idx"]

    # Handle treat_as_redundant parameter for spectral data
    if treat_as_redundant is None:
        treat_as_redundant = []

    if spatial_dims_present is None:
        # Exclude dimensions that should be treated as redundant
        spatial_dims_present = [
            dim
            for dim in SPATIAL_DIMS
            if dim in input_xarray.dims and dim not in treat_as_redundant
        ]

    if redundant_dims_present is None:
        # Include all non-spatial dims plus those forced to be redundant
        redundant_dims_present = [
            dim
            for dim in input_xarray.dims
            if dim not in spatial_dims_present or dim in treat_as_redundant
        ]

    # Validate dimensions
    if not spatial_dims_present:
        # For spectral-only data, we need at least acquisition_idx as "spatial" dimension
        if "acquisition_idx" in input_xarray.dims:
            spatial_dims_present = ["acquisition_idx"]
            redundant_dims_present = [
                dim for dim in input_xarray.dims if dim != "acquisition_idx"
            ]
            if db_mode:
                print(
                    "Note: Using 'acquisition_idx' as spatial dimension for spectral data."
                )
        else:
            raise ValueError(
                "No spatial or pixel dimensions found. "
                "For spectral data, 'acquisition_idx' is required. "
                "Available dimensions: " + str(list(input_xarray.dims))
            )

    if not redundant_dims_present:
        raise ValueError(
            "No redundant dimensions for PCA found. "
            "PCA denoising requires multiple measurements (e.g., repetitions, slices, echoes). "
            "Available dimensions: " + str(list(input_xarray.dims))
        )

    if db_mode:
        print(f"Spatial dimensions identified: {spatial_dims_present}")
        print(f"Redundant dimensions identified: {redundant_dims_present}")
        if treat_as_redundant:
            print(f"Dimensions forced to be redundant: {treat_as_redundant}")
        print()

    # --- 2. Reshape Data for PCA ---
    if db_mode:
        print("--- Step 2: Reshaping Data for PCA ---")

    original_dims = input_xarray.dims
    transposed_xarray = input_xarray.transpose(
        *redundant_dims_present, *spatial_dims_present
    )
    n_pixels = np.prod([transposed_xarray.sizes[dim] for dim in spatial_dims_present])
    n_repetitions = np.prod(
        [transposed_xarray.sizes[dim] for dim in redundant_dims_present]
    )
    X = transposed_xarray.values.reshape(n_repetitions, n_pixels).T

    # Check for NaNs or Infs
    if not np.all(np.isfinite(X)):
        msg = "Input data contains NaNs or Infs. Replacing with 0 for PCA denoising."
        if db_mode:
            print(f"Warning: {msg}")
        logger.warning(msg)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if db_mode:
        print(f"Shape of reshaped matrix X: {X.shape}\n")

    # --- 3. Center the Data ---
    mean_of_each_repetition = X.mean(axis=0)
    X_centered = X - mean_of_each_repetition

    # --- 4. Perform Eigendecomposition ---
    if db_mode:
        print(f"--- Step 4: Performing Eigendecomposition using '{pca_method}'---")
    M, N = X_centered.shape

    if pca_method.lower() == "eig":
        # Scale data to avoid overflow during matrix multiplication
        # covariance = (1/M) * (X.T @ X) = (X/sqrt(M)).T @ (X/sqrt(M))
        X_scaled = X_centered / np.sqrt(M)
        covariance_matrix = X_scaled.conj().T @ X_scaled
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    elif pca_method.lower() == "svd":
        # Scale data to avoid overflow during squared singular value calculation
        X_scaled = X_centered / np.sqrt(M)
        U, s, Vt = svd(X_scaled, full_matrices=False)
        eigenvalues = s**2
        eigenvectors = Vt.conj().T
    else:
        raise ValueError("pca_method should be either 'eig' or 'svd'")

    # Ensure eigenvalues is a numpy array (not xarray DataArray)
    if hasattr(eigenvalues, "values"):
        eigenvalues = eigenvalues.values

    if db_mode:
        print("Eigendecomposition complete.\n")

    # --- 5. Determine Components to Keep ---
    if db_mode:
        print("--- Step 5: Determining Components to Keep ---")

    # Indices of components to keep
    components_to_keep_indices = None

    if components is None:
        # Convert noise_threshold to scalar if it's an xarray or numpy array
        if noise_threshold is not None:
            if hasattr(noise_threshold, "values"):
                # xarray DataArray
                noise_threshold = float(noise_threshold.values)
            elif isinstance(noise_threshold, np.ndarray):
                # numpy array - take mean or first element
                noise_threshold = float(np.mean(noise_threshold))
            else:
                noise_threshold = float(noise_threshold)

        # Estimate noise variance based on method
        if noise_estimation_method == "manual":
            if noise_threshold is None:
                raise ValueError(
                    "noise_estimation_method='manual' requires noise_threshold to be provided."
                )
            noise_variance_prior = noise_threshold
            if db_mode:
                print(f"Using manual noise threshold: {noise_variance_prior:.4f}")

        elif noise_estimation_method == "corner":
            if db_mode:
                print("Estimating noise from corner pixels...")
            try:
                corner_noise = input_xarray.isel(
                    read_idx=[0, -1], phase_idx=[0, -1]
                ).values
                noise_variance_prior = np.var(corner_noise)
                if db_mode:
                    print(f"Corner-based noise variance: {noise_variance_prior:.4f}")
            except Exception as e:
                raise ValueError(
                    f"Corner-based noise estimation failed: {e}. "
                    "Try noise_estimation_method='last_reps' or provide noise_threshold."
                )

        elif noise_estimation_method == "last_reps":
            if db_mode:
                print("Estimating noise from last 10% of repetitions...")
            noise_variance_prior = np.var(X[:, -int(N * 0.1) :])
            if db_mode:
                print(f"Last-reps noise variance: {noise_variance_prior:.4f}")

        elif noise_estimation_method == "auto":
            if db_mode:
                print(
                    "Using automatic noise estimation (corner-based with fallback)..."
                )
            try:
                corner_noise = input_xarray.isel(
                    read_idx=[0, -1], phase_idx=[0, -1]
                ).values
                noise_variance_prior = np.var(corner_noise)
                if db_mode:
                    print(f"Corner-based noise variance: {noise_variance_prior:.4f}")
            except Exception:
                if db_mode:
                    print(
                        "Warning: Could not estimate noise from corners. Using last 10% of reps."
                    )
                noise_variance_prior = np.var(X[:, -int(N * 0.1) :])
                if db_mode:
                    print(f"Last-reps noise variance: {noise_variance_prior:.4f}")
        else:
            raise ValueError(
                f"Unknown noise_estimation_method: {noise_estimation_method}. "
                "Valid options: 'auto', 'corner', 'last_reps', 'manual'"
            )

        # Apply manual threshold override if provided
        if noise_threshold is not None and noise_estimation_method != "manual":
            if db_mode:
                print(
                    f"Overriding estimated noise ({noise_variance_prior:.4f}) "
                    f"with manual threshold ({noise_threshold:.4f})"
                )
            noise_variance_prior = noise_threshold

        # Calculate TPCA threshold
        gamma = n_pixels / n_repetitions
        lambda_t_tpca = (1 + np.sqrt(gamma)) ** 2 * noise_variance_prior
        n_components_to_keep = max(1, np.sum(eigenvalues > lambda_t_tpca))
        components_to_keep_indices = np.arange(n_components_to_keep)

        if db_mode:
            print(f"Noise variance estimate: {noise_variance_prior:.4f}")
            print(f"Gamma (pixels/repetitions): {gamma:.4f}")
            print(f"TPCA Threshold (lambda_t): {lambda_t_tpca:.4f}")
            print(f"Keeping top {n_components_to_keep} components.\n")

    elif isinstance(components, int):
        if db_mode:
            print(f"Keeping top {components} components (user-specified).")
        components_to_keep_indices = np.arange(components)
    elif isinstance(components, list):
        if db_mode:
            print(f"Keeping specific components: {components} (user-specified).")
        components_to_keep_indices = np.array(components)
    elif isinstance(components, slice):
        if db_mode:
            print(f"Keeping range of components: {components} (user-specified).")
        components_to_keep_indices = np.arange(n_repetitions)[components]
    else:
        raise ValueError(
            "components must be an int, list, slice, or None. "
            f"Got {type(components)}: {components}"
        )

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(eigenvalues.real, "o-")
        ax[0].set_title("Scree Plot: Eigenvalues")
        ax[0].set_xlabel("Principal Component Number")
        ax[0].set_ylabel("Eigenvalue (Variance Explained)")
        if components is None:
            ax[0].axhline(
                y=lambda_t_tpca,
                color="r",
                linestyle="--",
                label=f"TPCA Threshold ({lambda_t_tpca:.2f})",
            )

        # Highlight kept components
        for idx in components_to_keep_indices:
            if idx < len(eigenvalues):
                ax[0].plot(idx, eigenvalues[idx].real, "go")

        ax[0].legend()
        ax[1].plot(eigenvalues.real, "o-")
        ax[1].set_yscale("log")
        ax[1].set_title("Scree Plot (Log Scale)")
        ax[1].set_xlabel("Principal Component Number")
        ax[1].set_ylabel("Log(Eigenvalue)")
        if components is None:
            ax[1].axhline(y=lambda_t_tpca, color="r", linestyle="--")
        plt.tight_layout()
        plt.show()

    # --- 6. Denoise and Reconstruct Data ---
    if db_mode:
        print("--- Step 6: Denoising and Reconstructing ---")
    projected_data = X_centered @ eigenvectors
    projected_data_filtered = np.zeros_like(projected_data)

    # Keep only the selected components
    valid_indices = components_to_keep_indices[
        components_to_keep_indices < n_repetitions
    ]
    projected_data_filtered[:, valid_indices] = projected_data[:, valid_indices]

    X_denoised_centered = projected_data_filtered @ eigenvectors.conj().T
    X_denoised = X_denoised_centered + mean_of_each_repetition

    # --- 7. Reshape Data Back to Original Xarray Format ---
    if db_mode:
        print("--- Step 7: Reshaping Data to Original Format ---")

    denoised_multidim_array = X_denoised.T.reshape(transposed_xarray.shape)
    denoised_transposed_xarray = xr.DataArray(
        data=denoised_multidim_array,
        coords=transposed_xarray.coords,
        dims=transposed_xarray.dims,
    )
    input_xarray_dn = denoised_transposed_xarray.transpose(*original_dims)
    input_xarray_dn.attrs = input_xarray.attrs
    input_xarray_dn.name = input_xarray.name

    if db_mode:
        print("\n--- Denoising Complete ---")
        print(f"Shape of final denoised xarray: {input_xarray_dn.shape}\n")

    return input_xarray_dn
