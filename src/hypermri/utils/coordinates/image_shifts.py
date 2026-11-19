"""
Image shift and transformation utilities.

This module contains functions for shifting and transforming medical images,
extracted from the monolithic utils_general.py as part of Phase 2 refactoring.
"""

import logging

import numpy as np
import xarray as xr

# Initialize logger without dependencies
logger = logging.getLogger(__name__)


def get_orientation_flip_multipliers(data_obj) -> np.ndarray:
    """
    Determine the sign multipliers for read/phase/slice axes based on patient and slice orientation.
    This function uses the patient position, slice orientation, read orientation, to determine of raw reco Bruker data
    needs to be flipped along any of the three axes to match a standard orientation (seq2d).

    Parameters
    ----------
    data_obj : object
        Data object providing patient_pos, slice_orientation, read_orientation, and spat_dim attributes.

    Returns
    -------
    np.ndarray
        Array of length 3 with values {1, -1} indicating whether each axis should be flipped.
        Order corresponds to (read, phase, slice).
    """
    try:
        patient_pos = getattr(data_obj, "patient_pos", None)
        slice_orientation = getattr(data_obj, "slice_orientation", None)
        read_orientation = getattr(data_obj, "read_orientation", None)
        spat_dim = getattr(data_obj, "spat_dim", None)
    except Exception:
        # Fallback to identity if attributes are missing
        return np.ones(3, dtype=int)

    read_mult = 1
    phase_mult = 1
    slice_mult = 1

    if patient_pos == "Head_Prone":
        if slice_orientation == "axial":
            if read_orientation == "L_R":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "A_P":
                read_mult = -1
                phase_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Prone' and slice_orientation '{slice_orientation}'. No flips applied."
                )
        elif slice_orientation == "sagittal":
            if read_orientation == "A_P":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "H_F":
                read_mult = -1
                phase_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Prone' and slice_orientation '{slice_orientation}'. No flips applied."
                )

        elif slice_orientation == "coronal":
            read_mult = -1
            phase_mult = -1
            if spat_dim == "3D":
                slice_mult = -1

    elif patient_pos == "Head_Supine":
        if slice_orientation == "axial":
            if read_orientation == "L_R":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "A_P":
                read_mult = -1
                phase_mult = -1
                slice_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Supine' and slice_orientation '{slice_orientation}'. No flips applied."
                )
        elif slice_orientation == "sagittal":
            if read_orientation == "H_F":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "A_P":
                read_mult = -1
                phase_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Supine' and slice_orientation '{slice_orientation}'. No flips applied."
                )
        elif slice_orientation == "coronal":
            if read_orientation in ("H_F", "L_R"):
                read_mult = -1
                phase_mult = -1
            else:
                Warning(
                    f"Unexpected read_orientation '{read_orientation}' for patient_pos 'Head_Supine' and slice_orientation '{slice_orientation}'. No flips applied."
                )

    elif patient_pos == "Foot_Prone":
        Warning(
            "Patient position 'Foot_Prone' not fully tested, please verify results!"
        )
        if slice_orientation == "axial":
            if read_orientation in ("L_R", "A_P"):
                read_mult = -1
                phase_mult = -1
                slice_mult = -1
        elif slice_orientation == "sagittal":
            if read_orientation == "H_F":
                read_mult = -1
                phase_mult = -1
            elif read_orientation == "A_P":
                read_mult = -1
        elif slice_orientation == "coronal":
            if read_orientation in ("H_F", "L_R"):
                read_mult = -1
                phase_mult = -1

    return np.array([read_mult, phase_mult, slice_mult], dtype=int)


def shift_image(
    input_data=None,
    shift_vox=None,
    shift_method="phase",
    force_shift=False,
    data_obj=None,
    domain="kspace",
):
    """
    Shift the input data using either a phase ramp or a direct spatial shift.

    This function modifies the data using either a phase ramp or a direct spatial shift via the
    scipy.ndimage.shift function, based on the specified method. The input_data is expected to be
    in the k-space domain (default) and can be changed with the domain parameter.

    Parameters
    ----------
    input_data : ndarray or xarray.DataArray, optional
        The input data array. If None and `data_obj` is provided, the k-space data from
        `data_obj.kspace_xarray` is used. If None and `data_obj` is also None, an error is raised.
    shift_vox : array_like of float, optional
        The number of voxels to shift along each axis. If None, the shift is determined by the
        `determine_necessary_shift` method (only applicable when `data_obj` is provided).
        If `input_data` is a numpy array and `data_obj` is None, this parameter is required
        and must have the same number of dimensions as `input_data`.
    shift_method : {'phase', 'scipy'}, default 'phase'
        The method used to perform the shift. 'phase' uses a linear phase ramp, while 'scipy' uses
        scipy's shift function.
    force_shift : bool, default False
        If True, forces the shift operation even if shifts have been previously applied.
    domain : {'kspace', 'image'}, default 'kspace'
        The domain in which the shift should be applied. The input data will be transformed
        to this domain if necessary and transformed back afterwards.
    data_obj : object, optional
        An object (e.g., hypemri.sequence object) that might contain k-space data or information
        needed to determine the shift.

    Returns
    -------
    ndarray or xarray.DataArray
        The shifted data array.

    Raises
    ------
    ValueError
        If an unsupported shift method is provided.
        If `input_data` is None and `data_obj` is also None.
        If `input_data` is a numpy array, `data_obj` is None, and `shift_vox` is None.
        If `input_data` is a numpy array, `data_obj` is None, and the number of dimensions of
        `shift_vox` does not match the number of dimensions of `input_data`.
    ImportError
        If `shift_method` is 'scipy' and scipy is not installed.

    Notes
    -----
    The shift is applied in the specified domain. The input_data will be transformed to this
    domain if it is not already in it, and transformed back to the original domain afterwards.
    """
    if input_data is None:
        if data_obj is None:
            raise ValueError("Either input_data or data_obj must be provided.")
        if hasattr(data_obj, "kspace_xarray"):
            kspace_xarray = data_obj.kspace_xarray
            input_is_xarray = True
            data_np = kspace_xarray.to_numpy()
        else:
            raise ValueError("data_obj does not have the attribute kspace_xarray.")
    elif isinstance(input_data, xr.DataArray):
        kspace_xarray = input_data
        input_is_xarray = True
        data_np = kspace_xarray.to_numpy()
    elif isinstance(input_data, np.ndarray):
        data_np = input_data
        input_is_xarray = False
        if data_obj is None:
            if shift_vox is None:
                raise ValueError(
                    "shift_vox must be provided when input_data is a numpy array and data_obj is None."
                )
            if len(shift_vox) != data_np.ndim:
                raise ValueError(
                    "The number of dimensions of shift_vox must match the number of dimensions of input_data when data_obj is None."
                )
    else:
        raise ValueError(
            "Unsupported type for input_data. Must be ndarray or xarray.DataArray."
        )

    if input_is_xarray and "domain" in kspace_xarray.attrs:
        domain = kspace_xarray.attrs["domain"]

    if input_is_xarray:
        read_axis = kspace_xarray.get_axis_num("read_idx")
        phase_axis = kspace_xarray.get_axis_num("phase_idx")
        if kspace_xarray.spat_dim == "2D":
            slice_axis = None
        elif kspace_xarray.spat_dim == "3D":
            slice_axis = kspace_xarray.get_axis_num("slice_idx")
        else:
            slice_axis = None

        if slice_axis is not None:
            spatial_axes = (read_axis, phase_axis, slice_axis)
        else:
            spatial_axes = (read_axis, phase_axis)
        is_complex = np.iscomplexobj(data_np)

        num_image_fov_shift = kspace_xarray.attrs.get("num_image_fov_shift", 0)
        if num_image_fov_shift > 0 and not force_shift:
            return kspace_xarray

        if shift_vox is None:
            try:
                # Determine debug mode from log_mode
                shift_rps_vox, _ = determine_FOV_shift(
                    data_obj=data_obj,
                )

            except Exception as e:
                raise ValueError(
                    "Error determining FOV shift: "
                    f"{ str(e)}. Please provide shift_vox or ensure data_obj is valid."
                )
        else:
            shift_rps_vox = shift_vox

    else:  # numpy array
        if shift_vox is None:
            raise ValueError(
                "shift_vox must be provided for numpy array input when data_obj is None."
            )
        if len(shift_vox) != data_np.ndim:
            raise ValueError(
                "The length of shift_vox must match the number of dimensions of input_data."
            )
        spatial_axes = tuple(range(data_np.ndim - len(shift_vox), data_np.ndim))
        is_complex = np.iscomplexobj(data_np)
        shift_rps_vox = shift_vox

    # Determine if this is 3D data for debugging
    is_3d = (
        input_is_xarray
        and hasattr(kspace_xarray, "spat_dim")
        and kspace_xarray.spat_dim == "3D"
    )
    logger.debug(
        f"Shifting {'3D' if is_3d else '2D'} data with shape {data_np.shape}, "
        f"shift_rps_vox: {shift_rps_vox}, shift_method: {shift_method}, domain: {domain}"
    )
    if is_3d:
        logger.debug(f"3D spatial axes: {spatial_axes} (read, phase, slice)")
    else:
        logger.debug(f"2D spatial axes: {spatial_axes} (read, phase)")
    # Transform to the specified domain
    shifted_data = data_np.copy()
    if np.all(np.iscomplex(shifted_data)):
        pass
    else:
        shifted_data = shifted_data.astype(complex)

    # Apply the shift
    if shift_method == "scipy":
        from scipy.ndimage import shift

        if domain == "kspace":
            for axis in spatial_axes:
                shifted_data = np.fft.fftshift(shifted_data, axes=axis)
                shifted_data = np.fft.fft(shifted_data, axis=axis)
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)

        # perform shift:
        real_shifted = shift(np.real(shifted_data), shift_rps_vox, mode="wrap")
        imag_shifted = shift(np.imag(shifted_data), shift_rps_vox, mode="wrap")
        shifted_data = real_shifted + 1j * imag_shifted

        if domain == "kspace":
            # Transform back to k-space
            for axis in spatial_axes:
                shifted_data = np.fft.fftshift(shifted_data, axes=axis)
                shifted_data = np.fft.ifft(shifted_data, axis=axis)
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)
        else:
            # perform shift:
            real_shifted = shift(np.real(shifted_data), shift_rps_vox, mode="wrap")
            imag_shifted = shift(np.imag(shifted_data), shift_rps_vox, mode="wrap")
            shifted_data = real_shifted + 1j * imag_shifted

    elif shift_method == "phase":
        if domain != "kspace":
            # Transform to k-space to apply phase shift
            for axis in spatial_axes:
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)
                shifted_data = np.fft.ifft(shifted_data, axis=axis)
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)
        else:
            pass

        grid = np.meshgrid(*[np.arange(s) for s in shifted_data.shape], indexing="ij")
        for i, axis in enumerate(spatial_axes):
            axis_names = ["read", "phase", "slice"]
            axis_name = axis_names[i] if i < len(axis_names) else f"axis_{i}"
            logger.debug(
                f"Applying phase shift on {axis_name} axis {axis} with shift {shift_rps_vox[i]} voxels"
            )
            shift = shift_rps_vox[i]
            if shift == 0:
                continue
            size = shifted_data.shape[axis]
            phase_ramp = np.exp(1j * 2 * np.pi * grid[axis] / size * shift)
            shifted_data *= phase_ramp

        # Transform back to image domain
        if domain != "kspace":
            for axis in spatial_axes:
                shifted_data = np.fft.fftshift(shifted_data, axes=axis)
                shifted_data = np.fft.fft(shifted_data, axis=axis)
                shifted_data = np.fft.ifftshift(shifted_data, axes=axis)
        else:
            pass
    else:
        raise ValueError(
            f"Unsupported shift method: {shift_method}, use either 'phase' or 'scipy'."
        )

    if input_is_xarray:
        if not is_complex:
            shifted_data = np.abs(shifted_data)
        kspace_xarray.data = shifted_data
        if not hasattr(kspace_xarray.attrs, "num_image_fov_shift"):
            kspace_xarray.attrs["num_image_fov_shift"] = 0
        kspace_xarray.attrs["num_image_fov_shift"] += 1
        return kspace_xarray
    else:
        if not is_complex:
            shifted_data = np.abs(shifted_data)
        return shifted_data


def determine_FOV_shift(
    data_obj=None,
):
    """
    Calculate the necessary voxel and millimeter shifts required to align the MRI image
    based on the patient's position [Phase/Read/Slice Offset] and the scan orientation.

    Parameters
    ----------
    data_obj : object, optional
        Data object containing scan parameters (read_offset_mm, phase_offset_mm, slice_offset_mm, resolution_mm)

    Returns
    -------
    tuple
        (shift_rps_vox, shift_rps_mm) - shift in voxels and millimeters for read, phase, and slice directions
    """
    # Set logger level based on log_mode if provided
    if data_obj is None:
        logger.critical("data_obj is None, returning zero shifts")
        return np.array([0, 0]), np.array([0.0, 0.0])

    # Get offsets and resolution from data object
    try:
        read_offset_mm = data_obj.read_offset_mm
        phase_offset_mm = data_obj.phase_offset_mm

        # Try to get slice offset, default to 0 if not available
        slice_offset_mm = getattr(data_obj, "slice_offset_mm", 0)
        resolution_mm = data_obj.resolution_mm
    except AttributeError as e:
        logger.warning(
            f"Missing required attribute in data_obj: {e}, returning zero shifts"
        )
        return np.array([0, 0]), np.array([0.0, 0.0])

    if data_obj.type == "bruker":
        pass

    # Calculate shifts in mm and voxels
    shift_rps_mm = [read_offset_mm, phase_offset_mm, slice_offset_mm]
    shift_rps_vox = [0, 0, 0]

    # Check if this is a CSI sequence to determine read direction shifting
    is_csi = False
    if hasattr(data_obj, "type") and data_obj.type:
        # Check sequence type string (case insensitive)
        sequence_type = str(data_obj.type).lower()
        is_csi = "csi" in sequence_type
        logger.debug(f"Sequence type: {data_obj.type}, is_csi: {is_csi}")
    elif hasattr(data_obj, "__class__"):
        # Check class name if type attribute not available
        class_name = data_obj.__class__.__name__.lower()
        is_csi = "csi" in class_name
        logger.debug(f"Class name: {data_obj.__class__.__name__}, is_csi: {is_csi}")
    else:
        logger.debug("Could not determine sequence type, assuming non-CSI")

    # Convert mm shifts to voxel shifts (note the negative sign for proper alignment)
    # Read direction: only apply shift for CSI sequences
    shift_rps_vox[0] = (
        -shift_rps_mm[0] / resolution_mm[0] if is_csi else 0
    )  # read direction
    shift_rps_vox[1] = shift_rps_mm[1] / resolution_mm[1]  # phase direction
    shift_rps_vox[2] = (
        -shift_rps_mm[2] / resolution_mm[2] if len(resolution_mm) > 2 else 0
    )  # slice direction

    if is_csi:
        logger.debug("CSI sequence detected: applying read direction shift")
    else:
        logger.debug("Non-CSI sequence: read direction shift set to 0")

    # Output debug information based on db flag and log_mode
    flip_multipliers = get_orientation_flip_multipliers(data_obj)
    # Apply orientation multipliers to both voxel and mm shifts
    shift_rps_vox = np.multiply(shift_rps_vox, flip_multipliers).astype(float)
    shift_rps_mm = np.multiply(shift_rps_mm, flip_multipliers).astype(float)

    logger.info(f"shift_rps_vox: {shift_rps_vox}")
    logger.info(f"shift_rps_mm: {shift_rps_mm}")

    # For 3D data, return all 3 components; for 2D data, return only read/phase
    if hasattr(data_obj, "spat_dim") and data_obj.spat_dim == "3D":
        return shift_rps_vox, shift_rps_mm
    else:
        return shift_rps_vox[:2], shift_rps_mm[:2]


