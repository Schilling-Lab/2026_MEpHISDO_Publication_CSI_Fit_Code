"""
Plotter factory for easy access to plotters via the registry system.

This module provides a high-level interface for creating plotters based on
data type, sequence type, or explicit plotter selection.
"""

import logging
from typing import Any, Optional

import xarray as xr

from .registry import get_plotter_registry

logger = logging.getLogger(__name__)


def create_plotter(
    data: Optional[xr.DataArray] = None,
    sequence_obj=None,
    plotter_type: Optional[str] = None,
    category: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Create a plotter instance based on data type and preferences.

    This function provides a high-level interface to the plotter registry system,
    automatically selecting appropriate plotters based on data characteristics
    or explicit preferences.

    Parameters
    ----------
    data : xarray.DataArray, optional
        The data to be plotted (used for automatic plotter selection)
    sequence_obj : object, optional
        The sequence object that contains the data (e.g., FLASH, CSI instance)
    plotter_type : str, optional
        Explicit plotter type to use (e.g., 'anatomy', 'spectroscopy')
    category : str, optional
        Data category (e.g., 'anatomical', 'spectroscopy', 'quantitative')
    **kwargs
        Additional arguments passed to the plotter constructor

    Returns
    -------
    Plotter instance
        A plotter instance ready for plotting

    Examples
    --------
    # Explicit plotter selection with sequence object
    plotter = create_plotter(sequence_obj=flash_obj, plotter_type='anatomy')

    # Category-based selection
    plotter = create_plotter(sequence_obj=csi_obj, category='spectroscopy')

    # Automatic selection based on data
    plotter = create_plotter(data=anatomy_data, sequence_obj=flash_obj)
    """
    registry = get_plotter_registry()

    # 1. Explicit plotter type specified
    if plotter_type:
        plotter_class_or_instance = registry.get_plotter(
            plotter_type, sequence_obj, **kwargs
        )
        if plotter_class_or_instance:
            # Check if it's already instantiated or needs instantiation
            if hasattr(plotter_class_or_instance, "plot"):
                logger.debug(f"Created plotter from explicit type: {plotter_type}")
                return plotter_class_or_instance
            else:
                # It's a class, instantiate with sequence_obj
                if sequence_obj:
                    return plotter_class_or_instance(sequence_obj)
                logger.warning(
                    f"Plotter class {plotter_type} needs sequence_obj for instantiation"
                )

    # 2. Category-based selection
    if category:
        plotter_class_or_instance = registry.get_plotter_for_category(
            category, sequence_obj, **kwargs
        )
        if plotter_class_or_instance:
            # Check if it's already instantiated or needs instantiation
            if hasattr(plotter_class_or_instance, "plot"):
                logger.debug(f"Created plotter from category: {category}")
                return plotter_class_or_instance
            else:
                # It's a class, instantiate with sequence_obj
                if sequence_obj:
                    return plotter_class_or_instance(sequence_obj)
                logger.warning(
                    f"Plotter class for category {category} needs sequence_obj for instantiation"
                )

    # 3. Automatic selection based on data characteristics
    if data is not None:
        category = _infer_category_from_data(data, sequence_obj)
        if category:
            plotter_class_or_instance = registry.get_plotter_for_category(
                category, sequence_obj, **kwargs
            )
            if plotter_class_or_instance:
                # Check if it's already instantiated or needs instantiation
                if hasattr(plotter_class_or_instance, "plot"):
                    logger.debug(f"Created plotter from inferred category: {category}")
                    return plotter_class_or_instance
                else:
                    # It's a class, instantiate with sequence_obj
                    if sequence_obj:
                        return plotter_class_or_instance(sequence_obj)

    # 4. Fallback to unified plotter
    logger.info("Using fallback unified plotter")
    plotter_class_or_instance = registry.get_plotter("unified", sequence_obj, **kwargs)
    if plotter_class_or_instance:
        if hasattr(plotter_class_or_instance, "plot"):
            return plotter_class_or_instance
        else:
            # It's a class, instantiate with sequence_obj
            if sequence_obj:
                return plotter_class_or_instance(sequence_obj)

    # 5. Ultimate fallback - try to use anatomy plotter
    logger.warning("Attempting final fallback to anatomy")
    plotter_class_or_instance = registry.get_plotter("anatomy", sequence_obj, **kwargs)
    if plotter_class_or_instance:
        if hasattr(plotter_class_or_instance, "plot"):
            return plotter_class_or_instance
        else:
            # It's a class, instantiate with sequence_obj
            if sequence_obj:
                return plotter_class_or_instance(sequence_obj)

    raise ValueError("Could not create any plotter - no suitable options found")


def _infer_category_from_data(data: xr.DataArray, sequence_obj=None) -> Optional[str]:
    """
    Infer the appropriate plotter category from data characteristics.

    Parameters
    ----------
    data : xarray.DataArray
        The data to analyze
    sequence_obj : object, optional
        The sequence object for additional context

    Returns
    -------
    str or None
        The inferred category, or None if cannot determine
    """
    try:
        # Check if sequence object provides hints
        if sequence_obj:
            # Check for sequence type using registry patterns
            seq_name = getattr(
                sequence_obj, "__class__", type(sequence_obj)
            ).__name__.lower()

            # Spectroscopy sequences
            spectro_sequences = [
                "csi",
                "press",
                "steam",
                "cpmg",
                "singlepulse",
                "nspect",
                "gnspect",
                "ivrecovery",
                "semilaser",
            ]
            if any(seq in seq_name for seq in spectro_sequences):
                return "spectroscopy"

            # Quantitative sequences
            quant_sequences = ["umpiremge", "umpire"]
            if any(seq in seq_name for seq in quant_sequences):
                return "quantitative"

            # Most others are anatomical
            return "anatomical"

        # Analyze data dimensions and characteristics
        dims = data.dims

        # Spectroscopic data typically has acquisition dimension
        if "acquisition" in dims:
            return "spectroscopy"

        # CSI data has specific spatial + spectroscopic structure
        spatial_dims = [
            d for d in dims if d in ["x", "y", "z", "read", "phase", "slice"]
        ]
        if len(spatial_dims) >= 2 and "acquisition" in dims:
            return "spectroscopy"

        # Check for common spectroscopy attributes
        if hasattr(data, "attrs"):
            attrs = data.attrs
            if any(key in attrs for key in ["nucleus", "chemical_shift", "ppm"]):
                return "spectroscopy"

        # Default to anatomical for spatial data
        if len(spatial_dims) >= 2:
            return "anatomical"

    except Exception as e:
        logger.debug(f"Error inferring category from data: {e}")

    return None
