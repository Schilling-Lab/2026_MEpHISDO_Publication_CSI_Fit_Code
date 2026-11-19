"""
Unified plotting interfaces for hypermri visualization system.

This module provides a consolidated plotting interface that unifies the various
plotter abstractions found throughout the codebase, following the registry pattern
established in Phase 1 of the refactoring.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import xarray as xr
from IPython.display import display


class BasePlotter(ABC):
    """
    Abstract base class for all plotters in the hypermri system.

    This class consolidates the functionality from the various plotter base classes
    and provides a unified interface following the registry pattern.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plotter with optional configuration.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary for plotter-specific settings
        """
        self.config = config or {}
        self._denoised_cache = {}
        self._last_original_data_id = None
        self.state = []
        self.widgets = {}
        self.plot_handles = None
        self.fig = None
        self.ax = None

    @abstractmethod
    def plot(self, data: xr.DataArray, **kwargs) -> Any:
        """
        Create a plot of the provided data.

        Parameters
        ----------
        data : xarray.DataArray
            The data to plot
        **kwargs
            Additional plotting parameters

        Returns
        -------
        Any
            Plot handles or figure reference
        """


    def update(self) -> None:
        """
        Update the plot display.

        Default implementation does nothing. Subclasses should override
        to define how the plot updates when parameters change.
        """

    def show(self) -> None:
        """
        Display the plot.

        Default implementation shows the matplotlib figure.
        """
        if self.fig is not None:
            display(self.fig)
            plt.show()


